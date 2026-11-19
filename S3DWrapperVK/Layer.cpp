/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Instance and device chaining and proc-addr resolution. Hooks are Hook_<vk name without prefix>;
// everything else resolves straight through the next layer, at no cost to the game.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include "../S3DAPI/ReadData.h"
#include "../S3DAPI/KeyboardHook.h"
#include "../Shared/OutputModeNames.h"
#include <string.h>
#include <algorithm>

namespace wiz3d { namespace vk {

static std::mutex s_lock;
static std::unordered_map<void*, InstanceData*> s_instances;   // dispatch key -> data
static std::unordered_map<void*, DeviceData*>   s_devices;     // dispatch key -> data

InstanceData* FindInstance(const void* dispatchable)
{
    std::lock_guard<std::mutex> g(s_lock);
    auto it = s_instances.find(DispatchKey(dispatchable));
    return it == s_instances.end() ? nullptr : it->second;
}

std::atomic<uint64_t> g_deviceGen{1};

// Every hook starts here, from every recording thread: each thread remembers its last answer instead of taking
// the global mutex per call.
DeviceData* FindDevice(const void* dispatchable)
{
    struct Cache { void* key; DeviceData* d; uint64_t gen; };
    thread_local Cache c = {};
    void* key = DispatchKey(dispatchable);
    const uint64_t gen = g_deviceGen.load(std::memory_order_acquire);
    if (c.key == key && c.gen == gen) return c.d;
    std::lock_guard<std::mutex> g(s_lock);
    auto it = s_devices.find(key);
    c.key = key; c.gen = gen; c.d = it == s_devices.end() ? nullptr : it->second;
    return c.d;
}

// The loader hands the next layer's entry points in a VkLayer*CreateInfo node
// chained to pNext; each layer advances the node past itself before calling down.
template <typename T, VkStructureType S>
static T* FindLayerLinkInfo(const void* pNext, VkLayerFunction func)
{
    for (auto p = (const VkBaseInStructure*)pNext; p; p = p->pNext)
    {
        if (p->sType == S && ((const T*)p)->function == func)
            return (T*)p;
    }
    return nullptr;
}

void LogFrameStats(DeviceData* d)
{
    Log("[frame %u] gameDraws=%u (inStereoPass %u) | passes stereo=%u mono=%u (dynamic %u: stereo=%u mono=%u mixed=%u suspended=%u) replayDraws=%u (unshiftedVS %u) replayDispatch=%u secondary=%u eyeBinds=%u unpatchedDraws=%u"
        " | smallUnshift=%u offAspect=%u served=%u fallback=%u | shaders patched=%u byConstants=%u fsEyeOrigin=%u (%u products) fsCamera=%u (%u windows) spritePipes=%u posVaryings=%u prevClip=%u remapCS=%u (right runs %u) screenRead fs=%u pipes=%u skipped=%u vsReverted=%u images doubled=%u shadowSets=%u aliases=%u mirrors=%u\n",
        d->frames, d->statGameDraws, d->statGameDrawsInStereoPass, d->statStereoPasses, d->statMonoPasses,
        d->statDynRendering, d->statDynStereo, d->statDynMono, d->statDynMixed, d->statDynSuspended, d->statReplayDraws.load(), d->statUnshiftedDraws, d->statReplayDispatch.load(),
        d->statSecondaryPasses, d->statEyeBinds, d->statUnpatchedDraws, d->statSmallUnshift, d->statOffAspectDraws, d->statServedDraws, d->statFallbackDraws,
        d->statShadersPatched, d->statConstantsShaders, d->statFsEyeOrigin, d->statFsProducts, d->statFsCamera, d->statFsCameraWindows, d->statSpritePipes, d->statPosVaryingShaders, d->statPrevClipShaders, d->statRemapShaders, d->statRemapRight, d->statScreenReadFs, d->statScreenReadPipes, d->statShadersSkipped, d->statVSReverted, d->statImagesDoubled, d->statShadowSets,
        d->statAliases, d->statMirrors);
    FamiliesLogFrame(*d);
    if (d->camCounters)
    {
        // Right-eye fragments (1 in 64x64) per window that passed the camera test; [6], [7] stay for the sprite VS.
        uint32_t c[8];
        memcpy(c, d->camCounters, sizeof(c));
        memset(d->camCounters, 0, 6 * sizeof(uint32_t));
        float n32, n33;
        memcpy(&n32, &c[6], 4); memcpy(&n33, &c[7], 4);
        Log("[fscam %u] right-eye samples: camera=%u notCamera=%u | inverse row 3 z=%g w=%g\n", d->frames, c[1], c[2], n32, n33);
    }
    Log("[frame %u] mesh draws=%u (%s) mesh pipelines=%u | pipelines with SPIR-V we never saw=%u | conditional rendering begins=%u (in recorded passes %u)\n",
        d->frames, d->statMeshDraws, Vk().VKMeshDraws ? "replayed" : "not replayed", d->statMeshPipelines, d->statUnseenPipelines,
        d->statCondBegins, d->statCondInPass);
    d->statMeshDraws = d->statMeshPipelines = d->statUnseenPipelines = 0;
    d->statCondBegins = d->statCondInPass = 0;
    if (Vk().VKOcclusionBothEyes)
        Log("[occlusion %u] segments in stereo passes=%u, begun in the right eye=%u | slots read=%u, seen by the right eye alone=%u | GPU copies (left only)=%u"
            " | read hook %.3f ms total, twin slots waited for=%u\n",
            d->frames, d->statOccRecorded.exchange(0), d->statOccRight.exchange(0), d->statOccReads.exchange(0), d->statOccRightOnly.exchange(0),
            d->statOccGpuCopies.exchange(0), PerfMs(d->occReadTicks.exchange(0)), d->statOccWaits.exchange(0));
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        // The doubled images buffer copies landed in most this window: size, format, copies.
        std::vector<std::pair<unsigned, VkImage>> top;
        for (const auto& kv : d->bufferToDoubledImages) top.push_back({ kv.second, kv.first });
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        char line[512] = {}; size_t len = 0;
        for (size_t k = 0; k < top.size() && k < 6; ++k)
        {
            auto it = d->images.find(top[k].second);
            if (it == d->images.end()) continue;
            len += (size_t)snprintf(line + len, sizeof(line) > len ? sizeof(line) - len : 0, " %ux%u f%d x%u",
                                    it->second.ci.extent.width, it->second.ci.extent.height, (int)it->second.ci.format, top[k].first);
        }
        Log("[frame %u] left-in-right binds=%u | doubled->buffer copies=%u buffer->doubled copies=%u (into:%s)\n",
            d->frames, d->statLeftInRight.exchange(0), d->statDoubledToBuffer, d->statBufferToDoubled, line);
        d->statDoubledToBuffer = d->statBufferToDoubled = 0;
        d->bufferToDoubledImages.clear();
    }
    Log("[frame %u] replay sets: shadowed=%u unshadowed=%u unknown=%u\n", d->frames, d->statReplayShadowed.load(), d->statReplayUnshadowed.load(),
        d->statReplayUnknownSet.load());
    d->statReplayShadowed = d->statReplayUnshadowed = d->statReplayUnknownSet = 0;
    Log("[frame %u] constants: inlineBlocks=%u (%u bytes) pushConst=%u (%u bytes) updateBuffer=%u (%u bytes) pushSets=%u\n",
        d->frames, d->statInlineBlocks, d->statInlineBytes, d->statPushConst, d->statPushConstBytes,
        d->statUpdateBuf, d->statUpdateBufBytes, d->statPushSets);
    if (Vk().VKBufferTwins)
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        Log("[twin %u] buffers twinned=%u (%llu MB bound) failed=%u | sparse binds mirrored=%u | right dispatches=%u scratch slots=%u untwinned writes=%u twin args=%u | indirect draws from left=%u direct=%u (%zu VS)\n",
            d->frames, d->statTwins, d->twinBytes >> 20, d->statTwinFailures, d->statSparseBindsMirrored,
            d->statRightDispatches, d->statScratchSlots, d->statUntwinnedWrites, d->statTwinArgs, d->statIndirectFromLeft,
            d->statDirectFromLeft, d->fromLeftVs.size());
        d->statSparseBindsMirrored = d->statRightDispatches = d->statScratchSlots = d->statUntwinnedWrites = d->statTwinArgs = 0;
        d->statIndirectFromLeft = d->statDirectFromLeft = 0;
    }
    d->statStereoPasses = d->statMonoPasses = d->statReplayDraws = d->statReplayDispatch = d->statDynRendering = 0;
    d->statRemapRight = 0;
    d->statDynStereo = d->statDynMono = d->statDynMixed = d->statDynSuspended = 0;
    d->statGameDraws = d->statGameDrawsInStereoPass = 0;
    d->statSecondaryPasses = d->statEyeBinds = d->statUnpatchedDraws = d->statUnshiftedDraws = d->statSmallUnshift = d->statOffAspectDraws = 0;
    d->statScreenTriLeft = d->statScreenTriRefused = 0;
    d->statServedDraws = d->statFallbackDraws = 0;
    d->statInlineBlocks = d->statInlineBytes = d->statPushConst = d->statPushConstBytes = d->statUpdateBuf = d->statUpdateBufBytes = d->statPushSets = 0;

    // Doubled images whose sibling nothing has written: the right eye reads cleared memory there
    // (compute-only targets show up too, as dispatch doubling does not mark its outputs).
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        unsigned blank = 0, doubled = 0, shown = 0, idle = 0;
        for (const auto& kv : d->images)
        {
            if (!kv.second.sibling) continue;
            ++doubled;
            if (kv.second.siblingWritten) continue;
            // Never written by the game either: doubled but unused, not a gap.
            if (!kv.second.everWritten) { ++idle; continue; }
            ++blank;
            if (shown < 10)
            {
                ++shown;
                const VkImageCreateInfo& ci = kv.second.ci;
                Log("      [blank sibling] %ux%u fmt=%u mips=%u samples=%u usage=0x%x\n",
                    ci.extent.width, ci.extent.height, (unsigned)ci.format, ci.mipLevels, (unsigned)ci.samples, ci.usage);
            }
        }
        Log("[img %u] doubled live=%u, idle=%u, game writes but sibling blank=%u\n", d->frames, doubled, idle, blank);
        // Who carries the eye, per matrix-positioned pipeline: mixed = some of its ranges unpatched (those draws bind
        // sep=0 with no patch, so they read mono); unserved = fell back to the self-shift.
        unsigned pServed = 0, pUnserved = 0, pMixed = 0;
        for (const auto& kv : d->pipelines)
        {
            if (!kv.second.posUsesMatrix) continue;
            if (kv.second.served) { ++pServed; if (kv.second.servedSkipped * 10 > kv.second.servedApplied) ++pMixed; }
            else ++pUnserved;
        }
        Log("[pipe %u] matrix pipelines: served=%u (mixed %u) unserved=%u\n", d->frames, pServed, pMixed, pUnserved);

        // Sibling memory against the game's: how many images the game binds per allocation (aliasing),
        // and what our sibling allocations hold.
        unsigned mems = 0, aliased = 0, maxBinds = 0, boundImages = 0;
        unsigned long long liveGame = 0;
        for (const auto& kv : d->memory)
        {
            ++mems;
            liveGame += kv.second.size;
            boundImages += kv.second.imageBinds;
            if (kv.second.imageBinds > 1) ++aliased;
            if (kv.second.imageBinds > maxBinds) maxBinds = kv.second.imageBinds;
        }
        unsigned long long sibBytes = 0;
        unsigned sibs = 0;
        for (const auto& kv : d->images)
            if (kv.second.sibling && kv.second.siblingMemSize) { ++sibs; sibBytes += kv.second.siblingMemSize; }
        Log("[mem %u] game allocs=%u (dedicated %u) %llu MB cumulative, %u live = %llu MB | images bound=%u across %u allocs,"
            " aliased allocs=%u (max %u images in one) | our sibling allocs=%u cumulative %llu MB, live %u = %llu MB\n",
            d->frames, d->statGameAllocs, d->statDedicatedAllocs, d->statGameAllocBytes / 1048576ull,
            mems, liveGame / 1048576ull, boundImages, mems, aliased, maxBinds,
            d->statSibAllocs, d->statSibAllocBytes / 1048576ull, sibs, sibBytes / 1048576ull);
    }

    ConstantsLogStats(d);

    PerfCounters& p = d->perf;
    const size_t n = p.frameMs.size();
    double avg = 0.0, p95 = 0.0;
    if (n)
    {
        std::vector<float> sorted(p.frameMs);
        std::sort(sorted.begin(), sorted.end());
        for (float v : sorted) avg += v;
        avg /= (double)n;
        size_t k = (size_t)(n * 0.95);
        p95 = sorted[k < n ? k : n - 1];
    }
    const double per = n ? 1.0 / (double)n : 0.0;   // per-frame averages over the window
    auto ms = [&](std::atomic<long long>& a) { return PerfMs(a.exchange(0)) * per; };
    const double draws = ms(p.draws), binds = ms(p.binds), replay = ms(p.replay), descriptors = ms(p.descriptors), constants = ms(p.constants);
    const double descFree = ms(p.descFree);
    Log("[perf %u] frame avg=%.2f p95=%.2f ms (%.1f fps) | layer cpu/frame: draws=%.2f binds=%.2f replay=%.2f descriptors=%.2f (lookup %.2f, shadowCreate %.2f, write %.2f) free=%.2f constants=%.2f total=%.2f"
        " | composite=%.2f weave A=%.2f B=%.2f C=%.2f | vram added=%.0f MB of game %.0f MB\n",
        d->frames, avg, p95, avg > 0.0 ? 1000.0 / avg : 0.0, draws, binds, replay, descriptors, ms(p.descLookup), ms(p.descCreate), ms(p.descWrite), descFree, constants,
        draws + binds + replay + descriptors + descFree + constants,
        ms(p.composite), ms(p.weaveA), ms(p.weaveB), ms(p.weaveC),
        p.vramAdded.load() / 1048576.0, p.vramGame.load() / 1048576.0);
    // Held time is the serialised critical path, the rest of descLookup and constants waits for it;
    // constants total = wait + held + sort + apply.
    Log("[perf %u] lock held/frame: descriptors=%.2f constants=%.2f | wait: descriptors=%.2f constants=%.2f | constApply=%.2f"
        " | descriptor traffic/frame: writes=%.0f calls=%.0f templates=%.0f\n",
        d->frames, ms(p.descHeld), ms(p.constHeld), ms(p.descWaitW), ms(p.constWait), ms(p.constApply),
        d->statDescWrites.exchange(0) * per, d->statDescCalls.exchange(0) * per, d->statTemplateCalls * per);
    // constApply broken out: read is the uncached ReBAR fetch, math is
    // ApplyCamera, key is the learned-map lookup, rec is the GPU transport.
    Log("[perf %u] constApply parts/frame: read=%.2f math=%.2f key=%.2f rec=%.2f\n",
        d->frames, ms(p.constRead), ms(p.constMath), ms(p.constKey), ms(p.constRec));
    // The exclusive holders the game's descriptor writers wait behind (sharedCollect moves the collect off them).
    Log("[perf %u] exclusive holds/frame: setAllocate=%.2f setFree/poolReset=%.2f collect=%s\n",
        d->frames, ms(p.allocHeld), ms(p.freeHeld), Traits().sharedCollect ? "shared" : "exclusive");
    d->statTemplateCalls = 0;
    p.frameMs.clear();
}

// Instance
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance)
{
    // The profile's switches latch here, before any is read; presets come from the device-time read as before.
    static bool s_switchesRead = false;
    if (!s_switchesRead)
    {
        s_switchesRead = true;
        const DataInput keep = gInfo.Input;
        ReadCurrentProfile(0);
        gInfo.Input = keep;
        LoadVkSwitches();
        LoadFamilies();
        LogVkSwitches();
    }
    // pinLayer / VKPinLayer: hosts that destroy their instance between boots would unload the layer under its own
    // threads; it stays loaded for the process's life.
    if (Traits().pinLayer || Vk().VKPinLayer)
    {
        HMODULE self = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN, (LPCWSTR)&Hook_CreateInstance, &self);
    }
    auto* link = FindLayerLinkInfo<VkLayerInstanceCreateInfo, VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO>(
        pCreateInfo->pNext, VK_LAYER_LINK_INFO);
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr nextGIPA = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;

    auto nextCreate = (PFN_vkCreateInstance)nextGIPA(VK_NULL_HANDLE, "vkCreateInstance");
    if (!nextCreate) return VK_ERROR_INITIALIZATION_FAILED;

    // Games that ask for 1.1 but resolve 1.2/1.3 core functions fail under a spec-strict layer below us: the promise
    // is raised to 1.3, which enables nothing by itself.
    VkInstanceCreateInfo ci = *pCreateInfo;
    VkApplicationInfo ai = {};
    if (pCreateInfo->pApplicationInfo) ai = *pCreateInfo->pApplicationInfo;
    else ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    if (ai.apiVersion < VK_API_VERSION_1_3)
    {
        Log("vkCreateInstance: raising apiVersion %u.%u -> 1.3\n",
            VK_VERSION_MAJOR(ai.apiVersion), VK_VERSION_MINOR(ai.apiVersion));
        ai.apiVersion = VK_API_VERSION_1_3;
    }
    ci.pApplicationInfo = &ai;

    VkResult r = nextCreate(&ci, pAllocator, pInstance);
    if (r != VK_SUCCESS) { Log("vkCreateInstance failed below us: %d\n", r); return r; }

    auto* d = new InstanceData();
    d->instance = *pInstance;
    d->nextGIPA = nextGIPA;
    d->apiVersion = ai.apiVersion;
#define X(name) d->name = (PFN_vk##name)nextGIPA(*pInstance, "vk" #name);
    WIZ3D_VK_INSTANCE_FUNCS(X)
#undef X
    {
        std::lock_guard<std::mutex> g(s_lock);
        s_instances[DispatchKey(*pInstance)] = d;
    }
    Log("vkCreateInstance: app='%s' apiVersion=%u.%u extensions=%u\n",
        pCreateInfo->pApplicationInfo && pCreateInfo->pApplicationInfo->pApplicationName ? pCreateInfo->pApplicationInfo->pApplicationName : "",
        VK_VERSION_MAJOR(d->apiVersion), VK_VERSION_MINOR(d->apiVersion), pCreateInfo->enabledExtensionCount);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator)
{
    InstanceData* d = FindInstance(instance);
    if (!d) return;
    d->DestroyInstance(instance, pAllocator);
    {
        std::lock_guard<std::mutex> g(s_lock);
        s_instances.erase(DispatchKey(instance));
    }
    Log("vkDestroyInstance\n");
    delete d;
}

// Explicit layers answer these for their own name only; the loader aggregates.
static VKAPI_ATTR VkResult VKAPI_CALL Hook_EnumerateInstanceLayerProperties(uint32_t* pCount, VkLayerProperties* pProps)
{
    if (pProps && *pCount >= 1)
    {
        strcpy_s(pProps[0].layerName, "VK_LAYER_wiz3D_stereo");
        strcpy_s(pProps[0].description, "wiz3D stereoscopic 3D layer");
        pProps[0].specVersion = VK_API_VERSION_1_3;
        pProps[0].implementationVersion = 1;
    }
    *pCount = 1;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_EnumerateInstanceExtensionProperties(
    const char* pLayerName, uint32_t* pCount, VkExtensionProperties*)
{
    if (pLayerName && strcmp(pLayerName, "VK_LAYER_wiz3D_stereo") == 0) { *pCount = 0; return VK_SUCCESS; }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

static bool HiddenDeviceExtension(const char* name)
{
    return (Vk().VKHideDescriptorBuffer && strcmp(name, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) == 0) ||
           (Vk().VKHideDepthBiasControl && strcmp(name, VK_EXT_DEPTH_BIAS_CONTROL_EXTENSION_NAME) == 0);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_EnumerateDeviceExtensionProperties(
    VkPhysicalDevice physDev, const char* pLayerName, uint32_t* pCount, VkExtensionProperties* pProps)
{
    if (pLayerName && strcmp(pLayerName, "VK_LAYER_wiz3D_stereo") == 0) { *pCount = 0; return VK_SUCCESS; }
    InstanceData* d = FindInstance(physDev);
    if (!d) return VK_ERROR_INITIALIZATION_FAILED;
    if (pLayerName || (!Vk().VKHideDescriptorBuffer && !Vk().VKHideDepthBiasControl))
        return d->EnumerateDeviceExtensionProperties(physDev, pLayerName, pCount, pProps);
    // VKHideDescriptorBuffer / VKHideDepthBiasControl: the driver's list less the hidden names, under the usual
    // count / VK_INCOMPLETE contract.
    uint32_t n = 0;
    VkResult r = d->EnumerateDeviceExtensionProperties(physDev, nullptr, &n, nullptr);
    if (r != VK_SUCCESS) return r;
    std::vector<VkExtensionProperties> all(n);
    r = d->EnumerateDeviceExtensionProperties(physDev, nullptr, &n, all.data());
    if (r < 0) return r;
    all.resize(n);
    size_t kept = 0;
    for (size_t i = 0; i < all.size(); ++i)
    {
        if (!HiddenDeviceExtension(all[i].extensionName)) { all[kept++] = all[i]; continue; }
        static std::atomic<uint32_t> s_logged{ 0 };
        const uint32_t bit = strcmp(all[i].extensionName, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) == 0 ? 1u : 2u;
        if (!(s_logged.fetch_or(bit) & bit)) Log("[ext] %s hidden from the device's extension list\n", all[i].extensionName);
    }
    all.resize(kept);
    if (!pProps) { *pCount = (uint32_t)kept; return VK_SUCCESS; }
    const uint32_t copied = *pCount < kept ? *pCount : (uint32_t)kept;
    memcpy(pProps, all.data(), copied * sizeof(VkExtensionProperties));
    *pCount = copied;
    return copied < kept ? VK_INCOMPLETE : VK_SUCCESS;
}

template <class F> static void ClearUpdateAfterBind(F* f)
{
    f->descriptorBindingUniformBufferUpdateAfterBind = VK_FALSE;
    f->descriptorBindingSampledImageUpdateAfterBind = VK_FALSE;
    f->descriptorBindingStorageImageUpdateAfterBind = VK_FALSE;
    f->descriptorBindingStorageBufferUpdateAfterBind = VK_FALSE;
    f->descriptorBindingUniformTexelBufferUpdateAfterBind = VK_FALSE;
    f->descriptorBindingStorageTexelBufferUpdateAfterBind = VK_FALSE;
}

// bindTimeDescriptors: update-after-bind is hidden, so the host writes descriptors at bind rather than at submit,
// before the right eye's replay and the dispatch gate read the set.
static VKAPI_ATTR void VKAPI_CALL Hook_GetPhysicalDeviceFeatures2(VkPhysicalDevice physDev, VkPhysicalDeviceFeatures2* pFeatures)
{
    InstanceData* d = FindInstance(physDev);
    if (!d || !d->GetPhysicalDeviceFeatures2) return;
    d->GetPhysicalDeviceFeatures2(physDev, pFeatures);
    // VKDynamicRendering: graphics pipeline libraries are hidden, since a pipeline linked from stage libraries names
    // no stages at the link; the host compiles pipelines whole instead.
    if (Vk().VKDynamicRendering)
        for (auto p = (VkBaseOutStructure*)pFeatures->pNext; p; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT)
            {
                ((VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT*)p)->graphicsPipelineLibrary = VK_FALSE;
                static std::atomic<bool> s_gpl{ false };
                if (!s_gpl.exchange(true)) Log("[gpl] graphics pipeline libraries hidden: pipelines are compiled whole\n");
            }
    // A hidden extension's features read false too, for a caller that chains them without checking the extension list.
    for (auto p = (VkBaseOutStructure*)pFeatures->pNext; p; p = p->pNext)
        if (Vk().VKHideDescriptorBuffer && p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT)
        {
            auto* f = (VkPhysicalDeviceDescriptorBufferFeaturesEXT*)p;
            f->descriptorBuffer = f->descriptorBufferCaptureReplay = f->descriptorBufferImageLayoutIgnored = f->descriptorBufferPushDescriptors = VK_FALSE;
        }
        else if (Vk().VKHideDepthBiasControl && p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES_EXT)
        {
            auto* f = (VkPhysicalDeviceDepthBiasControlFeaturesEXT*)p;
            f->depthBiasControl = f->leastRepresentableValueForceUnormRepresentation = f->floatRepresentation = f->depthBiasExact = VK_FALSE;
        }
    if (!Traits().bindTimeDescriptors) return;
    for (auto p = (VkBaseOutStructure*)pFeatures->pNext; p; p = p->pNext)
    {
        if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES) ClearUpdateAfterBind((VkPhysicalDeviceDescriptorIndexingFeatures*)p);
        else if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) ClearUpdateAfterBind((VkPhysicalDeviceVulkan12Features*)p);
    }
    static std::atomic<bool> s_logged{ false };
    if (!s_logged.exchange(true)) Log("[ext] update-after-bind hidden: descriptors are written at bind\n");
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateWin32SurfaceKHR(
    VkInstance instance, const VkWin32SurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    InstanceData* d = FindInstance(instance);
    if (!d || !d->CreateWin32SurfaceKHR) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = d->CreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (r == VK_SUCCESS)
    {
        std::lock_guard<std::mutex> g(d->lock);   // InstanceData, not the device
        d->surfaces[*pSurface] = pCreateInfo->hwnd;
    }
    Log("vkCreateWin32SurfaceKHR: hwnd=%p -> %d\n", (void*)pCreateInfo->hwnd, r);
    return r;
}

// Device
static bool DeviceExtensionAvailable(InstanceData* inst, VkPhysicalDevice pd, const char* name)
{
    uint32_t n = 0;
    if (inst->EnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> props(n);
    inst->EnumerateDeviceExtensionProperties(pd, nullptr, &n, props.data());
    for (const auto& p : props) if (strcmp(p.extensionName, name) == 0) return true;
    return false;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDevice(
    VkPhysicalDevice physDev, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice)
{
    InstanceData* inst = FindInstance(physDev);
    if (!inst) return VK_ERROR_INITIALIZATION_FAILED;

    auto* link = FindLayerLinkInfo<VkLayerDeviceCreateInfo, VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO>(
        pCreateInfo->pNext, VK_LAYER_LINK_INFO);
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr nextGIPA = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   nextGDPA = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    // The loader's callback for command buffers a layer allocates itself (StampCommandBuffers).
    auto* loaderData = FindLayerLinkInfo<VkLayerDeviceCreateInfo, VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO>(
        pCreateInfo->pNext, VK_LOADER_DATA_CALLBACK);

    auto nextCreate = (PFN_vkCreateDevice)nextGIPA(inst->instance, "vkCreateDevice");
    if (!nextCreate) return VK_ERROR_INITIALIZATION_FAILED;

    // The weaver runs on a D3D11 device fed through shared textures, which
    // needs external memory; add the extensions when the game did not.
    std::vector<const char*> exts(pCreateInfo->ppEnabledExtensionNames, pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);
    bool extMem = false, extSem = false;
    if (WeaveSelected())
    {
        // VKWeaveGpuSync: the weave's stages wait on a D3D11 fence imported as a semaphore.
        if (Vk().VKWeaveGpuSync)
        {
            extSem = true;
            for (const char* w : { VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME })
            {
                bool present = false;
                for (const char* e : exts) if (strcmp(e, w) == 0) { present = true; break; }
                if (present) continue;
                if (DeviceExtensionAvailable(inst, physDev, w)) exts.push_back(w);
                else if (strcmp(w, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME) == 0) extSem = false;
            }
        }
        static const char* kWanted[] = {
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        };
        extMem = true;
        for (const char* w : kWanted)
        {
            bool present = false;
            for (const char* e : exts) if (strcmp(e, w) == 0) { present = true; break; }
            if (present) continue;
            if (DeviceExtensionAvailable(inst, physDev, w)) exts.push_back(w);
            else if (strcmp(w, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) == 0) extMem = false;
        }
    }
    // A game that aliases NV-dedicated images onto shared allocations lets siblings alias the same way
    // (dedicatedImageAliasing), instead of one dedicated allocation each.
    bool aliasing = false;
    for (const char* e : exts)
        if (strcmp(e, VK_NV_DEDICATED_ALLOCATION_IMAGE_ALIASING_EXTENSION_NAME) == 0) { aliasing = true; break; }
    VkDeviceCreateInfo ci = *pCreateInfo;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();

    VkResult r = nextCreate(physDev, &ci, pAllocator, pDevice);
    if (r != VK_SUCCESS) { Log("vkCreateDevice failed below us: %d\n", r); return r; }

    auto* d = new DeviceData();
    d->device = *pDevice;
    d->physDev = physDev;
    // Fragment shaders may write storage buffers only with this feature (the camera-window hit counter).
    if (pCreateInfo->pEnabledFeatures) d->fragmentAtomics = pCreateInfo->pEnabledFeatures->fragmentStoresAndAtomics != 0;
    for (const VkBaseInStructure* s = (const VkBaseInStructure*)pCreateInfo->pNext; s; s = s->pNext)
    {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
            d->fragmentAtomics = ((const VkPhysicalDeviceFeatures2*)s)->features.fragmentStoresAndAtomics != 0;
    }
    d->inst = inst;
    d->nextGDPA = nextGDPA;
    d->setLoaderData = loaderData ? loaderData->u.pfnSetDeviceLoaderData : nullptr;
    // Promoted functions may only exist under their KHR name on older devices.
#define X(name) d->name = (PFN_vk##name)nextGDPA(*pDevice, "vk" #name); \
                if (!d->name) d->name = (PFN_vk##name)nextGDPA(*pDevice, "vk" #name "KHR");
    WIZ3D_VK_DEVICE_FUNCS(X)
#undef X
    // Core 1.3 functions that also ship as an EXT alias, which the KHR fallback above does not cover.
    if (!d->CmdBindVertexBuffers2)
        d->CmdBindVertexBuffers2 = (PFN_vkCmdBindVertexBuffers2)nextGDPA(*pDevice, "vkCmdBindVertexBuffers2EXT");
    // Optional state functions, under their core, KHR or EXT names.
#define X(name) d->name = (PFN_vk##name)nextGDPA(*pDevice, "vk" #name); \
                if (!d->name) d->name = (PFN_vk##name)nextGDPA(*pDevice, "vk" #name "KHR"); \
                if (!d->name) d->name = (PFN_vk##name)nextGDPA(*pDevice, "vk" #name "EXT");
    WIZ3D_VK_OPTIONAL_DEVICE_FUNCS(X)
#undef X
    // A core entry point the layer chain left unresolved would be a null call: named in the log.
#define X(name) if (!d->name) Log("!! device fn UNRESOLVED: vk" #name "\n");
    WIZ3D_VK_DEVICE_FUNCS(X)
#undef X
    inst->GetPhysicalDeviceMemoryProperties(physDev, &d->memProps);
    inst->GetPhysicalDeviceProperties(physDev, &d->props);
    if (inst->GetPhysicalDeviceQueueFamilyProperties)
    {
        uint32_t families = 0;
        inst->GetPhysicalDeviceQueueFamilyProperties(physDev, &families, nullptr);
        d->familyProps.resize(families);
        inst->GetPhysicalDeviceQueueFamilyProperties(physDev, &families, d->familyProps.data());
    }
    {
        std::lock_guard<std::mutex> g(s_lock);
        s_devices[DispatchKey(*pDevice)] = d;
    }

    d->externalMemoryWin32 = extMem;
    d->externalSemaphoreWin32 = extSem;
    d->dedicatedImageAliasing = aliasing;
    Log("vkCreateDevice: gpu='%s' driverApi=%u.%u.%u queues=%u extensions=%u (game %u)\n",
        d->props.deviceName, VK_VERSION_MAJOR(d->props.apiVersion), VK_VERSION_MINOR(d->props.apiVersion),
        VK_VERSION_PATCH(d->props.apiVersion), pCreateInfo->queueCreateInfoCount, ci.enabledExtensionCount, pCreateInfo->enabledExtensionCount);
    for (uint32_t i = 0; i < ci.enabledExtensionCount; ++i)
        Log("  ext: %s\n", ci.ppEnabledExtensionNames[i]);

    // Per-game profile (presets, knobs), as the DX11 device proxy does.
    static bool s_profileRead = false;
    if (!s_profileRead)
    {
        s_profileRead = true;
        DataInput savedInput = gInfo.Input;
        ReadCurrentProfile(d->props.vendorID);
        FamiliesLoadProfile();
        const CameraPreset* loaded = gInfo.Input.GetActivePreset();
        if (!loaded || loaded->StereoBase == 0.f)
        {
            Log("Profile: discarded the loaded profile (%s), keeping what was already in gInfo\n",
                loaded ? "StereoBase is 0" : "no active preset");
            gInfo.Input = savedInput;
        }
        const CameraPreset* act = gInfo.Input.GetActivePreset();
        Log("Profile: StereoBase=%.6f One_div_ZPS=%.6f eyeShift=%.6f DetectInverseMatrix=%d RTMode=%d DriverDirectory='%S'\n",
            act ? act->StereoBase : 0.f, act ? act->One_div_ZPS : 0.f, wiz3D_GetEffectiveEyeShift(),
            (int)gInfo.DetectInverseMatrix, (int)gInfo.RenderTargetCreationMode, gInfo.DriverDirectory);
    }

    // Stereo needs EnableStereo, RouterType 0 and room for the stereo set; the passthrough row names which failed.
    const bool enableOk = gInfo.EnableStereo;
    const bool routerOk = gInfo.RouterType == 0;
    const bool setsOk = d->props.limits.maxBoundDescriptorSets > kStereoSet;
    const bool paramsOk = enableOk && routerOk && setsOk && StereoParamsInit(d);
    d->stereoEnabled = paramsOk;
    if (d->stereoEnabled) ConstantsInit(d);
    if (d->stereoEnabled) TilesInit(d);
    if (d->stereoEnabled) FgInit(d);
    if (d->stereoEnabled) InstallNgxWatcherVK();
    // The layer presents the SR weave or half side-by-side: any other OutputMode gets side-by-side, named in this row.
    if (d->stereoEnabled && !WeaveSelected() && gInfo.OutputMode != OUTMODE_HALF_SIDE_BY_SIDE)
    {
        const char* mode = OutputModeToCanonical((int)gInfo.OutputMode);
        Log("OutputMode %d (%s) is not one the Vulkan layer presents: half side-by-side is used\n", (int)gInfo.OutputMode, mode ? mode : "unknown");
    }
    if (d->stereoEnabled)
        Log("Stereo ENABLED, DetectInverseMatrix=%d\n", (int)gInfo.DetectInverseMatrix);
    else
        Log("Stereo disabled (passthrough): EnableStereo=%d (needs 1), RouterType=%d (needs 0), maxBoundDescriptorSets=%u (needs > %u), StereoParamsInit=%s\n",
            (int)gInfo.EnableStereo,
            (int)gInfo.RouterType, d->props.limits.maxBoundDescriptorSets, kStereoSet,
            !enableOk || !routerOk || !setsOk ? "not attempted" : "FAILED");
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    if (!d) return;
    Log("vkDestroyDevice after %u presents\n", d->frames);
    if (gInfo.CollectDebugInformation) LogFrameStats(d);
    WriteInputData(&gInfo.Input);
    gKbdHook.clear();
    d->DeviceWaitIdle(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (auto& kv : d->swapchains) { WeaveDestroySwapchain(d, kv.second); CompositeDestroy(d, kv.second); }
        OsdDestroyDevice(d);
        ScreenshotDestroy(d);
        Log("  swapchain resources released\n");
        WeaveDestroyDevice(d);
        Log("  weaver released\n");
        for (auto& kv : d->presentPools) d->DestroyCommandPool(device, kv.second, nullptr);
        while (!d->cmdStates.empty()) CmdStateFree(d, d->cmdStates.begin()->first);
        ShadowPoolsDestroy(d);
        StereoParamsDestroy(d);
        if (d->constants) ConstantsDestroy(d);
        TwinsDestroy(d);
        TilesDestroy(d);
        FgDestroy(d);
    }
    d->DestroyDevice(device, pAllocator);
    Log("  device destroyed\n");
    {
        std::lock_guard<std::mutex> g(s_lock);
        s_devices.erase(DispatchKey(device));
        g_deviceGen.fetch_add(1, std::memory_order_release);
    }
    delete d;
}

// Proc-addr resolution
static const HookEntry s_instanceHooks[] = {
    WIZ3D_HOOK(CreateInstance), WIZ3D_HOOK(DestroyInstance),
    WIZ3D_HOOK(EnumerateInstanceLayerProperties), WIZ3D_HOOK(EnumerateInstanceExtensionProperties),
    WIZ3D_HOOK(EnumerateDeviceExtensionProperties),
    WIZ3D_HOOK(GetPhysicalDeviceFeatures2),
    { "vkGetPhysicalDeviceFeatures2KHR", (PFN_vkVoidFunction)&Hook_GetPhysicalDeviceFeatures2 },
    WIZ3D_HOOK(CreateWin32SurfaceKHR),
    WIZ3D_HOOK(CreateDevice),
    { nullptr, nullptr },
};
static const HookEntry s_deviceCoreHooks[] = {
    WIZ3D_HOOK(DestroyDevice),
    { nullptr, nullptr },
};
static const HookEntry* const s_deviceTables[] = {
    s_deviceCoreHooks, g_resourceHooks, g_descriptorHooks, g_pipelineHooks, g_commandHooks, g_swapchainHooks, g_constantsHooks, g_twinHooks,
};
// Passthrough (RouterType 2) keeps only the per-object and present hooks: the command hooks' state lookups would
// cost a mono baseline CPU time.
static const HookEntry* const s_passthroughTables[] = {
    s_deviceCoreHooks, g_resourceHooks, g_swapchainHooks,
};

static PFN_vkVoidFunction LookupTable(const HookEntry* table, const char* name)
{
    for (; table->name; ++table)
        if (strcmp(table->name, name) == 0) return table->fn;
    return nullptr;
}

// Promoted extensions keep their KHR-suffixed aliases (and, with VKDynamicRendering, EXT ones such as
// vkCmdSetCullModeEXT) with identical signatures; a game calling the alias must reach the same hook.
static PFN_vkVoidFunction LookupDeviceHook(const char* name, bool passthrough = false)
{
    const HookEntry* const* tables = passthrough ? s_passthroughTables : s_deviceTables;
    const size_t count = passthrough ? _countof(s_passthroughTables) : _countof(s_deviceTables);
    for (size_t i = 0; i < count; ++i)
        if (PFN_vkVoidFunction f = LookupTable(tables[i], name)) return f;
    const size_t n = strlen(name);
    if (n > 3 && (strcmp(name + n - 3, "KHR") == 0 || (Vk().VKDynamicRendering && strcmp(name + n - 3, "EXT") == 0)))
    {
        char core[128];
        if (n - 3 < sizeof(core))
        {
            memcpy(core, name, n - 3);
            core[n - 3] = 0;
            for (size_t i = 0; i < count; ++i)
                if (PFN_vkVoidFunction f = LookupTable(tables[i], core)) return f;
        }
    }
    return nullptr;
}

PFN_vkVoidFunction LayerGetDeviceProcAddr(VkDevice device, const char* pName)
{
    if (!pName) return nullptr;
    if (strcmp(pName, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)&LayerGetDeviceProcAddr;
    DeviceData* d = FindDevice(device);
    if (!d) return nullptr;
    // Hooks only for functions the next layer actually provides, so an
    // unsupported extension stays unsupported.
    PFN_vkVoidFunction next = d->nextGDPA(device, pName);
    if (!next) return nullptr;
    if (PFN_vkVoidFunction f = LookupDeviceHook(pName, !d->stereoEnabled)) return f;
    return next;
}

PFN_vkVoidFunction LayerGetInstanceProcAddr(VkInstance instance, const char* pName)
{
    if (!pName) return nullptr;
    if (strcmp(pName, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)&LayerGetInstanceProcAddr;
    if (strcmp(pName, "vkGetDeviceProcAddr") == 0)   return (PFN_vkVoidFunction)&LayerGetDeviceProcAddr;
    if (PFN_vkVoidFunction f = LookupTable(s_instanceHooks, pName)) return f;
    if (!instance) return nullptr;
    InstanceData* d = FindInstance(instance);
    if (!d) return nullptr;
    PFN_vkVoidFunction next = d->nextGIPA(instance, pName);
    if (!next) return nullptr;
    // The loader may resolve device functions through the instance too.
    if (PFN_vkVoidFunction f = LookupDeviceHook(pName)) return f;
    return next;
}

}} // namespace wiz3d::vk
