/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Shader modules are patched at creation (SpirvPatch.cpp and the families' rules) and pipelines tracked with their
// layout and stages, so the recorder knows where the stereo set lives and which twin each eye binds.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include <stdio.h>
#include <spirv/unified1/spirv.hpp11>

namespace wiz3d { namespace vk {

static uint32_t Crc32(const void* data, size_t n)
{
    static uint32_t table[256];
    static bool init = false;
    if (!init)
    {
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

// Next to the layer DLL, where S3DWrapperVK.log goes.
static void DumpSpirv(const char* tag, uint32_t crc, const uint32_t* words, size_t n)
{
    char path[MAX_PATH];
    GetModuleFileNameA(GetModuleHandleA("S3DWrapperVK.dll"), path, MAX_PATH);
    char* p = strrchr(path, '\\');
    if (!p) return;
    _snprintf_s(p + 1, sizeof(path) - (p + 1 - path), _TRUNCATE, "vk_%08X_%s.spv", crc, tag);
    FILE* f = fopen(path, "wb");
    if (!f)
    {
        // Game folders under Program Files-style ACLs refuse writes.
        char local[MAX_PATH];
        if (!GetEnvironmentVariableA("LOCALAPPDATA", local, sizeof(local))) return;
        strcat_s(local, "\\wiz3D");
        CreateDirectoryA(local, nullptr);
        char name[64];
        _snprintf_s(name, sizeof(name), _TRUNCATE, "\\vk_%08X_%s.spv", crc, tag);
        strcat_s(local, name);
        f = fopen(local, "wb");
        if (!f) return;
    }
    fwrite(words, 4, n, f);
    fclose(f);
}

// Position.w is only ever the constant 1: whole stores of constant vectors with w 1, component stores to x..z only.
static bool PositionWIsOne(const uint32_t* code, size_t words)
{
    uint32_t pos = 0;
    bool stored = false;
    std::unordered_map<uint32_t, uint32_t> consts;
    std::unordered_map<uint32_t, std::vector<uint32_t>> vectors;
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> chains;   // pointer -> (base, index)
    for (size_t i = 5; i < words; )
    {
        const uint32_t n = code[i] >> 16;
        if (!n || i + n > words) return false;
        const uint32_t* w = code + i;
        const spv::Op op = (spv::Op)(w[0] & 0xFFFF);
        if (op == spv::Op::OpDecorate && n >= 4 && w[2] == (uint32_t)spv::Decoration::BuiltIn && w[3] == (uint32_t)spv::BuiltIn::Position) pos = w[1];
        else if (op == spv::Op::OpConstant && n == 4) consts[w[2]] = w[3];
        else if (op == spv::Op::OpConstantComposite) vectors[w[2]].assign(w + 3, w + n);
        else if (op == spv::Op::OpAccessChain || op == spv::Op::OpInBoundsAccessChain)
        {
            if (pos && w[3] == pos && n != 5) return false;
            if (n == 5) chains[w[2]] = { w[3], w[4] };
        }
        else if (op == spv::Op::OpStore && pos)
        {
            if (w[1] == pos)
            {
                auto v = vectors.find(w[2]);
                auto c = v != vectors.end() && v->second.size() == 4 ? consts.find(v->second[3]) : consts.end();
                if (c == consts.end() || c->second != 0x3F800000u) return false;
                stored = true;
            }
            auto ch = chains.find(w[1]);
            if (ch != chains.end() && ch->second.first == pos)
            {
                auto c = consts.find(ch->second.second);
                if (c == consts.end() || c->second >= 3) return false;
            }
        }
        i += n;
    }
    return stored;
}

static VkResult CreateShaderModuleImpl(
    VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule)
{
    DeviceData* d = FindDevice(device);
    if (!d->stereoEnabled || !pCreateInfo->pCode || pCreateInfo->codeSize < 20)
        return d->CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);

    const size_t words = pCreateInfo->codeSize / 4;
    // Matrices whose stride is not 16 are invisible to the register scan, so
    // record them per binding for the constants path to test directly.
    {
        std::vector<BlockMatrix> bms;
        AnalyzeBlockMatrices(pCreateInfo->pCode, words, bms);
        // Vectors only from bindings that also carry a packed matrix: those are
        // the scalar-layout blocks, and one of their vec3s is the camera.
        std::unordered_map<uint32_t, bool> packedBinding;
        for (const BlockMatrix& b : bms)
            if (b.cols >= 3 && b.stride && b.stride != 16) packedBinding[b.binding] = true;
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (const BlockMatrix& b : bms)
        {
            const bool vec = b.cols == 1;
            if (vec ? !packedBinding.count(b.binding) : (b.stride == 16 || b.rows < 3 || b.cols < 3)) continue;
            auto& v = d->packedMatrices[b.binding];
            bool seen = false;
            for (const BlockMatrix& e : v)
                if (e.offset == b.offset && e.stride == b.stride && e.rows == b.rows && e.cols == b.cols) { seen = true; break; }
            if (seen) continue;
            v.push_back(b);
            if (b.binding < 256) d->packedBindingMask[b.binding >> 6] |= 1ull << (b.binding & 63);
            Log("[packed] set=%u binding=%u offset=%u stride=%u %ux%u %s %s\n",
                b.set, b.binding, b.offset, b.stride, b.rows, b.cols,
                b.colMajor ? "col" : "row", b.ssbo ? "SSBO" : "uniform");
        }
    }
    std::vector<uint32_t> patched;
    const char* reason = "";
    spv::ExecutionModel model = spv::ExecutionModel::Max;
    // The DX11 split (Device11Proxy::TryBuildModifiedVS): a stage reaching gl_Position through a camera matrix is
    // served by patching that matrix for its draw, not by shifting itself; stages with no matrix self-shift.
    std::vector<PositionUse> uses;
    uint64_t posMask = 0;
    bool posUsesMatrix = false;
    uint32_t posSet = 0, posBinding = 0, posOffset = 0;
    if (AnalyzePositionChain(pCreateInfo->pCode, words, uses))
        for (const PositionUse& u : uses)
            if (u.matrix)
            {
                if (!posUsesMatrix) { posSet = u.set; posBinding = u.binding; posOffset = u.offset; }
                posUsesMatrix = true;
                if (u.offset / 16 < 64) posMask |= 1ull << (u.offset / 16);
            }
    // Screen-uv varyings move with the shift under VKScreenUvVaryings, or with VKPushDeltas at remap level 3+.
    SpirvRules rules;
    rules.screenUv = ((Vk().VKPushDeltas && Vk().VKComputeRemap >= 3) || Vk().VKScreenUvVaryings) && !ComputeRemapKeepsScreenUv(Crc32(pCreateInfo->pCode, pCreateInfo->codeSize));
    const bool screenUv = rules.screenUv;
    SpirvCounts counts;
    const bool vsPatched = PatchSpirv(pCreateInfo->pCode, words, rules, patched, &reason, &model, &counts);
    const unsigned posVaryings = counts.posVaryings, prevClip = counts.prevClip, ndcVaryings = counts.ndcVaryings, screenUvMoved = counts.screenUvMoved;
    if (vsPatched)
    {
        if (screenUvMoved && Vk().VKScreenUvVaryings)
            Log("[screenuv] VS %08X: %u screen-uv products take x after the shift\n", Crc32(pCreateInfo->pCode, pCreateInfo->codeSize), screenUvMoved);
        if (posVaryings) { std::lock_guard<std::shared_mutex> g(d->lock); ++d->statPosVaryingShaders; }
        if (prevClip) { std::lock_guard<std::shared_mutex> g(d->lock); ++d->statPrevClipShaders; }
        if (ndcVaryings)
        {
            const uint32_t crc = Crc32(pCreateInfo->pCode, pCreateInfo->codeSize);
            Log("[ndc] %s %08X: %u screen-position varyings shift with Position\n", model == spv::ExecutionModel::Vertex ? "VS" : "TES", crc, ndcVaryings);
        }
        // Served by constants: the submit patches the projection this draw reads. With VKServeFallback the module also
        // self-shifts and the replay picks the sep=0 set only for served draws (PipelineInfo::served).
        const bool byConstants = posUsesMatrix && gInfo.DetectInverseMatrix;
        const bool unmodified = byConstants && !Vk().VKServeFallback;
        VkShaderModuleCreateInfo ci = *pCreateInfo;
        if (!unmodified) { ci.codeSize = patched.size() * 4; ci.pCode = patched.data(); }
        VkResult r = d->CreateShaderModule(device, &ci, pAllocator, pShaderModule);
        if (byConstants && r == VK_SUCCESS)
        {
            ShaderModuleInfo info;
            info.model = model;
            info.selfShifted = !unmodified;
            // The GS revert in CreateGraphicsPipelines needs the unshifted twin.
            if (!unmodified && model == spv::ExecutionModel::Vertex)
                d->CreateShaderModule(device, pCreateInfo, pAllocator, &info.original);
            info.posUsesMatrix = true;
            info.posSet = posSet; info.posBinding = posBinding; info.posOffset = posOffset;
            info.posMask = posMask;
            info.crc = Crc32(pCreateInfo->pCode, pCreateInfo->codeSize);
            info.posVaryings = posVaryings;
            FamiliesOnPositionShader(info.crc);
            std::lock_guard<std::shared_mutex> g(d->lock);
            for (const PositionUse& u : uses) if (u.matrix) ++d->positionUse[u.binding][u.offset];
            d->shaderModules[*pShaderModule] = info;
            ++d->statConstantsShaders;
            // Dumped: a camera load that bypasses the descriptors (buffer device address) shows in spirv-dis.
            if (Vk().VKDumpModifiedShaders)
                DumpSpirv("const", Crc32(pCreateInfo->pCode, pCreateInfo->codeSize), pCreateInfo->pCode, words);
            return r;
        }
        const uint32_t crc = Crc32(pCreateInfo->pCode, pCreateInfo->codeSize);
        if (r == VK_SUCCESS)
        {
            ++d->statShadersPatched;
            ShaderModuleInfo info;
            info.model = model;
            if (model == spv::ExecutionModel::MeshEXT) Log("[mesh] %08X: the mesh stage shifts Position at its stores\n", crc);
            // A vertex stage followed by tessellation or geometry must not shift (the later stage does):
            // the original module is kept for pipeline creation to swap in.
            if (model == spv::ExecutionModel::Vertex)
                d->CreateShaderModule(device, pCreateInfo, pAllocator, &info.original);
            info.selfShifted = true;
            info.posUsesMatrix = posUsesMatrix;
            info.posSet = posSet; info.posBinding = posBinding; info.posOffset = posOffset;
            info.posMask = posMask;
            info.crc = crc;
            info.posVaryings = posVaryings;
            info.ndcVaryings = ndcVaryings;
            info.unitW = model == spv::ExecutionModel::Vertex && PositionWIsOne(pCreateInfo->pCode, words);
            if (Traits().flatNoDepthDraws && model == spv::ExecutionModel::Vertex) info.rsxInputs = CountRsxVertexInputs(pCreateInfo->pCode, words);
            // The VS (or a listed TES) with its clip copies shifted, for pipelines whose FS reads the screen there.
            const bool varyStage = model == spv::ExecutionModel::Vertex ||
                (model == spv::ExecutionModel::TessellationEvaluation && ScreenReadTes(crc));
            const bool skyVs = (Vk().VKComputeRemap >= 3 && Vk().VKPushDeltas && ComputeRemapSkyVS(crc)) || (Traits().skyTwins && GtsSkyVS(crc));
            if ((posVaryings || skyVs) && Vk().VKScreenReadVaryings && varyStage)
            {
                std::vector<uint32_t> vt;
                const char* why = "";
                spv::ExecutionModel m3;
                SpirvRules vary;
                vary.forceVaryings = true;
                vary.screenUv = screenUv;
                if (PatchSpirv(pCreateInfo->pCode, words, vary, vt, &why, &m3))
                {
                    VkShaderModuleCreateInfo vci = *pCreateInfo;
                    vci.codeSize = vt.size() * 4;
                    vci.pCode = vt.data();
                    if (d->CreateShaderModule(device, &vci, pAllocator, &info.varyTwin) != VK_SUCCESS) info.varyTwin = VK_NULL_HANDLE;
                    if (skyVs) Log("[spv] %08X sky VS: rays from the right eye's x, twin %s\n", crc, info.varyTwin ? "made" : "NOT made");
                    else Log("[screenread] VS %08X: %u clip copies move in its twin%s\n", crc, posVaryings, info.varyTwin ? "" : " (module creation FAILED)");
                    if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv(skyVs ? "sky" : "vary", crc, vt.data(), vt.size());
                }
            }
            // The sprite VS's twin, moved by its depth's parallax (EmitSpriteShift).
            if (model == spv::ExecutionModel::Vertex && Vk().VKFsCameraWindows && CameraSpriteVS(crc))
            {
                std::vector<uint32_t> st;
                const char* why = "";
                spv::ExecutionModel m2;
                SpirvRules sprite;
                sprite.spriteShift = true;
                sprite.spriteVarying = CameraSpriteVarying(crc);
                if (PatchSpirv(pCreateInfo->pCode, words, sprite, st, &why, &m2))
                {
                    VkShaderModuleCreateInfo sci = *pCreateInfo;
                    sci.codeSize = st.size() * 4;
                    sci.pCode = st.data();
                    if (d->CreateShaderModule(device, &sci, pAllocator, &info.spriteTwin) != VK_SUCCESS) info.spriteTwin = VK_NULL_HANDLE;
                    if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv("sprite", crc, st.data(), st.size());
                }
                Log("[spv] %08X vertex sprite shift twin %s (%s)\n", crc, info.spriteTwin ? "made" : "NOT made", why);
            }
            std::lock_guard<std::shared_mutex> g(d->lock);
            d->shaderModules[*pShaderModule] = info;
            if (!uses.empty())
            {
                static int s_useLog = 12;
                char line[512] = {}; size_t len = 0;
                for (const PositionUse& u : uses)
                {
                    if (u.matrix) ++d->positionUse[u.binding][u.offset];
                    if (s_useLog > 0) len += (size_t)snprintf(line + len, sizeof(line) - len > 0 ? sizeof(line) - len : 0, " s%u:b%u+%u%s", u.set, u.binding, u.offset, u.matrix ? "m" : "");
                }
                if (s_useLog > 0) { --s_useLog; Log("[spv] %08X position chain:%s\n", crc, line); }
            }
            if (Vk().VKDumpModifiedShaders)
            {
                DumpSpirv("orig", crc, pCreateInfo->pCode, words);
                DumpSpirv("patched", crc, patched.data(), patched.size());
            }
            return r;
        }
        Log("[spv] %08X patched module rejected by driver (%d), using original\n", crc, r);
        ++d->statShadersSkipped;
        return d->CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    }
    // Camera matrices a fragment shader loads as raw dwords get the eye's shift in-shader; same set 8 revert rule.
    const uint32_t camCrc = Vk().VKFsCameraWindows ? Crc32(pCreateInfo->pCode, pCreateInfo->codeSize) : 0;
    const bool camDecal = CameraDecalFS(camCrc);
    const bool camShaft = d->camCounters && CameraShaftFS(camCrc);
    if (Vk().VKFsCameraWindows && CameraShaftFS(camCrc) && !camShaft) Log("[spv] %08X shaft FS left alone: no fragment stores for set 8 binding 1\n", camCrc);
    if (Vk().VKFsCameraWindows && (CameraWindowsAllowed(camCrc) || CameraScreenXFS(camCrc) || camDecal || camShaft))
    {
        std::vector<uint32_t> fs;
        unsigned wins = 0;
        // The decal: screen x to the left eye's before both of its matrices, the camera inverse being dwords 16..31.
        CameraWindowsRules win;
        win.counters = d->camCounters != nullptr;
        win.screenX = CameraScreenXFS(camCrc) || camDecal;
        win.screenXBase = camDecal ? 16u : ~0u;
        const bool camPatched = camShaft ? PatchShaftFS(pCreateInfo->pCode, words, CameraShaftBlur(camCrc), kStereoSet, fs)
                                         : PatchCameraWindowsFS(pCreateInfo->pCode, words, kStereoSet, win, fs, &wins);
        if (camShaft && !camPatched) Log("[spv] %08X shaft FS: its values were not found, left alone\n", camCrc);
        if (camPatched)
        {
            const uint32_t crc = camCrc;
            VkShaderModuleCreateInfo ci = *pCreateInfo;
            ci.codeSize = fs.size() * 4;
            ci.pCode = fs.data();
            VkResult r = d->CreateShaderModule(device, &ci, pAllocator, pShaderModule);
            if (r == VK_SUCCESS)
            {
                ShaderModuleInfo info;
                info.model = spv::ExecutionModel::Fragment;
                info.selfShifted = false;
                info.fsEyeOrigin = true;
                info.fsCamera = true;
                info.crc = crc;
                // A decal reading depth at its clip varying keeps the VS twin that moves it, so the right camera
                // meets the right position.
                info.screenRead = camDecal && Vk().VKScreenReadVaryings && ReadsScreenAtClipVarying(pCreateInfo->pCode, words);
                d->CreateShaderModule(device, pCreateInfo, pAllocator, &info.original);
                {
                    std::lock_guard<std::shared_mutex> g(d->lock);
                    d->shaderModules[*pShaderModule] = info;
                    if (info.screenRead) ++d->statScreenReadFs;
                    ++d->statFsCamera;
                    d->statFsCameraWindows += wins;
                }
                static std::atomic<int> s_camLog{16};
                if (s_camLog.fetch_sub(1) > 0) Log("[spv] %08X fragment: %u matrix window(s) given the camera test\n", crc, wins);
                if (Vk().VKDumpModifiedShaders >= 2)
                {
                    DumpSpirv("other", crc, pCreateInfo->pCode, words);
                    DumpSpirv("fscam", crc, fs.data(), fs.size());
                }
                return r;
            }
            Log("[spv] %08X camera-window fragment rejected by driver (%d), using original\n", crc, r);
        }
    }
    // A family's fragment rules (FamilyVK.h), one path for whichever takes the module. Its original stays as the twin
    // for pipelines whose layout has no set 8 (Hook_CreateGraphicsPipelines).
    if (Traits().fragmentRules)
    {
        const uint32_t moduleCrc = Crc32(pCreateInfo->pCode, pCreateInfo->codeSize);
        FamilyPatch rp;
        if (const VkFamily* fam = FamiliesPatchFragment(pCreateInfo->pCode, words, moduleCrc, rp))
        {
            VkShaderModuleCreateInfo ci = *pCreateInfo;
            ci.codeSize = rp.code.size() * 4;
            ci.pCode = rp.code.data();
            VkResult r = d->CreateShaderModule(device, &ci, pAllocator, pShaderModule);
            if (r == VK_SUCCESS)
            {
                ShaderModuleInfo info;
                info.model = spv::ExecutionModel::Fragment;
                info.selfShifted = false;
                info.fsEyeOrigin = true;
                info.crc = moduleCrc;
                d->CreateShaderModule(device, pCreateInfo, pAllocator, &info.original);
                {
                    std::lock_guard<std::shared_mutex> g(d->lock);
                    d->shaderModules[*pShaderModule] = info;
                    if (rp.eyeOriginStats) { ++d->statFsEyeOrigin; d->statFsProducts += rp.products; }
                }
                fam->LogRule("FS", moduleCrc, rp, true);
                if (Vk().VKDumpModifiedShaders >= 2 && rp.dumpTag)
                {
                    if (rp.dumpOriginal) DumpSpirv("other", moduleCrc, pCreateInfo->pCode, words);
                    DumpSpirv(rp.dumpTag, moduleCrc, rp.code.data(), rp.code.size());
                }
                return r;
            }
            Log("[%s] FS %08X %s rejected by driver (%d), using original\n", fam->Tag(), moduleCrc, rp.rule, r);
        }
    }
    // Fragment/compute modules land here too; only log real failures.
    if (strcmp(reason, "noPositionStage") != 0)
    {
        ++d->statShadersSkipped;
        Log("[spv] %08X (%s) not patched: %s\n", Crc32(pCreateInfo->pCode, pCreateInfo->codeSize),
            model == spv::ExecutionModel::Vertex ? "vs" : model == spv::ExecutionModel::TessellationEvaluation ? "ds" :
            model == spv::ExecutionModel::Geometry ? "gs" : "?", reason);
    }
    // VKDumpModifiedShaders=2: every module, so fragment and compute reconstruction can be read offline.
    if (Vk().VKDumpModifiedShaders >= 2)
        DumpSpirv("other", Crc32(pCreateInfo->pCode, pCreateInfo->codeSize), pCreateInfo->pCode, words);
    // Compute modules land here; the writable-storage list feeds the
    // dispatch-replay gate (BindComputeShadows).
    std::vector<uint64_t> sw;
    AnalyzeStorageWrites(pCreateInfo->pCode, words, sw);
    VkResult tail = d->CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    // Every module is recorded (not only writers) so the Ctrl+F9 rows can name a pipeline's fragment shader.
    if (tail == VK_SUCCESS)
    {
        ShaderModuleInfo info;
        info.model = model;
        for (size_t i = 5; i + 1 < words; )
        {
            const uint32_t w0 = pCreateInfo->pCode[i];
            if (!(w0 >> 16)) break;
            if ((w0 & 0xFFFF) == (uint32_t)spv::Op::OpEntryPoint) { info.model = (spv::ExecutionModel)pCreateInfo->pCode[i + 1]; break; }
            i += w0 >> 16;
        }
        info.crc = Crc32(pCreateInfo->pCode, pCreateInfo->codeSize);
        info.selfShifted = false;
        info.gsPassthrough = info.model == spv::ExecutionModel::Geometry && !strcmp(reason, "noEmit");
        info.storageWrites = std::move(sw);
        if (info.model == spv::ExecutionModel::Fragment)
            for (size_t i = 5; i + 3 < words; )
            {
                const uint32_t w0 = pCreateInfo->pCode[i];
                if (!(w0 >> 16)) break;
                // OpDecorate %id BuiltIn FragCoord
                if ((w0 & 0xFFFF) == (uint32_t)spv::Op::OpDecorate && (w0 >> 16) >= 4 &&
                    pCreateInfo->pCode[i + 2] == (uint32_t)spv::Decoration::BuiltIn && pCreateInfo->pCode[i + 3] == (uint32_t)spv::BuiltIn::FragCoord)
                { info.readsFragCoord = true; break; }
                if ((w0 & 0xFFFF) == (uint32_t)spv::Op::OpFunction) break;
                i += w0 >> 16;
            }
        if (Vk().VKScreenReadVaryings && info.model == spv::ExecutionModel::Fragment)
        {
            info.screenRead = ReadsScreenAtClipVarying(pCreateInfo->pCode, words) || (Vk().VKComputeRemap >= 3 && ComputeRemapScreenRead(info.crc)) ||
                (Traits().skyTwins && GtsSkyFS(info.crc));
            if (info.screenRead) { std::lock_guard<std::shared_mutex> g2(d->lock); ++d->statScreenReadFs; }
            if (info.screenRead) Log("[screenread] FS %08X reads the screen at a clip varying: its pipelines take their VS's twin\n", info.crc);
        }
        // A camera-window twin for every FS, swapped in only by pipelines whose VS is a lamp-tile VS (LampTileVS).
        if (Vk().VKFsCameraWindows && info.model == spv::ExecutionModel::Fragment)
        {
            std::vector<uint32_t> ct;
            unsigned cw = 0;
            CameraWindowsRules win;
            win.counters = d->camCounters != nullptr;
            if (PatchCameraWindowsFS(pCreateInfo->pCode, words, kStereoSet, win, ct, &cw))
            {
                VkShaderModuleCreateInfo cci = *pCreateInfo;
                cci.codeSize = ct.size() * 4;
                cci.pCode = ct.data();
                if (d->CreateShaderModule(device, &cci, pAllocator, &info.camTwin) != VK_SUCCESS) info.camTwin = VK_NULL_HANDLE;
            }
        }
        // A surface-ray twin for every FS of that shape, swapped in only where the VS writes a screen-position varying.
        if (Vk().VKLightRaysCamera && Vk().VKShiftNdcVaryings && info.model == spv::ExecutionModel::Fragment)
        {
            std::vector<uint32_t> rt;
            unsigned rays = 0, uses = 0;
            if (PatchLightRaysFS(pCreateInfo->pCode, words, kStereoSet, rt, &rays, &uses))
            {
                VkShaderModuleCreateInfo rci = *pCreateInfo;
                rci.codeSize = rt.size() * 4;
                rci.pCode = rt.data();
                if (d->CreateShaderModule(device, &rci, pAllocator, &info.rayTwin) != VK_SUCCESS) info.rayTwin = VK_NULL_HANDLE;
                Log("[lightray] FS %08X: %u rays moved to the surface, %u uses%s\n", info.crc, rays, uses, info.rayTwin ? "" : " (module creation FAILED)");
                if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv("lightrays", info.crc, rt.data(), rt.size());
            }
        }
        // VKTileListsLeft: an FS reading per-tile lists built for the left eye reads them at the left's pixel.
        // VKScreenMarch: a march toward a point the left camera projected ends at this eye's; one twin carries both.
        if ((Vk().VKTileListsLeft || Vk().VKScreenMarch) && info.model == spv::ExecutionModel::Fragment)
        {
            std::vector<uint32_t> tt, mt;
            unsigned tiles = 0, marches = 0;
            const uint32_t* src = pCreateInfo->pCode;
            size_t srcWords = words;
            const float vmZ = Vk().VKViewModelClipZ, vmSep = Vk().VKViewModelSep, vmConv = Vk().VKViewModelConv;
            if (Vk().VKTileListsLeft && PatchTileListsLeftFS(src, srcWords, kStereoSet, tt, &tiles, vmZ, vmSep, vmConv))
            { src = tt.data(); srcWords = tt.size(); }
            if (Vk().VKScreenMarch && PatchScreenMarchFS(src, srcWords, kStereoSet, mt, &marches, vmZ, vmSep, vmConv))
            { src = mt.data(); srcWords = mt.size(); }
            if (src != pCreateInfo->pCode)
            {
                VkShaderModuleCreateInfo tci = *pCreateInfo;
                tci.codeSize = srcWords * 4;
                tci.pCode = src;
                if (d->CreateShaderModule(device, &tci, pAllocator, &info.tileTwin) != VK_SUCCESS) info.tileTwin = VK_NULL_HANDLE;
                static std::atomic<int> s_tileLog{ 8 };
                if (s_tileLog.fetch_sub(1) > 0 || marches)
                    Log("[tiles] FS %08X: %u tile lookup(s) at the left's pixel, %u screen march(es) to this eye's end%s\n", info.crc, tiles,
                        marches, info.tileTwin ? "" : " (module creation FAILED)");
                if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv("tiles", info.crc, src, srcWords);
            }
        }
        // A right-camera twin for compute (and image-storing fragment shaders: decals), swapped in by pipelines whose
        // layout carries set 8.
        if (Vk().VKComputeRemap && (info.model == spv::ExecutionModel::GLCompute || info.model == spv::ExecutionModel::Fragment) &&
            !ComputeRemapSkipped(info.crc))
        {
            std::vector<uint32_t> ct, fusedCode;
            unsigned fused = 0;
            // fusedMad: unfused multiply-adds become the Fma the remap rules match (FuseUnfusedMad).
            const uint32_t* src = pCreateInfo->pCode;
            size_t srcWords = words;
            if (Traits().fusedMad && FuseUnfusedMad(src, words, fusedCode, &fused)) { src = fusedCode.data(); srcWords = fusedCode.size(); }
            const int level = Vk().VKComputeRemap;
            ComputeRemapRules rules;
            rules.wide = level >= 2;
            rules.uvRays = level >= 3;
            rules.storeCamera = ComputeRemapStoresCamera(info.crc);
            rules.counters = d->camCounters != nullptr;
            rules.tiledLights = ComputeRemapTiledLights(info.crc);
            rules.rayRows = level >= 3 && ComputeRemapRayRows(info.crc);
            rules.softRead = level >= 3 && ComputeRemapSoftParticle(info.crc);
            rules.viewRows = level >= 3 && Vk().VKPushDeltas;
            rules.flareX = level >= 3 && Vk().VKPushDeltas && ComputeRemapFlareFS(info.crc);
            rules.uvAffine = level >= 4;
            rules.froxelLookups = Vk().VKFroxelLookups;
            rules.fragRays = level >= 3 && Vk().VKFragCoordRays;
            ComputeRemapCounts c;
            bool remapped = PatchComputeRemap(src, srcWords, kStereoSet, rules, ct, c);
            // (1/FragCoord.w) * (v * FragCoord.w) is v only where w is 1: that twin serves unit-w pipelines alone.
            if (!remapped && rules.uvAffine && info.model == spv::ExecutionModel::Fragment)
            {
                rules.perspInputs = true;
                remapped = info.remapUnitW = PatchComputeRemap(src, srcWords, kStereoSet, rules, ct, c);
            }
            if (remapped)
            {
                VkShaderModuleCreateInfo cci = *pCreateInfo;
                cci.codeSize = ct.size() * 4;
                cci.pCode = ct.data();
                if (d->CreateShaderModule(device, &cci, pAllocator, &info.remapTwin) != VK_SUCCESS) info.remapTwin = VK_NULL_HANDLE;
                Log("[csremap] %s %08X: ndc=%u rows=%u rays=%u reproj=%u eye=%u fused=%u%s%s\n", info.model == spv::ExecutionModel::Fragment ? "FS" : "CS",
                    info.crc, c.ndc, c.rows, c.rays, c.reproj, c.eye, fused, info.remapUnitW ? " (unit-w pipelines)" : "",
                    info.remapTwin ? "" : " (module creation FAILED)");
                if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv("remap", info.crc, ct.data(), ct.size());
            }
        }
        // A family's compute rules (FamilyVK.h) as a twin each run binds set 8 for.
        if (Traits().computeRules && info.model == spv::ExecutionModel::GLCompute && !info.remapTwin)
        {
            FamilyPatch rp;
            bool ok = false;
            if (const VkFamily* fam = FamiliesPatchCompute(pCreateInfo->pCode, words, info.crc, rp, &ok))
            {
                if (ok)
                {
                    VkShaderModuleCreateInfo cci = *pCreateInfo;
                    cci.codeSize = rp.code.size() * 4;
                    cci.pCode = rp.code.data();
                    if (d->CreateShaderModule(device, &cci, pAllocator, &info.remapTwin) != VK_SUCCESS) info.remapTwin = VK_NULL_HANDLE;
                    if (Vk().VKDumpModifiedShaders >= 2 && rp.dumpTag) DumpSpirv(rp.dumpTag, info.crc, rp.code.data(), rp.code.size());
                }
                fam->LogRule("CS", info.crc, rp, info.remapTwin != VK_NULL_HANDLE);
            }
        }
        if (Traits().rsxPositionTwins && info.model == spv::ExecutionModel::Fragment && info.readsFragCoord)
        {
            std::vector<uint32_t> rt;
            unsigned divides = 0;
            if (PatchRsxPositionFS(pCreateInfo->pCode, words, kStereoSet, rt, &divides))
            {
                VkShaderModuleCreateInfo rci = *pCreateInfo;
                rci.codeSize = rt.size() * 4;
                rci.pCode = rt.data();
                if (d->CreateShaderModule(device, &rci, pAllocator, &info.rsxTwin) != VK_SUCCESS) info.rsxTwin = VK_NULL_HANDLE;
                Log("[rsxpos] FS %08X: %u position(s) moved to the left's pixel%s\n", info.crc, divides, info.rsxTwin ? "" : " (module creation FAILED)");
                if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv("rsxpos", info.crc, rt.data(), rt.size());
            }
        }
        // decalTwins: a box decal rebuilding positions from depth, at the left's pixel for its surface.
        if (Traits().decalTwins && info.model == spv::ExecutionModel::Fragment)
        {
            std::vector<uint32_t> dt;
            unsigned rows = 0;
            bool varying = false;
            if (PatchDecalFS(pCreateInfo->pCode, words, kStereoSet, dt, &rows, &varying))
            {
                VkShaderModuleCreateInfo dci = *pCreateInfo;
                dci.codeSize = dt.size() * 4;
                dci.pCode = dt.data();
                if (d->CreateShaderModule(device, &dci, pAllocator, &info.decalTwin) != VK_SUCCESS) info.decalTwin = VK_NULL_HANDLE;
                info.decalVarying = varying;
                Log("[decal] FS %08X: %u rows rebuild at the left's pixel, x from %s%s\n", info.crc, rows, varying ? "a varying" : "FragCoord", info.decalTwin ? "" : " (module creation FAILED)");
                if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv("decal", info.crc, dt.data(), dt.size());
            }
        }
        // glassTwins: a read of the scene behind glass moves to this eye's pixel; on its own where there is no velocity
        // target, inside the velocity twin where there is one.
        std::vector<uint32_t> glass;
        if (Traits().glassTwins && info.model == spv::ExecutionModel::Fragment)
        {
            unsigned sites = 0;
            if (PatchGlassFS(pCreateInfo->pCode, words, kStereoSet, glass, &sites))
            {
                VkShaderModuleCreateInfo gci = *pCreateInfo;
                gci.codeSize = glass.size() * 4;
                gci.pCode = glass.data();
                if (d->CreateShaderModule(device, &gci, pAllocator, &info.glassTwin) != VK_SUCCESS) info.glassTwin = VK_NULL_HANDLE;
                Log("[glass] FS %08X: %u scene read(s) moved to this eye's pixel%s\n", info.crc, sites, info.glassTwin ? "" : " (module creation FAILED)");
                if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv("glass", info.crc, glass.data(), glass.size());
                if (!info.glassTwin) glass.clear();
            }
        }
        if (Traits().velocityTwins && info.model == spv::ExecutionModel::Fragment && info.readsFragCoord)
        {
            std::vector<uint32_t> vt;
            if (PatchVelocityFS(glass.empty() ? pCreateInfo->pCode : glass.data(), glass.empty() ? words : glass.size(), kStereoSet, vt, Vk().VKSkipUnitW, Vk().VKUnitWEpsilon))
            {
                VkShaderModuleCreateInfo vci = *pCreateInfo;
                vci.codeSize = vt.size() * 4;
                vci.pCode = vt.data();
                if (d->CreateShaderModule(device, &vci, pAllocator, &info.veloTwin) != VK_SUCCESS)
                {
                    info.veloTwin = VK_NULL_HANDLE;
                    Log("[velo] FS %08X: twin rejected by the driver\n", info.crc);
                }
                if (Vk().VKDumpModifiedShaders >= 2) DumpSpirv("velofix", info.crc, vt.data(), vt.size());
            }
        }
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (info.remapTwin) ++d->statRemapShaders;
        d->shaderModules[*pShaderModule] = info;
    }
    return tail;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule)
{
    return CreateShaderModuleImpl(device, pCreateInfo, pAllocator, pShaderModule);
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyShaderModule(VkDevice device, VkShaderModule module, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    VkShaderModule original = VK_NULL_HANDLE, spriteTwin = VK_NULL_HANDLE, varyTwin = VK_NULL_HANDLE, camTwin = VK_NULL_HANDLE;
    VkShaderModule remapTwin = VK_NULL_HANDLE, rayTwin = VK_NULL_HANDLE, rsxTwin = VK_NULL_HANDLE, veloTwin = VK_NULL_HANDLE, glassTwin = VK_NULL_HANDLE, decalTwin = VK_NULL_HANDLE;
    VkShaderModule tileTwin = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->shaderModules.find(module);
        if (it != d->shaderModules.end())
        {
            original = it->second.original; spriteTwin = it->second.spriteTwin; varyTwin = it->second.varyTwin; camTwin = it->second.camTwin;
            remapTwin = it->second.remapTwin; rayTwin = it->second.rayTwin; rsxTwin = it->second.rsxTwin; veloTwin = it->second.veloTwin; glassTwin = it->second.glassTwin; decalTwin = it->second.decalTwin;
            tileTwin = it->second.tileTwin;
            d->shaderModules.erase(it);
        }
    }
    if (original) d->DestroyShaderModule(device, original, pAllocator);
    if (spriteTwin) d->DestroyShaderModule(device, spriteTwin, pAllocator);
    if (varyTwin) d->DestroyShaderModule(device, varyTwin, pAllocator);
    if (camTwin) d->DestroyShaderModule(device, camTwin, pAllocator);
    if (remapTwin) d->DestroyShaderModule(device, remapTwin, pAllocator);
    if (rayTwin) d->DestroyShaderModule(device, rayTwin, pAllocator);
    if (rsxTwin) d->DestroyShaderModule(device, rsxTwin, pAllocator);
    if (veloTwin) d->DestroyShaderModule(device, veloTwin, pAllocator);
    if (glassTwin) d->DestroyShaderModule(device, glassTwin, pAllocator);
    if (decalTwin) d->DestroyShaderModule(device, decalTwin, pAllocator);
    if (tileTwin) d->DestroyShaderModule(device, tileTwin, pAllocator);
    d->DestroyShaderModule(device, module, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateGraphicsPipelines(
    VkDevice device, VkPipelineCache cache, uint32_t count, const VkGraphicsPipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    DeviceData* d = FindDevice(device);
    // Only the last position-writing stage shifts: with a tessellation or geometry stage present, the vertex stage
    // reverts to its original module, unless the geometry stage is a passthrough that forwards the VS's Position.
    std::vector<VkGraphicsPipelineCreateInfo> infos(pCreateInfos, pCreateInfos + count);
    std::vector<std::vector<VkPipelineShaderStageCreateInfo>> stages(count);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (uint32_t i = 0; i < count; ++i)
        {
            const VkGraphicsPipelineCreateInfo& ci = pCreateInfos[i];
            bool later = false, fsRevert = false, fsCamera = false, screenRead = false;
            auto lit = d->pipelineLayouts.find(ci.layout);
            const bool hasStereoSet = lit != d->pipelineLayouts.end() && lit->second.patched;
            for (uint32_t k = 0; k < ci.stageCount; ++k)
            {
                auto it = d->shaderModules.find(ci.pStages[k].module);
                const bool passthrough = it != d->shaderModules.end() && it->second.gsPassthrough;
                if ((ci.pStages[k].stage & (VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT)) && !passthrough) later = true;
                if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && !hasStereoSet && it != d->shaderModules.end() && it->second.fsEyeOrigin) fsRevert = true;
                if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && hasStereoSet && it != d->shaderModules.end() && it->second.fsCamera) fsCamera = true;
                if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && it != d->shaderModules.end() && it->second.screenRead) screenRead = true;
            }
            // Never unshifted by vertex-attribute count: full-screen passes need the shift like every other pass.
            bool spriteVs = false, lampVs = false, fsTwin = false;
            uint32_t fsCrc = 0, vsCrc = 0;
            if (Vk().VKFsCameraWindows)
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                {
                    auto it = d->shaderModules.find(ci.pStages[k].module);
                    if (it == d->shaderModules.end()) continue;
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_VERTEX_BIT && it->second.spriteTwin) spriteVs = true;
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_VERTEX_BIT && LampTileVS(it->second.crc)) { lampVs = true; vsCrc = it->second.crc; }
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT) { fsTwin = it->second.camTwin != VK_NULL_HANDLE; fsCrc = it->second.crc; }
                }
            // Every light a lamp-tile VS draws takes the right eye's camera, so no lamp light stays on the left one.
            const bool lampTwin = lampVs && hasStereoSet && !fsCamera && fsTwin;
            if (lampVs && hasStereoSet)
                Log("[lamppipe] pipeline %u: vs=%08X fs=%08X camera=%s\n", i, vsCrc, fsCrc, fsCamera ? "allowlist" : lampTwin ? "twin" : "NONE");
            // VKLightRaysCamera: the FS of a screen-position VS (light volume, decal) takes its surface-ray twin.
            bool rayFs = false;
            if (Vk().VKLightRaysCamera && hasStereoSet && !later && !lampTwin)
            {
                bool ndcVs = false, rayTwin = false;
                uint32_t rvs = 0, rfs = 0;
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                {
                    auto it = d->shaderModules.find(ci.pStages[k].module);
                    if (it == d->shaderModules.end()) continue;
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_VERTEX_BIT && it->second.ndcVaryings) { ndcVs = true; rvs = it->second.crc; }
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && it->second.rayTwin) { rayTwin = true; rfs = it->second.crc; }
                }
                rayFs = ndcVs && rayTwin;
                if (rayFs) Log("[lightray] pipeline vs=%08X fs=%08X takes the surface-ray FS\n", rvs, rfs);
            }
            // VKComputeRemap: a decal FS (image stores after a depth rebuild) takes its right-camera twin.
            bool remapFs = false;
            if (Vk().VKComputeRemap && hasStereoSet && !lampTwin && !rayFs)
            {
                bool unitWVs = false;
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                {
                    auto it = d->shaderModules.find(ci.pStages[k].module);
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_VERTEX_BIT && it != d->shaderModules.end() && it->second.unitW) unitWVs = true;
                }
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                {
                    auto it = d->shaderModules.find(ci.pStages[k].module);
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && it != d->shaderModules.end() && it->second.remapTwin &&
                        (!it->second.remapUnitW || (unitWVs && !later)))
                        remapFs = true;
                }
            }
            // rsxPositionTwins: an FS rebuilding its position from WPOS takes the twin doing so at the left's pixel.
            bool rsxFs = false;
            if (Traits().rsxPositionTwins && hasStereoSet && !lampTwin && !rayFs && !remapFs)
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                {
                    auto it = d->shaderModules.find(ci.pStages[k].module);
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && it != d->shaderModules.end() && it->second.rsxTwin) rsxFs = true;
                }
            // velocityTwins: a scene pass with an RG16F velocity target at colour 1 beside B10G11R11 takes its FS twin
            // (2-colour post passes write B10G11R11 there).
            bool veloFs = false;
            if (Traits().velocityTwins && hasStereoSet && !lampTwin && !rayFs && !remapFs && !rsxFs)
                for (auto p = (const VkBaseInStructure*)ci.pNext; p; p = p->pNext)
                {
                    if (p->sType != VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) continue;
                    const auto* pr = (const VkPipelineRenderingCreateInfo*)p;
                    if (pr->colorAttachmentCount != 2 || pr->pColorAttachmentFormats[1] != VK_FORMAT_R16G16_SFLOAT) break;
                    for (uint32_t k = 0; k < ci.stageCount; ++k)
                    {
                        auto it = d->shaderModules.find(ci.pStages[k].module);
                        if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && it != d->shaderModules.end() && it->second.veloTwin) veloFs = true;
                    }
                    break;
                }
            if (veloFs)
            {
                static std::atomic<unsigned> s_velo{ 0 };
                const unsigned n = ++s_velo;
                if (n <= 3 || n % 1000 == 0) Log("[velo] %u pipelines' velocity takes off the stereo shift\n", n);
            }
            // glassTwins: the glass twin where the velocity twin did not take it; the row counts both.
            bool glassFs = false, glassRead = false;
            if (Traits().glassTwins && hasStereoSet && !lampTwin && !rayFs && !remapFs && !rsxFs)
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                {
                    auto it = d->shaderModules.find(ci.pStages[k].module);
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && it != d->shaderModules.end() && it->second.glassTwin) { glassRead = true; glassFs = !veloFs; }
                }
            if (glassRead)
            {
                static std::atomic<unsigned> s_glass{ 0 };
                const unsigned n = ++s_glass;
                if (n <= 3 || n % 100 == 0) Log("[glass] %u pipelines read the scene at this eye's pixel\n", n);
            }
            // decalTwins: x from a varying is R's pixel only if the VS's twin moved it; otherwise x - D subtracts the
            // shift twice.
            bool decalFs = false;
            if (Traits().decalTwins && hasStereoSet && !later && !lampTwin && !rayFs && !remapFs && !rsxFs && !veloFs && !glassFs)
            {
                const ShaderModuleInfo* dv = nullptr;
                const ShaderModuleInfo* df = nullptr;
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                {
                    auto it = d->shaderModules.find(ci.pStages[k].module);
                    if (it == d->shaderModules.end()) continue;
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_VERTEX_BIT) dv = &it->second;
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && it->second.decalTwin) df = &it->second;
                }
                if (df)
                {
                    decalFs = !df->decalVarying || (screenRead && dv && dv->varyTwin);
                    Log("[decal] pipeline vs=%08X fs=%08X %s\n", dv ? dv->crc : 0, df->crc, decalFs ? "rebuilds at the left's pixel" : "left alone: its VS does not move the varying");
                }
            }
            // VKTileListsLeft / VKScreenMarch: the FS's tile and march twin, where no other twin.
            bool tileFs = false;
            if ((Vk().VKTileListsLeft || Vk().VKScreenMarch) && hasStereoSet && !lampTwin && !rayFs && !remapFs && !rsxFs && !veloFs && !glassFs && !decalFs)
                for (uint32_t k = 0; k < ci.stageCount; ++k)
                {
                    auto it = d->shaderModules.find(ci.pStages[k].module);
                    if (ci.pStages[k].stage == VK_SHADER_STAGE_FRAGMENT_BIT && it != d->shaderModules.end() && it->second.tileTwin) tileFs = true;
                }
            if (!later && !fsRevert && !screenRead && !spriteVs && !lampTwin && !rayFs && !remapFs && !rsxFs && !veloFs && !glassFs && !decalFs && !tileFs) continue;
            stages[i].assign(ci.pStages, ci.pStages + ci.stageCount);
            for (auto& st : stages[i])
            {
                const bool vs = st.stage == VK_SHADER_STAGE_VERTEX_BIT && later;
                const bool fs = st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && fsRevert;
                auto it = d->shaderModules.find(st.module);
                if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && lampTwin && it != d->shaderModules.end() && it->second.camTwin)
                { st.module = it->second.camTwin; continue; }
                if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && rayFs && it != d->shaderModules.end() && it->second.rayTwin)
                { st.module = it->second.rayTwin; continue; }
                if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && remapFs && it != d->shaderModules.end() && it->second.remapTwin)
                { st.module = it->second.remapTwin; continue; }
                if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && rsxFs && it != d->shaderModules.end() && it->second.rsxTwin)
                { st.module = it->second.rsxTwin; continue; }
                if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && veloFs && it != d->shaderModules.end() && it->second.veloTwin)
                { st.module = it->second.veloTwin; continue; }
                if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && glassFs && it != d->shaderModules.end() && it->second.glassTwin)
                { st.module = it->second.glassTwin; continue; }
                if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && decalFs && it != d->shaderModules.end() && it->second.decalTwin)
                { st.module = it->second.decalTwin; continue; }
                if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && tileFs && it != d->shaderModules.end() && it->second.tileTwin)
                { st.module = it->second.tileTwin; continue; }
                if (st.stage == VK_SHADER_STAGE_VERTEX_BIT && !later && spriteVs && hasStereoSet && it != d->shaderModules.end() && it->second.spriteTwin)
                { st.module = it->second.spriteTwin; ++d->statSpritePipes; continue; }
                if (st.stage == VK_SHADER_STAGE_VERTEX_BIT && !later && screenRead && it != d->shaderModules.end() && it->second.varyTwin)
                { st.module = it->second.varyTwin; ++d->statScreenReadPipes; continue; }
                if (st.stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT && screenRead && it != d->shaderModules.end() && it->second.varyTwin)
                {
                    st.module = it->second.varyTwin; ++d->statScreenReadPipes;
                    Log("[screenread] pipeline: TES %08X takes its clip-copy twin\n", it->second.crc);
                    continue;
                }
                if (!vs && !fs) continue;
                if (it != d->shaderModules.end() && it->second.original) { st.module = it->second.original; if (vs) ++d->statVSReverted; }
            }
            infos[i].pStages = stages[i].data();
        }
    }
    VkResult r = d->CreateGraphicsPipelines(device, cache, count, infos.data(), pAllocator, pPipelines);
    std::lock_guard<std::shared_mutex> g(d->lock);
    for (uint32_t i = 0; i < count; ++i)
    {
        if (pPipelines[i] == VK_NULL_HANDLE) continue;
        PipelineInfo info;
        info.layout = pCreateInfos[i].layout;
        info.bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        // Libraries and pipeline binaries may carry their dynamic state elsewhere: those stay "declares everything".
        bool dynElsewhere = !pCreateInfos[i].stageCount || (pCreateInfos[i].flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR);
        for (auto p = (const VkBaseInStructure*)pCreateInfos[i].pNext; p; p = p->pNext)
            dynElsewhere |= p->sType == VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
        if (!dynElsewhere) info.dynDeclared = DynDeclaredMask(pCreateInfos[i].pDynamicState);
        if (!dynElsewhere)
        {
            info.coreDynDeclared = 0;
            const VkPipelineDynamicStateCreateInfo* ds = pCreateInfos[i].pDynamicState;
            for (uint32_t k = 0; ds && k < ds->dynamicStateCount; ++k)
                if (ds->pDynamicStates[k] <= VK_DYNAMIC_STATE_STENCIL_REFERENCE) info.coreDynDeclared |= 1u << (uint32_t)ds->pDynamicStates[k];
        }
        bool mesh = false, unseen = pCreateInfos[i].stageCount == 0;   // no stages: made from pipeline binaries
        for (uint32_t k = 0; k < pCreateInfos[i].stageCount; ++k)
        {
            if (pCreateInfos[i].pStages[k].stage & VK_SHADER_STAGE_MESH_BIT_EXT) mesh = true;
            if (pCreateInfos[i].pStages[k].module == VK_NULL_HANDLE) unseen = true;   // maintenance5 inline SPIR-V
        }
        if (mesh) ++d->statMeshPipelines;
        if (unseen) ++d->statUnseenPipelines;
        // shaderModules holds patched modules only; a reverted VS still
        // counts because the DS/GS that replaced it is patched.
        for (uint32_t k = 0; k < pCreateInfos[i].stageCount; ++k)
        {
            auto mit = d->shaderModules.find(pCreateInfos[i].pStages[k].module);
            if (mit == d->shaderModules.end()) continue;
            if (mit->second.selfShifted) info.shifted = true;
            // A mesh pipeline's position stage stands in for the VS in the logs (VKMeshDraws patches mesh modules).
            if ((mit->second.model == spv::ExecutionModel::Vertex || mit->second.model == spv::ExecutionModel::MeshEXT) && !info.vsCrc) info.vsCrc = mit->second.crc;
            if (mit->second.model == spv::ExecutionModel::Vertex) info.rsxInputs = mit->second.rsxInputs;
            if (mit->second.model == spv::ExecutionModel::Fragment) info.fsCrc = mit->second.crc;
            if (mit->second.posUsesMatrix)
            {
                info.posUsesMatrix = true;
                info.posSet = mit->second.posSet;
                info.posBinding = mit->second.posBinding;
                info.posOffset = mit->second.posOffset;
                info.posMask = mit->second.posMask;
                info.vsCrc = mit->second.crc;
            }
        }
        // Both pointers are ignored, and may be garbage, when rasterization is discarded.
        const bool raster = pCreateInfos[i].pRasterizationState && !pCreateInfos[i].pRasterizationState->rasterizerDiscardEnable;
        if (const VkPipelineDepthStencilStateCreateInfo* ds = raster ? pCreateInfos[i].pDepthStencilState : nullptr)
        {
            info.depthOp = ds->depthTestEnable ? (int)ds->depthCompareOp : -1;
            info.depthWrite = ds->depthWriteEnable != 0;
        }
        if (const VkPipelineColorBlendStateCreateInfo* cb = raster ? pCreateInfos[i].pColorBlendState : nullptr)
            if (cb->attachmentCount && cb->pAttachments)
            {
                info.colorWrite = cb->pAttachments[0].colorWriteMask != 0;
                info.blend = cb->pAttachments[0].blendEnable != 0;
            }
        d->pipelines[pPipelines[i]] = info;
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateComputePipelines(
    VkDevice device, VkPipelineCache cache, uint32_t count, const VkComputePipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    DeviceData* d = FindDevice(device);
    // VKComputeRemap: the twin needs set 8, so only a layout that carries it takes the twin.
    std::vector<VkComputePipelineCreateInfo> infos(pCreateInfos, pCreateInfos + count);
    std::vector<uint8_t> remap(count, 0);
    if (Vk().VKComputeRemap || Traits().computeRules)
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        for (uint32_t i = 0; i < count; ++i)
        {
            auto mit = d->shaderModules.find(infos[i].stage.module);
            auto lit = d->pipelineLayouts.find(infos[i].layout);
            if (mit == d->shaderModules.end() || !mit->second.remapTwin || lit == d->pipelineLayouts.end() || !lit->second.patched) continue;
            infos[i].stage.module = mit->second.remapTwin;
            remap[i] = 1;
        }
    }
    VkResult r = d->CreateComputePipelines(device, cache, count, infos.data(), pAllocator, pPipelines);
    std::lock_guard<std::shared_mutex> g(d->lock);
    for (uint32_t i = 0; i < count; ++i)
    {
        if (pPipelines[i] == VK_NULL_HANDLE) continue;
        PipelineInfo info;
        info.layout = pCreateInfos[i].layout;
        info.bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
        info.remapCS = remap[i] != 0;
        if (pCreateInfos[i].stage.module == VK_NULL_HANDLE) ++d->statUnseenPipelines;   // inline SPIR-V or binaries
        auto mit = d->shaderModules.find(pCreateInfos[i].stage.module);
        if (mit != d->shaderModules.end()) { info.storageWrites = mit->second.storageWrites; info.csCrc = mit->second.crc; }
        d->pipelines[pPipelines[i]] = info;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->pipelines.erase(pipeline);
        d->objGen.fetch_add(1, std::memory_order_release);
    }
    d->DestroyPipeline(device, pipeline, pAllocator);
}

const HookEntry g_pipelineHooks[] = {
    WIZ3D_HOOK(CreateShaderModule), WIZ3D_HOOK(DestroyShaderModule),
    WIZ3D_HOOK(CreateGraphicsPipelines), WIZ3D_HOOK(CreateComputePipelines), WIZ3D_HOOK(DestroyPipeline),
    { nullptr, nullptr },
};

}} // namespace wiz3d::vk
