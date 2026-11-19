/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Layer core: x += sep * (w - conv) in every vertex stage from our block at kStereoSet, right-eye siblings for
// scene targets, each pass into them replayed with right-eye bindings, and the eyes composited at present.

#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <spirv/unified1/spirv.hpp11>
#include "FsPatch.h"
#include "ProfileVK.h"

namespace wiz3d { namespace vk {

// Every instance and physical-device function the layer calls, resolved through the next layer's
// vkGetInstanceProcAddr.
#define WIZ3D_VK_INSTANCE_FUNCS(X)               \
    X(DestroyInstance)                           \
    X(EnumeratePhysicalDevices)                  \
    X(GetPhysicalDeviceProperties)               \
    X(GetPhysicalDeviceMemoryProperties)         \
    X(GetPhysicalDeviceFeatures)                 \
    X(GetPhysicalDeviceFormatProperties)         \
    X(GetPhysicalDeviceQueueFamilyProperties)    \
    X(EnumerateDeviceExtensionProperties)        \
    X(GetPhysicalDeviceProperties2)              \
    X(GetPhysicalDeviceFeatures2)                \
    X(CreateWin32SurfaceKHR)                     \
    X(CreateDevice)

// Resolved through the next layer's vkGetDeviceProcAddr. Every device-level
// function the layer forwards or issues itself.
#define WIZ3D_VK_DEVICE_FUNCS(X)                 \
    X(DestroyDevice)                             \
    X(GetDeviceQueue)                            \
    X(GetDeviceQueue2)                           \
    X(QueueSubmit)                               \
    X(QueueBindSparse)                           \
    X(QueueWaitIdle)                             \
    X(DeviceWaitIdle)                            \
    X(CreateSwapchainKHR)                        \
    X(DestroySwapchainKHR)                       \
    X(GetSwapchainImagesKHR)                     \
    X(AcquireNextImageKHR)                       \
    X(QueuePresentKHR)                           \
    X(AllocateMemory)                            \
    X(FreeMemory)                                \
    X(MapMemory)                                 \
    X(UnmapMemory)                               \
    X(CreateBuffer)                              \
    X(DestroyBuffer)                             \
    X(GetBufferMemoryRequirements)               \
    X(BindBufferMemory)                          \
    X(BindBufferMemory2)                         \
    X(QueueSubmit2)                              \
    X(CreateImage)                               \
    X(DestroyImage)                              \
    X(GetImageMemoryRequirements)                \
    X(GetImageMemoryRequirements2)               \
    X(BindImageMemory)                           \
    X(BindImageMemory2)                          \
    X(CreateImageView)                           \
    X(DestroyImageView)                          \
    X(CreateSampler)                             \
    X(DestroySampler)                            \
    X(CreateFramebuffer)                         \
    X(DestroyFramebuffer)                        \
    X(CreateRenderPass)                          \
    X(CreateRenderPass2)                         \
    X(DestroyRenderPass)                         \
    X(CreateShaderModule)                        \
    X(DestroyShaderModule)                       \
    X(CreateDescriptorSetLayout)                 \
    X(DestroyDescriptorSetLayout)                \
    X(CreatePipelineLayout)                      \
    X(DestroyPipelineLayout)                     \
    X(CreateGraphicsPipelines)                   \
    X(CreateComputePipelines)                    \
    X(DestroyPipeline)                           \
    X(CreateDescriptorPool)                      \
    X(DestroyDescriptorPool)                     \
    X(ResetDescriptorPool)                       \
    X(AllocateDescriptorSets)                    \
    X(FreeDescriptorSets)                        \
    X(UpdateDescriptorSets)                      \
    X(CreateDescriptorUpdateTemplate)            \
    X(DestroyDescriptorUpdateTemplate)           \
    X(UpdateDescriptorSetWithTemplate)           \
    X(CreateCommandPool)                         \
    X(DestroyCommandPool)                        \
    X(ResetCommandPool)                          \
    X(AllocateCommandBuffers)                    \
    X(FreeCommandBuffers)                        \
    X(BeginCommandBuffer)                        \
    X(EndCommandBuffer)                          \
    X(ResetCommandBuffer)                        \
    X(CreateFence)                               \
    X(DestroyFence)                              \
    X(WaitForFences)                             \
    X(ResetFences)                               \
    X(CreateSemaphore)                           \
    X(DestroySemaphore)                          \
    X(CmdBindPipeline)                           \
    X(CmdSetViewport)                            \
    X(CmdSetScissor)                             \
    X(CmdSetLineWidth)                           \
    X(CmdSetDepthBias)                           \
    X(CmdSetBlendConstants)                      \
    X(CmdSetDepthBounds)                         \
    X(CmdSetStencilCompareMask)                  \
    X(CmdSetStencilWriteMask)                    \
    X(CmdSetStencilReference)                    \
    X(CmdBindDescriptorSets)                     \
    X(CmdBindIndexBuffer)                        \
    X(CmdBindVertexBuffers)                      \
    X(CmdBindVertexBuffers2)                     \
    X(CmdDraw)                                   \
    X(CmdDrawIndexed)                            \
    X(CmdDrawIndirect)                           \
    X(CmdDrawIndexedIndirect)                    \
    X(CmdDrawIndirectCount)                      \
    X(CmdDrawIndexedIndirectCount)               \
    X(CmdDispatch)                               \
    X(CmdDispatchIndirect)                       \
    X(CmdCopyBuffer)                             \
    X(CmdCopyImage)                              \
    X(CmdBlitImage)                              \
    X(CmdCopyBufferToImage)                      \
    X(CmdCopyImageToBuffer)                      \
    X(CmdUpdateBuffer)                           \
    X(CmdFillBuffer)                             \
    X(CmdClearColorImage)                        \
    X(CmdClearDepthStencilImage)                 \
    X(CmdClearAttachments)                       \
    X(CmdResolveImage)                           \
    X(CmdSetEvent)                               \
    X(CmdResetEvent)                             \
    X(CmdWaitEvents)                             \
    X(CmdPipelineBarrier)                        \
    X(CmdBeginQuery)                             \
    X(CmdEndQuery)                               \
    X(CmdResetQueryPool)                         \
    X(CmdWriteTimestamp)                         \
    X(CreateQueryPool)                           \
    X(DestroyQueryPool)                          \
    X(GetQueryPoolResults)                       \
    X(CmdCopyQueryPoolResults)                   \
    X(CmdPushConstants)                          \
    X(CmdBeginRenderPass)                        \
    X(CmdNextSubpass)                            \
    X(CmdEndRenderPass)                          \
    X(CmdBeginRenderPass2)                       \
    X(CmdNextSubpass2)                           \
    X(CmdEndRenderPass2)                         \
    X(CmdBeginRendering)                         \
    X(CmdEndRendering)                           \
    X(CmdExecuteCommands)                        \
    X(CmdPipelineBarrier2)                       \
    X(CmdPushDescriptorSetKHR)                   \
    X(CmdPushDescriptorSetWithTemplateKHR)       \
    X(GetMemoryWin32HandlePropertiesKHR)

// Resolved like the list above plus the EXT alias, and legitimately null when the game did not enable the
// extension: extended dynamic state 1-3, vertex input, feedback loop, maintenance5, copy commands 2, semaphore import.
#define WIZ3D_VK_OPTIONAL_DEVICE_FUNCS(X)        \
    X(GetBufferDeviceAddress)                    \
    X(ResetQueryPool)                            \
    X(CmdSetCullMode)                            \
    X(CmdSetFrontFace)                           \
    X(CmdSetPrimitiveTopology)                   \
    X(CmdSetViewportWithCount)                   \
    X(CmdSetScissorWithCount)                    \
    X(CmdSetDepthTestEnable)                     \
    X(CmdSetDepthWriteEnable)                    \
    X(CmdSetDepthCompareOp)                      \
    X(CmdSetDepthBoundsTestEnable)               \
    X(CmdSetStencilTestEnable)                   \
    X(CmdSetStencilOp)                           \
    X(CmdSetRasterizerDiscardEnable)             \
    X(CmdSetDepthBiasEnable)                     \
    X(CmdSetPrimitiveRestartEnable)              \
    X(CmdSetPatchControlPointsEXT)               \
    X(CmdSetLogicOpEXT)                          \
    X(CmdSetTessellationDomainOriginEXT)         \
    X(CmdSetDepthClampEnableEXT)                 \
    X(CmdSetPolygonModeEXT)                      \
    X(CmdSetRasterizationSamplesEXT)             \
    X(CmdSetSampleMaskEXT)                       \
    X(CmdSetAlphaToCoverageEnableEXT)            \
    X(CmdSetAlphaToOneEnableEXT)                 \
    X(CmdSetLogicOpEnableEXT)                    \
    X(CmdSetColorBlendEnableEXT)                 \
    X(CmdSetColorBlendEquationEXT)               \
    X(CmdSetColorWriteMaskEXT)                   \
    X(CmdSetColorBlendAdvancedEXT)               \
    X(CmdSetRasterizationStreamEXT)              \
    X(CmdSetConservativeRasterizationModeEXT)    \
    X(CmdSetExtraPrimitiveOverestimationSizeEXT) \
    X(CmdSetDepthClipEnableEXT)                  \
    X(CmdSetSampleLocationsEnableEXT)            \
    X(CmdSetProvokingVertexModeEXT)              \
    X(CmdSetLineRasterizationModeEXT)            \
    X(CmdSetLineStippleEnableEXT)                \
    X(CmdSetDepthClipNegativeOneToOneEXT)        \
    X(CmdSetColorWriteEnableEXT)                 \
    X(CmdSetVertexInputEXT)                      \
    X(CmdSetAttachmentFeedbackLoopEnableEXT)     \
    X(CmdBindIndexBuffer2)                       \
    X(CmdCopyImage2)                             \
    X(CmdBlitImage2)                             \
    X(CmdResolveImage2)                          \
    X(CmdCopyBufferToImage2)                     \
    X(CmdCopyImageToBuffer2)                     \
    X(CmdCopyBuffer2)                            \
    X(CmdDrawMeshTasksEXT)                       \
    X(CmdDrawMeshTasksIndirectEXT)               \
    X(CmdDrawMeshTasksIndirectCountEXT)          \
    X(CmdBeginConditionalRenderingEXT)           \
    X(CmdEndConditionalRenderingEXT)             \
    X(CreateBufferView)                          \
    X(DestroyBufferView)                         \
    X(ImportSemaphoreWin32HandleKHR)

struct InstanceData
{
    VkInstance                 instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr  nextGIPA = nullptr;
    uint32_t                   apiVersion = 0;
    std::mutex                 lock;
    std::unordered_map<VkSurfaceKHR, HWND> surfaces;   // the weaver needs the window
#define X(name) PFN_vk##name name = nullptr;
    WIZ3D_VK_INSTANCE_FUNCS(X)
#undef X
};

// Our descriptor set index for the stereo uniform block; game layouts are
// padded with empty sets up to it. NVIDIA allows 32 bound sets.
static const uint32_t kStereoSet = 8;

struct ImageInfo
{
    VkImageCreateInfo ci = {};           // pNext dropped
    bool              nvDedicated = false;
    VkImage           sibling = VK_NULL_HANDLE;
    VkDeviceMemory    siblingMem = VK_NULL_HANDLE;   // owned only when not a mirror
    VkDeviceSize      siblingMemSize = 0;             // of the owned allocation
    bool              siblingInMirror = false;
    VkDeviceMemory    gameMem = VK_NULL_HANDLE;
    VkDeviceSize      gameOff = 0;
    bool              swapchain = false;
    // Whether anything has written the sibling: an unwritten one reads as cleared memory in the right eye.
    bool              siblingWritten = false;
    // Whether the game wrote the image itself: only a written image with a blank sibling is a real gap.
    bool              everWritten = false;
    // readColorBuffers (KeepRightFrame): the frames this target was last drawn in both eyes and last read back;
    // cpuLoaded: a transfer helper holding bytes the CPU just wrote.
    unsigned          rightDrawnFrame = ~0u;
    unsigned          readBackFrame = ~0u;
    bool              cpuLoaded = false;
    // Reloaded and not drawn in a later frame: a read-back now is the finished picture, not the CPU's input.
    unsigned          reloadFrame = ~0u;
    bool              finished = false;
    const char*       monoWhy = "";                   // ShouldDoubleImage's reason when it stayed mono (a literal)
};

// Game allocations that stereo images bind into get a mirror of the same size and type; siblings bind into it at the
// same offsets, so transient allocators aliasing several images over one range work for the right eye too.
struct MemoryInfo
{
    VkDeviceSize   size = 0;
    uint32_t       typeIndex = 0;
    bool           dedicated = false;
    VkMemoryAllocateFlags allocFlags = 0;  // repeated on the mirror for buffer twins (device-address memory)
    VkDeviceMemory mirror = VK_NULL_HANDLE;
    struct Range { VkDeviceSize off, size; VkImage image; };
    std::vector<Range> stereoRanges;       // live sibling binds, for alias counting
    // Host mapping, so uniform ranges can be read at submit (Constants.cpp)
    void*          mapped = nullptr;
    VkDeviceSize   mapOff = 0, mapSize = 0;
    unsigned       imageBinds = 0;         // images the game bound into this allocation (>1 = aliasing)
    // Mapped by the layer because the game never did, so its uniform ranges can be read and copied too.
    bool           layerMapped = false;
};

// Game buffers, for the constants path: a buffer carrying a camera gets a host-visible sibling that the right eye's
// shadow sets reference; its used ranges are copied and patched at submit.
struct BufferInfo
{
    VkDeviceSize       size = 0;
    VkBufferUsageFlags usage = 0;
    VkDeviceMemory     memory = VK_NULL_HANDLE;
    VkDeviceSize       offset = 0;
    VkBuffer           sibling = VK_NULL_HANDLE;
    VkDeviceMemory     siblingMem = VK_NULL_HANDLE;
    void*              siblingMapped = nullptr;
    // claimMirror: a second sibling only claimed camera keys write and read, so no unpatched copy lands on it.
    VkBuffer           claimSibling = VK_NULL_HANDLE;
    VkDeviceMemory     claimSiblingMem = VK_NULL_HANDLE;
    void*              claimSiblingMapped = nullptr;
    // Right-eye twin (VKBufferTwins): sparse buffers get a sparse twin whose binds mirror the game's
    // into mirror memory; GPU-only buffers get one bound into the mirror of their memory at the same offset.
    VkBufferCreateFlags flags = 0;
    VkBuffer           twin = VK_NULL_HANDLE;
};

struct ImageViewInfo
{
    VkImage     image = VK_NULL_HANDLE;
    VkImageView sibling = VK_NULL_HANDLE;
    VkImageSubresourceRange range = {};   // counts resolved: what a mono pass wrote here, copied to the sibling
};

// A doubled attachment of a framebuffer that stays mono because another attachment was refused.
struct MixedAttachment
{
    uint32_t index;                    // in the render pass's attachments
    VkImage  image, sibling;
    VkImageSubresourceRange range;
    uint32_t w, h;
};

struct FramebufferInfo
{
    VkFramebuffer sibling = VK_NULL_HANDLE;
    VkRenderPass  renderPass = VK_NULL_HANDLE;
    std::vector<VkImage> doubled;      // game images behind the sibling attachments
    std::vector<MixedAttachment> mixed;
    uint32_t      width = 0, height = 0, attachments = 0;   // names the target a pipeline drew into
    std::vector<VkImageView> views;    // the game's attachments
};

// A sibling whose game image has been destroyed: the replay may still have work reading it in flight.
struct RetiredImage
{
    VkImage        image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    unsigned       frame = 0;
};

struct DescriptorSetLayoutInfo
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
};

// A matrix inside a block at its true byte offset and stride: scalar-layout blocks put matrices at strides the
// 16-byte register scan cannot see, so the constants path tests these directly.
struct BlockMatrix
{
    uint32_t set, binding;
    uint32_t offset;      // bytes from the start of the block
    uint32_t stride;      // bytes between columns, or rows when row-major
    uint8_t  rows, cols;
    bool     colMajor;
    bool     ssbo;        // storage buffer, not uniform: the constants path never reads these
};

struct PipelineLayoutInfo
{
    bool     patched = false;            // carries our set at kStereoSet
    uint32_t gameSetCount = 0;
    std::vector<uint32_t> dynCounts;     // dynamic offsets each game set consumes
};

struct ShaderModuleInfo
{
    spv::ExecutionModel model = spv::ExecutionModel::Max;
    VkShaderModule      original = VK_NULL_HANDLE;   // unpatched twin (vertex stages)
    // Whether gl_Position is built from a uniform matrix, and where: only then can patching the projection move it.
    bool                selfShifted = true;   // carries the SPIR-V shift; false when constants serve it instead
    bool                posUsesMatrix = false;
    uint32_t            posSet = 0, posBinding = 0, posOffset = 0;
    uint64_t            posMask = 0;      // bit per 16-byte register the position chain multiplies by
    std::vector<uint64_t> storageWrites;  // (set << 32 | binding) of writable storage buffers
    uint32_t            crc = 0;          // names the vk_<crc>_orig.spv dump and the pipeline rows
    bool                fsEyeOrigin = false;   // FS reads set 8 (reconstruction, windows); `original` is unpatched
    bool                fsCamera = false;      // patched by PatchCameraWindowsFS itself
    VkShaderModule      spriteTwin = VK_NULL_HANDLE;   // VS moved by its depth's parallax in the right eye
    VkShaderModule      varyTwin = VK_NULL_HANDLE;  // VS with its clip varyings shifted (VKScreenReadVaryings)
    VkShaderModule      camTwin = VK_NULL_HANDLE;   // FS with the camera-window patch, for a lamp-tile VS
    VkShaderModule      remapTwin = VK_NULL_HANDLE; // CS reading set 8 for the right run's camera (VKComputeRemap)
    VkShaderModule      rayTwin = VK_NULL_HANDLE;   // FS with its ray varyings moved to the surface (VKLightRaysCamera)
    VkShaderModule      rsxTwin = VK_NULL_HANDLE;   // FS rebuilding its position at the left's pixel (rsxPositionTwins)
    VkShaderModule      veloTwin = VK_NULL_HANDLE;  // FS taking the shift off its velocity output (velocityTwins)
    VkShaderModule      glassTwin = VK_NULL_HANDLE; // FS reading the scene behind glass at this pixel (glassTwins)
    VkShaderModule      decalTwin = VK_NULL_HANDLE; // FS rebuilding a box decal at the left's pixel (decalTwins)
    VkShaderModule      tileTwin = VK_NULL_HANDLE;  // FS: tile lists at the left's pixel, marches to this eye's end
    bool                decalVarying = false;       // its screen x comes from a varying its VS must move too
    unsigned            rsxInputs = 0;              // VS: the vertex attributes it reads (flatNoDepthDraws)
    unsigned            ndcVaryings = 0;            // VS outputs that are Position.xy / w (VKShiftNdcVaryings)
    bool                screenRead = false;        // FS reads a texture at a varying's x/w (ReadsScreenAtClipVarying)
    bool                readsFragCoord = false;     // FS decorates BuiltIn FragCoord (the [velo] rows)
    unsigned            posVaryings = 0;            // VS outputs storing gl_Position's own values (PatchSpirv)
    bool                gsPassthrough = false;      // never-emitting GS (NV passthrough): forwards the VS's Position
    bool                unitW = false;              // VS: Position.w is only ever the constant 1 (full-screen passes)
    bool                remapUnitW = false;         // remapTwin reads perspective inputs: unit-w VS pipelines only
};

struct PipelineInfo
{
    VkPipelineLayout    layout = VK_NULL_HANDLE;
    VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    uint64_t            dynDeclared = ~0ull;  // a bit per Dyn it declares dynamic (VKDynByPipeline); all when not known
    uint32_t            coreDynDeclared = ~0u;  // per core VkDynamicState up to stencil ref (staticPipelineState)
    unsigned            rsxInputs = 0;          // its VS's vertex attributes (flatNoDepthDraws)
    // flatNoDepthDraws: depth-off but drawn before last frame's last depth-tested draw: a world sprite, still shifted
    bool                midFrameSprite = false;
    bool                shifted = false;      // a position-writing stage carries the stereo patch
    bool                posUsesMatrix = false;            // position from a camera matrix: constants-patched
    uint32_t            posSet = 0, posBinding = 0, posOffset = 0;
    uint64_t            posMask = 0;      // bit per 16-byte register the position chain multiplies by
    std::vector<uint64_t> storageWrites;  // compute: writable storage buffers, for the replay gate
    // Whether the last submit patched this pipeline's camera range (the replay is recorded before the submit): a served
    // draw binds the sep=0 set, an unserved one keeps the SPIR-V self-shift.
    bool                served = false;
    unsigned            servedApplied = 0, servedSkipped = 0;
    // Failed verdicts in a row: one rejected range must not flip a pipeline's mechanism for a frame.
    unsigned            missStreak = 0;
    // Its shaders' CRCs, for the log rows.
    uint32_t            vsCrc = 0, fsCrc = 0;
    uint32_t            csCrc = 0;                                      // compute: the module's CRC
    bool                remapCS = false;                                // compute: from a remapTwin (per-eye set 8)
    bool                triRefusedNamed = false;                        // screen triangle: its refusal was logged
    int                 depthOp = -1; bool depthWrite = false, colorWrite = true, blend = false;
};

// A buffer descriptor the set holds, kept so the range can be read at submit and moved to a sibling once one exists.
struct UboRef
{
    uint32_t     binding = 0, element = 0;
    bool         dynamic = false;
    bool         uniform = true;                   // false: dynamic storage buffer, kept only for offset order
    VkBuffer     buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, range = 0;
    VkBuffer     shadowBuffer = VK_NULL_HANDLE;    // what the shadow set currently references
};

// A game descriptor pool and the 1:1 mirror its shadows come from, so a pool reset frees them in one call.
struct PoolInfo
{
    VkDescriptorPool mirror = VK_NULL_HANDLE;
    std::unordered_set<VkDescriptorSet> sets;   // this pool's sets, for an O(pool) forget
};

struct DescriptorSetInfo
{
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;   // the game's pool
    VkDescriptorSet       shadow = VK_NULL_HANDLE; // right-eye sibling, from our pools
    VkDescriptorPool      shadowPool = VK_NULL_HANDLE;  // where `shadow` came from; null = legacy shadowPools
    // Every set owns a shadow (pools are mirrored); this makes it visible to the replay.
    bool                  stereoAny = false;
    bool                  stereoImages = false;    // holds a view of a doubled image (not just a camera block)
    bool                  stereoStorage = false;   // holds a STORAGE view of a doubled image (a per-eye write target)
    // Bindings the game has written, so a new shadow seeds from those alone;
    // a binding past 63 sets `wideBindings`, which copies all of them.
    uint64_t              writtenMask = 0;
    bool                  wideBindings = false;
    std::vector<UboRef>   ubos;
};

// The descriptor-set map, sharded by handle with a lock per shard (lock order: device, shard). A returned info stays
// valid after the shard unlocks, as Vulkan forbids freeing a set the app still uses.
class SetTable
{
public:
    using Map = std::unordered_map<VkDescriptorSet, DescriptorSetInfo>;
    using Node = Map::node_type;
    DescriptorSetInfo* Find(VkDescriptorSet s)
    {
        Shard& sh = For(s);
        std::shared_lock<std::shared_mutex> g(sh.m);
        auto it = sh.map.find(s);
        return it == sh.map.end() ? nullptr : &it->second;
    }
    // A handle the driver reuses replaces its stale entry, which goes to `dead` for the caller to free unlocked.
    void Insert(Node&& node, std::vector<Node>& dead)
    {
        Shard& sh = For(node.key());
        std::lock_guard<std::shared_mutex> g(sh.m);
        auto it = sh.map.find(node.key());
        if (it != sh.map.end()) dead.push_back(sh.map.extract(it));
        sh.map.insert(std::move(node));
    }
    void Extract(VkDescriptorSet s, std::vector<Node>& dead)
    {
        Shard& sh = For(s);
        std::lock_guard<std::shared_mutex> g(sh.m);
        auto it = sh.map.find(s);
        if (it != sh.map.end()) dead.push_back(sh.map.extract(it));
    }
    template <class F> void ForEach(F f)
    {
        for (Shard& sh : shards) { std::lock_guard<std::shared_mutex> g(sh.m); for (auto& kv : sh.map) f(kv.first, kv.second); }
    }
    template <class P> void ExtractIf(P pred, std::vector<Node>& dead)
    {
        for (Shard& sh : shards)
        {
            std::lock_guard<std::shared_mutex> g(sh.m);
            for (auto it = sh.map.begin(); it != sh.map.end();)
            {
                auto next = std::next(it);
                if (pred(it->second)) dead.push_back(sh.map.extract(it));
                it = next;
            }
        }
    }
private:
    struct Shard { std::shared_mutex m; Map map; };
    static const size_t kShards = 64;
    Shard shards[kShards];
    Shard& For(VkDescriptorSet s) { return shards[(((uint64_t)s >> 4) * 0x9E3779B97F4A7C15ull) >> 58]; }
};

struct DescriptorUpdateTemplateInfo
{
    std::vector<VkDescriptorUpdateTemplateEntry> entries;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineBindPoint   bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
    uint32_t              set = 0;
    bool                  pushDescriptors = false;
};

struct WeaveState;                       // Weave.cpp (D3D11 interop, per swapchain)
struct WeaveDevice;                      // Weave.cpp (D3D11 device + SR weaver, per device)

struct SwapchainInfo
{
    VkSwapchainCreateInfoKHR ci = {};    // pNext dropped
    HWND                     hwnd = nullptr;
    std::vector<VkImage>     images;
    // Composite scratch: one command buffer + fence per swapchain image.
    std::vector<VkCommandBuffer> cmds;
    std::vector<VkFence>         fences;
    std::vector<VkSemaphore>     done;
    VkImage        scratch = VK_NULL_HANDLE;     // left copy, the blit source
    VkDeviceMemory scratchMem = VK_NULL_HANDLE;
    VkCommandPool  pool = VK_NULL_HANDLE;
    uint32_t       cmdFamily = UINT32_MAX;       // the queue family `cmds` and the weave's were allocated for
    WeaveState*    weave = nullptr;
};

struct CmdState;                         // CommandBuffer.cpp

// Layer CPU time per area (QPC ticks, driver calls excluded where the hook can pause) and memory the layer adds,
// reported with the frame stats; atomics, as games record on several threads.
struct PerfCounters
{
    std::atomic<long long> draws{0}, binds{0}, replay{0}, descriptors{0}, descWait{0}, descCreate{0}, constants{0}, composite{0};
    std::atomic<long long> descLookup{0}, descWrite{0};   // the two halves of the descriptor path either side of create
    std::atomic<long long> descFree{0};                   // pool reset / set free
    // Time `lock` is HELD, not waited for: the serialised critical path.
    std::atomic<long long> descHeld{0}, constHeld{0};
    // Exclusive holds of `lock` on the descriptor allocate / free paths, which descriptor writers wait behind.
    std::atomic<long long> allocHeld{0}, freeHeld{0};
    // ConstantsSubmit and MirrorWrites: the wait for the device lock, then the patch pass under the constants lock.
    std::atomic<long long> constWait{0}, constApply{0}, descWaitW{0};
    // constApply per range: the uncached read of the game's block, the camera math, the key lookup, the GPU record.
    std::atomic<long long> constRead{0}, constMath{0}, constKey{0}, constRec{0};
    std::atomic<long long> weaveA{0}, weaveB{0}, weaveC{0};
    std::atomic<long long> vramAdded{0}, vramGame{0};   // bytes: sibling/mirror allocations, game allocations
    long long lastPresent = 0;
    std::vector<float> frameMs;                          // one per present since the last report
};

// Pixels cropped off one edge of both eyes before compositing: the right eye's far field moves by `sep` NDC
// (width/2 px each), leaving that much of one edge undrawn (the right one when sep is negative). `mult` is VKEdgeCrop.
inline int32_t EdgeCropPx(float sep, int32_t width, float mult)
{
    if (mult <= 0.f || width <= 0) return 0;
    const float s = sep < 0.f ? -sep : sep;
    int32_t px = (int32_t)(s * (float)width * 0.5f * mult + 0.5f);
    if (px < 0) px = 0;
    if (px > width / 4) px = width / 4;   // a runaway sep must not eat the frame
    return px;
}

inline long long PerfNow() { LARGE_INTEGER t; QueryPerformanceCounter(&t); return t.QuadPart; }
inline double PerfMs(long long ticks) { static LARGE_INTEGER f = [] { LARGE_INTEGER v; QueryPerformanceFrequency(&v); return v; }(); return ticks * 1000.0 / (double)f.QuadPart; }

struct PerfScope
{
    std::atomic<long long>& acc;
    long long t0, held = 0;
    explicit PerfScope(std::atomic<long long>& a) : acc(a), t0(PerfNow()) {}
    void pause() { held += PerfNow() - t0; }
    void resume() { t0 = PerfNow(); }
    ~PerfScope() { acc.fetch_add(held + (PerfNow() - t0), std::memory_order_relaxed); }
};

struct DeviceData
{
    VkDevice                   device = VK_NULL_HANDLE;
    VkPhysicalDevice           physDev = VK_NULL_HANDLE;
    InstanceData*              inst = nullptr;
    PFN_vkGetDeviceProcAddr    nextGDPA = nullptr;
    PFN_vkSetDeviceLoaderData  setLoaderData = nullptr;   // the loader's, for StampCommandBuffers
    VkPhysicalDeviceProperties props = {};
    VkPhysicalDeviceMemoryProperties memProps = {};
    VkExtent2D                 bbSize = {};        // last created swapchain extent
    unsigned                   frames = 0;         // presents seen
    bool                       stereoEnabled = false;
    std::vector<RetiredImage>  retiredImages;   // guarded by `lock`, drained per present
    unsigned                   swapImages = 0;  // frames in flight cannot exceed this
    // Storage buffers a compute dispatch writes and needs a sibling for (the gate refuses it until then); `lock`.
    std::vector<VkBuffer>      pendingBufferSiblings;
    struct OsdState*           osd = nullptr;         // the on-screen display (OSDRendererVK.cpp); `lock`
#define X(name) PFN_vk##name name = nullptr;
    WIZ3D_VK_DEVICE_FUNCS(X)
    WIZ3D_VK_OPTIONAL_DEVICE_FUNCS(X)
#undef X

    // Generation counters let hot hooks cache lookups without the lock: cmdGen bumps when a command buffer's state is
    // freed, objGen when a pipeline or pipeline layout is destroyed (handles may be reused).
    std::atomic<uint64_t> cmdGen{1}, objGen{1};

    // Guards every map below. Shared only in MirrorWrites (the app synchronises updates of a set, so its per-set
    // writes race with nothing) and where noted.
    std::shared_mutex lock;
    std::unordered_map<VkImage, ImageInfo>                      images;
    std::unordered_map<VkImageView, ImageViewInfo>              views;
    std::unordered_map<VkFramebuffer, FramebufferInfo>          framebuffers;
    std::unordered_map<VkDeviceMemory, MemoryInfo>              memory;
    std::unordered_map<VkBuffer, BufferInfo>                    buffers;
    std::unordered_map<VkDescriptorSetLayout, DescriptorSetLayoutInfo> setLayouts;
    std::unordered_map<VkPipelineLayout, PipelineLayoutInfo>    pipelineLayouts;
    std::unordered_map<VkPipeline, PipelineInfo>                pipelines;
    std::unordered_map<VkShaderModule, ShaderModuleInfo>        shaderModules;
    SetTable                                                    sets;    // its own locks, not `lock`
    std::unordered_map<VkDescriptorPool, PoolInfo>              pools;
    std::mutex                                                  poolSetsLock;   // PoolInfo::sets under a shared lock
    std::unordered_map<VkDescriptorUpdateTemplate, DescriptorUpdateTemplateInfo> templates;
    std::unordered_map<VkSwapchainKHR, SwapchainInfo>           swapchains;
    std::unordered_map<VkCommandBuffer, CmdState*>              cmdStates;
    std::unordered_map<VkCommandBuffer, VkCommandPool>          cmdPools;

    // Stereo parameter block (StereoParams.cpp)
    VkDescriptorSetLayout stereoSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout emptySetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      stereoPool = VK_NULL_HANDLE;
    VkDescriptorSet       stereoSetLeft = VK_NULL_HANDLE;   // sep = 0
    VkDescriptorSet       stereoSetRight = VK_NULL_HANDLE;  // sep = s
    VkDescriptorSet       stereoSetRightNoShift = VK_NULL_HANDLE;  // sep = 0, eye = e: right-eye draws the replay pins
    VkBuffer              stereoBuffer = VK_NULL_HANDLE;
    VkDeviceMemory        stereoMemory = VK_NULL_HANDLE;
    float*                stereoMapped = nullptr;
    VkDeviceSize          stereoStride = 256;
    float                 lastSep = 0.f, lastConv = 0.f, lastEye = 0.f;
    // widthSets: right-eye sets per viewport width, member 5 = half of it; the other sets hold 0 there, which leaves
    // those shaders' position alone.
    static constexpr uint32_t kWidthSets = 8;
    uint32_t              stereoWidths[kWidthSets] = {};
    uint32_t              stereoWidthCount = 0;
    VkDescriptorSet       stereoWidthRight[kWidthSets] = {};
    VkDescriptorSet       stereoWidthNoShift[kWidthSets] = {};
    std::mutex            stereoWidthLock;
    // flatNoDepthDraws, lock held: sprite-shaped depth-off pipelines since the last depth-tested draw (NoteSpriteOrder)
    std::vector<VkPipeline> spritesPending;
    // portalScissor, lock held: small scissors whose first draw cut a depth hole, this frame or the last.
    struct RsxPortal { VkRect2D rc = {}; unsigned frame = ~0u; };
    RsxPortal rsxPortals[4];
    uint32_t rsxPortalNext = 0;
    // readColorBuffers, lock held, per reloaded shape: whose read-back a reload holds, measured on the left
    // (RcbFrameEnd), and the right image at read-backs when that is another target's.
    struct RcbStash
    {
        static constexpr int kSlots = 3;
        static constexpr uint32_t kProbes = 32;
        uint32_t w = 0, h = 0;
        VkFormat fmt = VK_FORMAT_UNDEFINED;
        uint64_t reads = 0;                 // read-backs of the CPU's input, numbered from 1
        uint64_t reloads = 0;
        VkImage readFrom[kSlots] = {};      // by number % kSlots
        VkImage img[kSlots] = {};
        VkDeviceMemory mem[kSlots] = {};
        uint64_t slotRead[kSlots] = {};
        bool wantSlots = false;
        int source = -1;                    // 0: reload holds its own read-back, 1: another's; -1 (as 0): unmeasured
        int votedSource = -1, votes = 0;
        unsigned lastReload = ~0u;
        bool cpuIsFilter = false;           // the CPU's frame is its source but for edges: both eyes keep their own
        unsigned filterPasses = 0;
        bool averageLogged = false;
        struct Probe { unsigned frame = ~0u; uint64_t read = 0, reloadNo = 0; VkImage image = VK_NULL_HANDLE; bool reload = false, checked = false; };
        Probe probes[kProbes];
        uint32_t probeNext = 0;
        VkBuffer probeBuf = VK_NULL_HANDLE;
        VkDeviceMemory probeMem = VK_NULL_HANDLE;
        const uint8_t* probeMap = nullptr;
        bool probeFailed = false;
        VkImage mid = VK_NULL_HANDLE, low = VK_NULL_HANDLE;   // RcbMatchScale's road to the left's 1x
        VkDeviceMemory midMem = VK_NULL_HANDLE, lowMem = VK_NULL_HANDLE;
        uint32_t lowW = 0, lowH = 0;
    };
    std::vector<RcbStash> rcbStashes;
    // Binding 1: uint[8] the camera-window fragment shaders count their right-eye tests into (CameraWindows.cpp).
    VkBuffer              camCounterBuffer = VK_NULL_HANDLE;
    VkDeviceMemory        camCounterMemory = VK_NULL_HANDLE;
    uint32_t*             camCounters = nullptr;
    bool                  fragmentAtomics = false;   // the game enabled fragmentStoresAndAtomics

    // Shadow descriptor pools (Descriptors.cpp); shadowLock serialises pool
    // allocation so shadows are created without holding `lock`.
    std::vector<VkDescriptorPool> shadowPools;
    std::mutex shadowLock;

    // Constant mirrors carried by the GPU (Constants.cpp): a CPU copy of ReBAR uniform memory crosses PCIe uncached.
    struct XferRing
    {
        VkCommandPool                pool = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> cbs;
        std::vector<VkFence>         fences;
        size_t                       next = 0;
    };
    std::unordered_map<uint32_t, XferRing> xferRings;   // by queue family
    std::mutex xferLock;

    // Constants path (Constants.cpp). cameraActive: a camera block is being
    // rewritten for the right eye, so the shader shift is switched off.
    struct Constants* constants = nullptr;
    std::atomic<bool> cameraActive{false};
    // (layout, binding) keys whose range holds a camera block: under `lock`, for the descriptor path, which must never
    // take the constants mutex (a submit holds it across thousands of copies).
    std::unordered_set<uint64_t> cameraKeys;
    // Camera projection _11 from the constants path: the shader shift matches the inverses' clip scale with it.
    std::atomic<float> cameraP11{0.f};
    // Keys the constants path has given up on: the descriptor path stops recording their ranges.
    std::unordered_set<uint64_t> deadKeys;
    // unpatchedKeys: keys holding a full-screen draw's position matrix that skips its claim, and keys any draw claimed;
    // a skip-only position key reads the game's buffer in the right eye. Guarded by `lock`.
    std::unordered_set<uint64_t> skipPosKeys, claimedKeys;
    std::unordered_set<uint64_t> repointKeys;   // claimed position keys whose cached bindings are re-pointed
    std::unordered_set<uint64_t> claimMirrorKeys;   // claimMirror: claimed keys whose right sets read the claim sibling
    // binding -> byte offset -> vertex shaders whose Position chain loads a matrix there
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, unsigned>> positionUse;
    // binding -> matrices the register scan cannot see (stride != 16)
    std::unordered_map<uint32_t, std::vector<BlockMatrix>> packedMatrices;
    // Same bindings as a 256-bit set, so the constants path tests each storage range without a hash lookup.
    uint64_t packedBindingMask[4] = {};

    // Composite (Composite.cpp)
    std::unordered_map<uint32_t, VkCommandPool> presentPools;   // by queue family
    std::unordered_map<VkQueue, uint32_t>       queueFamilies;
    std::vector<VkQueueFamilyProperties>        familyProps;    // a game may present from a family with no graphics
    bool          externalMemoryWin32 = false;     // VK_KHR_external_memory_win32 enabled
    bool          externalSemaphoreWin32 = false;  // VK_KHR_external_semaphore_win32 enabled (VKWeaveGpuSync)
    bool          dedicatedImageAliasing = false;  // VK_NV_dedicated_allocation_image_aliasing enabled
    WeaveDevice*  weave = nullptr;

    // Per-frame diagnostics
    unsigned statStereoPasses = 0, statMonoPasses = 0, statShadowSets = 0, statSecondaryPasses = 0,
             statEyeBinds = 0, statUnpatchedDraws = 0;
    std::atomic<unsigned> statReplayDraws{0}, statReplayDispatch{0};   // added by every replaying thread
    unsigned statShadersPatched = 0, statShadersSkipped = 0, statImagesDoubled = 0, statVSReverted = 0, statAliases = 0, statMirrors = 0;
    unsigned statConstantsShaders = 0;    // served by patching their matrix per draw instead of self-shifting
    unsigned statFsEyeOrigin = 0, statFsProducts = 0;   // fragment modules / reconstructions given eye*right
    unsigned statFsCamera = 0, statFsCameraWindows = 0; // fragment modules / matrix windows given the camera test
    unsigned statPosVaryingShaders = 0;                 // position stages whose clip varyings shift with gl_Position
    unsigned statPrevClipShaders = 0;                   // vertex stages whose previous-frame clip varying shifts too
    unsigned statRemapShaders = 0, statRemapRight = 0;  // VKComputeRemap: modules with a twin / right runs on set 8
    struct TileState* tiles = nullptr;                  // KytyTiles.cpp: the right eye's tile lists
    unsigned statSpritePipes = 0;                       // pipelines given a sprite VS's parallax twin
    std::atomic<bool> shadGtHudPhase{false};   // flatUiPasses: from the first display-format pass to the final copy
    // Attachment layouts per render pass (lock held).
    std::unordered_map<VkRenderPass, std::vector<VkAttachmentDescription>> renderPassAtts;
    unsigned statScreenReadFs = 0, statScreenReadPipes = 0;   // FS reading the screen at a clip varying / their pipes
    // Dynamic rendering (Vulkan 1.3 core): passes begun, replayed, kept mono, mono because only
    // some attachments are doubled, mono because suspended/resumed across command buffers.
    unsigned statDynRendering = 0, statDynStereo = 0, statDynMono = 0, statDynMixed = 0, statDynSuspended = 0;
    // Buffer twins (StereoBuffers.cpp, VKBufferTwins). Texel views of twinned buffers get twin views.
    struct BufferViewInfo { VkBufferViewCreateInfo ci; VkBufferView twin; };
    std::unordered_map<VkBufferView, BufferViewInfo> bufferViews;   // guarded by `lock`
    // The right run's writes to a small buffer with no twin (GDS, CPU-visible) go to a seeded scratch slot.
    VkBuffer scratchBuffer = VK_NULL_HANDLE;
    VkDeviceMemory scratchMemory = VK_NULL_HANDLE;
    VkDeviceSize scratchHead = 0;                              // guarded by `lock`
    unsigned statTwins = 0, statTwinFailures = 0, statSparseBindsMirrored = 0;   // live / failed / per window
    unsigned long long twinBytes = 0;
    unsigned statRightDispatches = 0, statScratchSlots = 0, statUntwinnedWrites = 0;   // per window
    unsigned statIndirectFromLeft = 0;                         // per window: right indirect draws, left's buffers
    // Twinned storage ranges a right run wrote, by frame (lock): an indirect dispatch's args there are the right's own.
    struct RightWrite { VkBuffer buffer; VkDeviceSize offset, size; uint64_t frame; };
    std::vector<RightWrite> rightWrites;
    unsigned statTwinArgs = 0;                                 // per window: right indirect dispatches on twin args
    // VKIndirectFromLeft=2: VS seen drawing the left's particles indirectly; their direct draws do too (lock).
    std::unordered_set<uint32_t> fromLeftVs;
    // Lock held: from-left draws' set layouts, whose shadows keep the left's buffers.
    std::unordered_set<VkDescriptorSetLayout> leftBufferSetLayouts;
    unsigned statDirectFromLeft = 0;                           // per window
    unsigned statGameDraws = 0, statGameDrawsInStereoPass = 0;
    // Per window: mesh-task draws (passed through, never replayed), pipelines with a mesh stage, and pipelines
    // whose SPIR-V never reached CreateShaderModule (maintenance5 inline code, pipeline binaries).
    unsigned statMeshDraws = 0, statMeshPipelines = 0, statUnseenPipelines = 0;
    // The game's conditional rendering (occlusion verdicts): begins, and those inside a recorded pass.
    unsigned statCondBegins = 0, statCondInPass = 0;
    // VKOcclusionBothEyes: each occlusion pool's twin the right eye's replay counts into (lock held);
    // `begun` marks the slots it began since their reset.
    struct OccTwin { VkQueryPool twin = VK_NULL_HANDLE; uint32_t count = 0; std::unique_ptr<std::atomic<uint8_t>[]> begun; };
    std::unordered_map<VkQueryPool, OccTwin> occTwins;
    // Per window: occlusion segments recorded in stereo passes, begun in the right replay, slots read back, read slots
    // the left counted 0 and the right did not, and GPU copies of an occlusion pool (not summed).
    std::atomic<unsigned> statOccRecorded{ 0 }, statOccRight{ 0 }, statOccReads{ 0 }, statOccRightOnly{ 0 }, statOccGpuCopies{ 0 };
    // Per window: time the read hook adds on the game's thread (twin read and sum), and twin slots waited for.
    std::atomic<long long> occReadTicks{ 0 };
    std::atomic<unsigned> statOccWaits{ 0 };
    // Per window: copies out of a doubled image into a buffer and buffer copies into one (with buffer twins off both
    // eyes then get the buffer's left-eye bytes), and right-eye bindings that kept a doubled image's left view.
    unsigned statDoubledToBuffer = 0, statBufferToDoubled = 0;
    std::atomic<unsigned> statLeftInRight{0};
    std::unordered_map<VkImage, unsigned> bufferToDoubledImages;   // lock held
    // Allocation census: what the game asks the driver for versus what we add.
    unsigned statGameAllocs = 0, statDedicatedAllocs = 0, statSibAllocs = 0;
    unsigned long long statGameAllocBytes = 0, statSibAllocBytes = 0;
    unsigned statScopedRanges = 0, statScopedSkipped = 0;   // ranges patched for a consuming draw / never claimed
    unsigned statUnshiftedDraws = 0;          // right-eye draws through a pipeline with no patched stage
    unsigned statSmallUnshift = 0;            // replay draws forced to sep=0 by the <=6-vertex rule
    unsigned statOffAspectDraws = 0;          // replay draws on the sep=0 set: the pass target is off-aspect
    // screenTriangles: right-eye draws given the game's own camera set, and those left on the shadow (a doubled image
    // in the set, or its bind not known); added under the shared lock.
    std::atomic<unsigned> statScreenTriLeft{0}, statScreenTriRefused{0};
    // Replay draws of matrix-positioned pipelines by who carried the eye: the patched constants (sep=0 bound),
    // or the self-shift where the last submit did not patch the range.
    unsigned statServedDraws = 0, statFallbackDraws = 0;
    std::atomic<unsigned> statReplayShadowed{0}, statReplayUnshadowed{0}, statReplayUnknownSet{0};   // replay set binds
    // Constant data paths the layer can see (per report window)
    unsigned statInlineBlocks = 0, statInlineBytes = 0, statPushConst = 0, statPushConstBytes = 0, statUpdateBuf = 0, statUpdateBufBytes = 0, statPushSets = 0;
    // Bumped under the SHARED lock, so they must be atomic.
    std::atomic<unsigned> statDescWrites{0}, statDescCalls{0};
    unsigned statTemplateCalls = 0;
    PerfCounters perf;
};

// Dispatchable handles all start with the loader's dispatch pointer, the key layers find their per-object data by.
inline void* DispatchKey(const void* dispatchable) { return *(void* const*)dispatchable; }

InstanceData* FindInstance(const void* dispatchable);
DeviceData*   FindDevice(const void* dispatchable);
extern std::atomic<uint64_t> g_deviceGen;          // bumps when a device is destroyed

PFN_vkVoidFunction LayerGetInstanceProcAddr(VkInstance instance, const char* pName);
PFN_vkVoidFunction LayerGetDeviceProcAddr(VkDevice device, const char* pName);

struct HookEntry { const char* name; PFN_vkVoidFunction fn; };
#define WIZ3D_HOOK(name) { "vk" #name, (PFN_vkVoidFunction)&Hook_##name }

// Hook tables contributed by each module (terminated by a null name).
extern const HookEntry g_resourceHooks[];
extern const HookEntry g_descriptorHooks[];
extern const HookEntry g_commandHooks[];
extern const HookEntry g_pipelineHooks[];
extern const HookEntry g_swapchainHooks[];

// Module entry points
bool StereoParamsInit(DeviceData* d);
void StereoParamsDestroy(DeviceData* d);
void StereoParamsUpdate(DeviceData* d);              // refresh sep/conv from gInfo
VkDescriptorSet StereoSetForWidth(DeviceData* d, VkDescriptorSet set, uint32_t width);
uint32_t FindMemoryType(DeviceData* d, uint32_t typeBits, VkMemoryPropertyFlags want, uint32_t preferred);

// The vertex patch (SpirvPatch.cpp): gl_Position shifts for the right eye; with no rule set, its clip varyings
// follow the switches.
struct SpirvRules
{
    bool spriteShift = false;      // a w == 1 sprite moves by its depth's parallax instead (EmitSpriteShift)
    int  spriteVarying = -1;       // ... and this output varying with it; -1 none
    bool forceVaryings = false;    // clip varyings shift as for a screen-reading FS (the VS twin)
    bool screenUv = false;         // screen-uv varyings, k * (x / w) + c, take x after the shift
    bool depthSprites = false;     // w == 1 sprites tested against depth move by their own depth's parallax
};
struct SpirvCounts
{
    unsigned posVaryings = 0;      // outputs storing gl_Position's own values
    unsigned prevClip = 0;         // previous-frame clip varyings
    unsigned ndcVaryings = 0;      // xy / w and projected varyings
    unsigned screenUvMoved = 0;    // screen-uv products
    unsigned depthSpriteInputs = 0;   // inputs a depth sprite reads moved (its position first)
};
bool PatchSpirv(const uint32_t* code, size_t words, const SpirvRules& rules, std::vector<uint32_t>& out, const char** reason,
                spv::ExecutionModel* model, SpirvCounts* counts = nullptr);
// Uniform-block members a vertex shader multiplies into gl_Position (SpirvPatch.cpp).
struct PositionUse { uint32_t set, binding, offset; bool matrix; int depth; };
bool AnalyzePositionChain(const uint32_t* code, size_t words, std::vector<PositionUse>& out);
void AnalyzeStorageWrites(const uint32_t* code, size_t words, std::vector<uint64_t>& out);

void AnalyzeBlockMatrices(const uint32_t* code, size_t words, std::vector<BlockMatrix>& out);
// The fragment / compute stereo patchers of FsPatch.cpp and CameraWindows.cpp are declared in FsPatch.h.

// VKLightRaysCamera: a light-volume FS's camera and light-space rays (each times the linear depth it samples at its
// screen-position varying) moved to the right eye's surface, using the camera block of its depth constants.
bool PatchLightRaysFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* rays, unsigned* uses);
// rsxPositionTwins (RsxPosition.cpp): an FS's P = col*wpos.x + ..., P/P.w moved to the left's pixel for its surface.
bool PatchRsxPositionFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* divides);
bool PatchVelocityFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, bool skipUnitW, float unitWEps);
bool PatchGlassFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* sites);
bool PatchDecalFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* rows, bool* fromVarying);

unsigned CountRsxVertexInputs(const uint32_t* code, size_t words);
void SpriteFrameEnd(DeviceData* d);
void RcbFrameEnd(DeviceData* d);
// FMul+FAdd pairs back to Fma.
bool FuseUnfusedMad(const uint32_t* code, size_t words, std::vector<uint32_t>& out, unsigned* fused);
uint64_t DynDeclaredMask(const VkPipelineDynamicStateCreateInfo* ci);
bool PatchShaftFS(const uint32_t* code, size_t words, bool blur, uint32_t stereoSet, std::vector<uint32_t>& out);
bool ReadsScreenAtClipVarying(const uint32_t* code, size_t words);

void DrainRetiredImages(DeviceData* d);   // frees siblings the GPU has finished with
void DrainPendingBufferSiblings(DeviceData* d);   // creates the siblings the compute gate asked for
void ShadowPoolsDestroy(DeviceData* d);
VkDescriptorSet ShadowSetFor(DeviceData* d, VkDescriptorSet set);   // lock held
// Lock NOT held; returns the stale sets found (forced: those re-pointed anyway).
unsigned RepointLearnedSets(DeviceData* d, bool apply, unsigned* forced);
bool BufferSibling(DeviceData* d, VkBuffer buf, VkBuffer* sibling);  // lock held; true when buf has one
bool BufferClaimSibling(DeviceData* d, VkBuffer buf, VkBuffer* sibling);   // lock held; claimMirror's second sibling
// This (layout, binding) holds a camera block.
bool ConstantsRangeIsStereo(DeviceData* d, VkDescriptorSetLayout layout, uint32_t binding);
// The copy of a uniform block the right eye's set binds (Constants.cpp has the decision table); lock held.
enum class RightSource { Game, Sibling, ClaimSibling };
RightSource ConstantsRightSource(DeviceData* d, VkDescriptorSetLayout layout, uint32_t binding, VkBuffer buffer, VkBuffer* copy);
// Identity of a uniform range across every set sharing a layout.
inline uint64_t LearnKey(VkDescriptorSetLayout layout, uint32_t binding)
{
    return (uint64_t)layout * 0x9E3779B97F4A7C15ull ^ (uint64_t)binding;
}

// Constants path: a set seen while recording; consumerMask: the registers a matrix-positioned draw reads, so the submit
// patches only those (DX11's per-draw scoping). noClaimPipe: a draw that must stay unpatched (<=6 verts, EQUAL).
struct UsedSet { VkDescriptorSet set; uint32_t dynCount; uint32_t dyn[8]; uint64_t consumerMask; uint32_t fs = 0; VkPipeline pipeline = VK_NULL_HANDLE; VkPipeline noClaimPipe = VK_NULL_HANDLE; uint32_t noClaimW = 0, noClaimH = 0;
                 uint32_t noClaimBinding = UINT32_MAX;      // the skip-claim draw's position binding
                 uint32_t repointBinding = UINT32_MAX;      // the claimed position binding to re-point
                 uint32_t posBinding = UINT32_MAX; };       // claimedKeyRetry: the claimed position binding
void ConstantsInit(DeviceData* d);
void ConstantsDestroy(DeviceData* d);
// Before the real submit.
void ConstantsSubmit(DeviceData* d, VkQueue queue, uint32_t cbCount, const VkCommandBuffer* cbs);
void ConstantsLogStats(DeviceData* d);
void ConstantsMapMemory(DeviceData* d, VkDeviceMemory mem, VkDeviceSize off, VkDeviceSize size, void* ptr); // lock held
void ConstantsUnmapMemory(DeviceData* d, VkDeviceMemory mem);                                               // lock held
extern const HookEntry g_constantsHooks[];

void InstallNgxWatcherVK();                                          // NgxVK.cpp
bool ImageIsStereo(DeviceData* d, VkImage img, VkImage* sibling);    // lock held
bool OffAspect(uint32_t w, uint32_t h, const VkExtent2D& bb);        // aspect off the backbuffer's by over 0.03
VkDeviceMemory EnsureMirror(DeviceData* d, MemoryInfo& mi);          // lock held; same size and type as the game's

// Buffer twins (StereoBuffers.cpp). All return / take the lock as noted.
VkBuffer     BufferTwin(DeviceData* d, VkBuffer b);                  // lock held; null when none
VkBufferView BufferViewTwin(DeviceData* d, VkBufferView v);          // lock held; v itself when its buffer has no twin
void TwinOnCreateBuffer(DeviceData* d, VkBuffer buf);                // lock NOT held; sparse buffers
void TwinOnBindBuffer(DeviceData* d, VkBuffer buf);                  // lock NOT held; GPU-only buffers
void TwinOnDestroyBuffer(DeviceData* d, VkBuffer twin);              // lock NOT held
VkBuffer ScratchSlot(DeviceData* d, VkDeviceSize n, VkDeviceSize* off);   // lock NOT held
// KytyTiles.cpp, lock NOT held. TileRightBlock: pc's block pointer becomes a copy naming the right eye's lists;
// false: nothing to change.
void TilesLoadProfile();   // the <VKKyty> family's tile chain shaders and their list offsets
void TilesInit(DeviceData* d);
void TilesDestroy(DeviceData* d);
void TileNoteBuffer(DeviceData* d, VkBuffer buffer);
void TileForgetBuffer(DeviceData* d, VkBuffer buffer, VkDeviceSize size);
bool TileChainShader(DeviceData* d, uint32_t crc);
bool TileRightBlock(DeviceData* d, VkPipeline pipeline, uint32_t crc, uint32_t* pc, uint32_t dwords);
bool TileRightArgs(DeviceData* d, VkBuffer buf, VkDeviceSize off, VkBuffer* out, VkDeviceSize* outOff);
// FrameGenVK.cpp: DLSS-G over two squeezed views. t_foreignCalls: inside a Streamline call recording into the
// game's command buffer.
extern thread_local int t_foreignCalls;
void FgInit(DeviceData* d);
void FgDestroy(DeviceData* d);
bool FgOn();
std::unique_lock<std::mutex> FgQueueLock();   // held around each queue call while DLSS-G runs
void FgSwapchainMade();
unsigned FgPresentsPerFrame();
void FgRegion(uint32_t w, uint32_t h, VkRect2D* r);
bool FgInputImage(const VkImageCreateInfo& ci);
bool FgForeignCaller();
void TwinsDestroy(DeviceData* d);
extern const HookEntry g_twinHooks[];
bool ViewSibling(DeviceData* d, VkImageView view, VkImageView* sibling); // lock held
bool ViewOfDoubledImage(DeviceData* d, VkImageView view);              // lock held
bool RegisterSwapchainImages(DeviceData* d, VkSwapchainKHR sc);

// Command buffers the layer allocates itself take the loader's dispatch data here: a layer below this one finds its
// device through it, and without it fails on the first command.
inline void StampCommandBuffers(DeviceData* d, VkCommandBuffer* cbs, uint32_t count)
{
    for (uint32_t i = 0; d->setLoaderData && i < count; ++i) d->setLoaderData(d->device, cbs[i]);
}

void CompositePresent(DeviceData* d, VkQueue queue, VkPresentInfoKHR* info, std::vector<VkSemaphore>& outWait);
void CompositeDestroy(DeviceData* d, SwapchainInfo& sc);

// The DX9/DX11 on-screen display over the presented image (OSDRendererVK.cpp). Lock held for all of these.
void OsdTick(DeviceData* d);                // once per present
bool OsdWanted(DeviceData* d);              // an element is showing
// dst is w x h and in TRANSFER_DST_OPTIMAL before and after.
void OsdDraw(DeviceData* d, VkCommandBuffer cb, VkImage dst, VkFormat format, uint32_t w, uint32_t h);
void OsdDestroySwapchain(DeviceData* d, SwapchainInfo& sc);
void OsdDestroyDevice(DeviceData* d);

// The DX9/DX11 stereo screenshot, Ctrl+F12 (ScreenshotVK.cpp). Lock held for all of these.
void ScreenshotTick();                      // once per present: takes the hotkey's request
bool ScreenshotArmed();                     // a request waits for a present to copy its eyes
// Both images are w x h and in TRANSFER_SRC_OPTIMAL; right null = one eye; squeezed = left holds both eyes already.
void ScreenshotCopy(DeviceData* d, VkCommandBuffer cb, VkImage left, VkImage right, VkFormat format, uint32_t w, uint32_t h, bool squeezed);
void ScreenshotWrite(DeviceData* d, VkQueue queue);   // after the submit holding the copies
void ScreenshotDestroy(DeviceData* d);

// SR weave through a private D3D11 device and shared textures (Weave.cpp).
// Lock held for all of these.
bool WeaveSelected();
bool WeaveFrame(DeviceData* d, SwapchainInfo& sc, VkQueue queue, uint32_t idx, VkImage img, VkImage sib,
                const VkPresentInfoKHR* info, bool firstSwapchain, bool graphics, VkSemaphore* outDone);
void WeaveFreeCommands(DeviceData* d, SwapchainInfo& sc);   // before the swapchain's command pool changes
void WeaveDestroySwapchain(DeviceData* d, SwapchainInfo& sc);
void WeaveDestroyDevice(DeviceData* d);

void CmdStateFree(DeviceData* d, VkCommandBuffer cb);                // lock held
void Log(const char* fmt, ...);
void Breadcrumb(const char* fmt, ...);   // last layer action, printed by the fault handler
void LogFrameStats(DeviceData* d);

}} // namespace wiz3d::vk
