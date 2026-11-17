/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Recording and the right-eye replay: a pass into a sibling-backed framebuffer (or dynamic rendering) is recorded and
// re-issued for the right eye at its end; dispatches are doubled in place; copies, clears and barriers are mirrored.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <array>
#include <set>
#include <string>

namespace wiz3d { namespace vk {

enum class Op : uint8_t
{
    BindPipeline, BindSets, BindIndex, BindVertex, BindVertex2, PushConstants,
    Viewport, Scissor, LineWidth, DepthBias, BlendConstants, DepthBounds,
    StencilCompare, StencilWrite, StencilRef,
    Draw, DrawIndexed, DrawIndirect, DrawIndexedIndirect,
    DrawIndirectCount, DrawIndexedIndirectCount,
    ClearAttachments, NextSubpass, Execute, Barrier, PushDescriptors,
    DynState, BindIndex2, Barrier2, CondBegin, CondEnd,
    // VKMeshDraws, after DrawIndexedIndirectCount so its range checks keep their meaning
    DrawMeshTasks, DrawMeshTasksIndirect, DrawMeshTasksIndirectCount,
    QueryBegin, QueryEnd,                                               // VKOcclusionBothEyes
};

// vkCmdSet* calls beyond the core 1.0 set (VKDynamicRendering), restored at a replay's start.
enum class Dyn : uint8_t
{
    CullMode, FrontFace, PrimitiveTopology, ViewportWithCount, ScissorWithCount,
    DepthTestEnable, DepthWriteEnable, DepthCompareOp, DepthBoundsTestEnable, StencilTestEnable, StencilOp,
    RasterizerDiscardEnable, DepthBiasEnable, PrimitiveRestartEnable, PatchControlPoints, LogicOp,
    TessellationDomainOrigin, DepthClampEnable, PolygonMode, RasterizationSamples, SampleMask,
    AlphaToCoverageEnable, AlphaToOneEnable, LogicOpEnable, ColorBlendEnable, ColorBlendEquation,
    ColorWriteMask, ColorBlendAdvanced, RasterizationStream, ConservativeRasterizationMode,
    ExtraPrimitiveOverestimationSize, DepthClipEnable, SampleLocationsEnable, ProvokingVertexMode,
    LineRasterizationMode, LineStippleEnable, DepthClipNegativeOneToOne, ColorWriteEnable, VertexInput,
    AttachmentFeedbackLoopEnable,
};

// The latest value of one dynamic state; `key` separates per-face and per-attachment ranges, `seq` restores them
// in the order they were last set. Inline storage: GfxState is copied at every pass boundary.
struct DynState
{
    Dyn      fn;
    uint32_t key = 0, seq = 0, len = 0;
    uint8_t  inl[64];
    std::vector<uint8_t> big;                   // vertex input, blend equations
    const uint8_t* data() const { return len <= sizeof(inl) ? inl : big.data(); }
    void assign(const void* p, size_t n)
    {
        len = (uint32_t)n;
        if (n <= sizeof(inl)) { if (n) memcpy(inl, p, n); }
        else big.assign((const uint8_t*)p, (const uint8_t*)p + n);
    }
};

// VKDynByPipeline: which tracked states a pipeline declares dynamic, a bit per Dyn.
uint64_t DynDeclaredMask(const VkPipelineDynamicStateCreateInfo* ci)
{
    static const struct { Dyn fn; VkDynamicState vk; } k[] = {
        { Dyn::CullMode, VK_DYNAMIC_STATE_CULL_MODE }, { Dyn::FrontFace, VK_DYNAMIC_STATE_FRONT_FACE },
        { Dyn::PrimitiveTopology, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY }, { Dyn::ViewportWithCount, VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT },
        { Dyn::ScissorWithCount, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT }, { Dyn::DepthTestEnable, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE },
        { Dyn::DepthWriteEnable, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE }, { Dyn::DepthCompareOp, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP },
        { Dyn::DepthBoundsTestEnable, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE }, { Dyn::StencilTestEnable, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE },
        { Dyn::StencilOp, VK_DYNAMIC_STATE_STENCIL_OP }, { Dyn::RasterizerDiscardEnable, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE },
        { Dyn::DepthBiasEnable, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE }, { Dyn::PrimitiveRestartEnable, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE },
        { Dyn::PatchControlPoints, VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT }, { Dyn::LogicOp, VK_DYNAMIC_STATE_LOGIC_OP_EXT },
        { Dyn::TessellationDomainOrigin, VK_DYNAMIC_STATE_TESSELLATION_DOMAIN_ORIGIN_EXT }, { Dyn::DepthClampEnable, VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT },
        { Dyn::PolygonMode, VK_DYNAMIC_STATE_POLYGON_MODE_EXT }, { Dyn::RasterizationSamples, VK_DYNAMIC_STATE_RASTERIZATION_SAMPLES_EXT },
        { Dyn::SampleMask, VK_DYNAMIC_STATE_SAMPLE_MASK_EXT }, { Dyn::AlphaToCoverageEnable, VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT },
        { Dyn::AlphaToOneEnable, VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT }, { Dyn::LogicOpEnable, VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT },
        { Dyn::ColorBlendEnable, VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT }, { Dyn::ColorBlendEquation, VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT },
        { Dyn::ColorWriteMask, VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT }, { Dyn::ColorBlendAdvanced, VK_DYNAMIC_STATE_COLOR_BLEND_ADVANCED_EXT },
        { Dyn::RasterizationStream, VK_DYNAMIC_STATE_RASTERIZATION_STREAM_EXT }, { Dyn::ConservativeRasterizationMode, VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT },
        { Dyn::ExtraPrimitiveOverestimationSize, VK_DYNAMIC_STATE_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_EXT }, { Dyn::DepthClipEnable, VK_DYNAMIC_STATE_DEPTH_CLIP_ENABLE_EXT },
        { Dyn::SampleLocationsEnable, VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_ENABLE_EXT }, { Dyn::ProvokingVertexMode, VK_DYNAMIC_STATE_PROVOKING_VERTEX_MODE_EXT },
        { Dyn::LineRasterizationMode, VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT }, { Dyn::LineStippleEnable, VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT },
        { Dyn::DepthClipNegativeOneToOne, VK_DYNAMIC_STATE_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT }, { Dyn::ColorWriteEnable, VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT },
        { Dyn::VertexInput, VK_DYNAMIC_STATE_VERTEX_INPUT_EXT }, { Dyn::AttachmentFeedbackLoopEnable, VK_DYNAMIC_STATE_ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT },
    };
    uint64_t mask = 0;
    for (uint32_t i = 0; ci && i < ci->dynamicStateCount; ++i)
        for (const auto& e : k) if (e.vk == ci->pDynamicStates[i]) mask |= 1ull << (unsigned)e.fn;
    return mask;
}

struct Rec
{
    Op       op;
    uint32_t u[6] = {};
    // Four: the indirect-count draws name both an argument and a count buffer.
    uint64_t h[4] = {};
    float    f[4] = {};
    uint32_t dataOff = 0, dataLen = 0;   // payload in CmdState::blob
};

struct SetBind
{
    VkDescriptorSet  set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<uint32_t> dyn;
};

struct PushRange
{
    VkPipelineLayout   layout;
    VkShaderStageFlags stages;
    uint32_t           offset, size;
    uint8_t            data[256];
    uint32_t           seq = 0;   // VKPushConstantOrder: when this range was last written
};

// Everything needed to put the command buffer back into its state at a pass's start, for the replay.
struct GfxState
{
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    SetBind          sets[32];
    VkBuffer         indexBuf = VK_NULL_HANDLE;
    VkDeviceSize     indexOff = 0;
    VkIndexType      indexType = VK_INDEX_TYPE_UINT16;
    bool             hasIndex = false;
    VkDeviceSize     indexSize = VK_WHOLE_SIZE;     // maintenance5 bind: restored through the same call
    bool             index2 = false;
    VkBuffer         vb[32] = {};
    VkDeviceSize     vbOff[32] = {};
    uint32_t         vbMask = 0;
    // A binding taken through vkCmdBindVertexBuffers2 (size, and the stride a dynamic-stride pipeline needs) is
    // restored through it.
    VkDeviceSize     vbSize[32] = {};
    VkDeviceSize     vbStride[32] = {};
    uint32_t         vb2Mask = 0;
    // A null pStrides keeps the pipeline's static stride (unlike a stride of 0), so it is restored null too;
    // a null pSizes equals VK_WHOLE_SIZE and needs no mask.
    uint32_t         vbStrideMask = 0;
    std::vector<VkViewport> viewports;
    uint32_t         viewportFirst = 0;
    std::vector<VkRect2D>   scissors;
    uint32_t         scissorFirst = 0;
    uint32_t         viewportSeq = 0, scissorSeq = 0;   // vs DynState::seq: the WithCount variants share this state
    bool  hasLineWidth = false, hasDepthBias = false, hasBlend = false, hasDepthBounds = false;
    float lineWidth = 1.f, depthBias[3] = {}, blend[4] = {}, depthBounds[2] = {};
    struct { bool has = false; uint32_t front = 0, back = 0; } stencilCompare, stencilWrite, stencilRef;
    std::vector<PushRange> pushes;
    uint32_t             pushSeq = 0;           // write counter behind PushRange::seq
    std::vector<uint8_t> pushSets[32];          // serialised push descriptors per set
    VkPipelineLayout     pushLayout[32] = {};
    std::vector<DynState> dyn;
    uint32_t             dynSeq = 0;
};

// Push descriptors touch no descriptor set: kept serialised per (bind point, set) and re-pushed for the right eye
// with sibling views (IssuePushes).
struct PushRecord
{
    uint32_t binding, arrayElement, count, type, dataOff, dataLen;
};

struct CmdState
{
    VkCommandBuffer cb = VK_NULL_HANDLE;
    bool secondary = false;
    bool secondaryLevel = false;   // allocated secondary; `secondary`: it continues a pass (Hook_BeginCommandBuffer)
    GfxState gfx;
    SetBind  compSets[32];
    std::vector<uint8_t> compPushes[32];        // serialised pushes, compute bind point
    VkPipelineLayout     compPushLayout[32] = {};
    VkPipeline       compPipeline = VK_NULL_HANDLE;
    VkPipelineLayout compLayout = VK_NULL_HANDLE;
    bool             compRemap = false;          // bound compute pipeline reads set 8 (VKComputeRemap)
    uint32_t         compCrc = 0;                // bound compute pipeline's module
    bool             foreign = false;            // allocated by Streamline or NGX: their work runs once
    bool             portalWide = false;   // portalScissor, replay: the right eye's scissor is RsxPortalScissor's
    // The last doubled image copied into a buffer: if the next copy out of that buffer lands in a doubled image,
    // the right eye repeats the trip.
    VkBuffer         tripBuffer = VK_NULL_HANDLE;
    VkImage          tripSrcR = VK_NULL_HANDLE;
    VkImageLayout    tripLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::vector<VkBufferImageCopy> tripRegions;
    // tripKernels: the byte-swap kernels run on that buffer between the copies, one bound pipeline and its dispatches.
    VkPipeline       tripPipeline = VK_NULL_HANDLE;
    std::vector<std::array<uint32_t, 3>> tripDispatches;
    // KytyTiles.cpp: the compute push constants in effect, which hold the block pointer a right run replaces.
    uint32_t         compPc[32] = {};
    uint32_t         compPcSize = 0;
    VkPipelineLayout compPcLayout = VK_NULL_HANDLE;
    VkShaderStageFlags compPcStages = 0;
    // Stereo set tracking: which layout bound it last, or null when dirty.
    VkPipelineLayout eyeLayout = VK_NULL_HANDLE;
    VkDescriptorSet  eyeSet = VK_NULL_HANDLE;
    // Current pass
    bool inPass = false, stereoPass = false, replaying = false;
    // VKOcclusionBothEyes: the twin query the right replay has open (one at a time per command buffer).
    VkQueryPool rQueryOpen = VK_NULL_HANDLE;
    uint32_t rQueryIdx = 0;
    // VKIndirectFromLeft, during a right replay: the graphics pushes and vertex/index binds in effect (game handles),
    // re-issued with the left's buffers around an indirect draw and with the right's after it.
    std::vector<uint8_t> rPush[8];
    VkDescriptorSet rGameSet[8] = {};                // the game's own sets a right replay has bound, and their layouts
    VkPipelineLayout rGameSetLayout[8] = {};
    // screenTriangles: how a right replay bound each slot (the game's set, layout, dynamic offsets), if one bind's own.
    SetBind rBind[8];
    bool rBindKnown[8] = {};
    VkPipelineLayout rPushLayout[8] = {};
    VkBuffer rIb = VK_NULL_HANDLE;
    VkDeviceSize rIbOff = 0, rIbSize = VK_WHOLE_SIZE;
    VkIndexType rIbType = VK_INDEX_TYPE_UINT16;
    bool rIbTwo = false;
    VkBuffer rVb[32] = {};
    VkDeviceSize rVbOff[32] = {}, rVbSize[32] = {}, rVbStride[32] = {};
    uint32_t rVbMask = 0, rVb2Mask = 0, rVbStrideMask = 0;
    VkRenderPassBeginInfo passBegin = {};
    std::vector<VkClearValue> clears;
    VkFramebuffer passSibling = VK_NULL_HANDLE;
    // Dynamic rendering: the right eye's begin info, every view swapped for its sibling.
    bool dynPass = false, dynHasDepth = false, dynHasStencil = false;
    bool shadGtHud = false;   // flatUiPasses: this pass is UI (BeginDynamicPass)
    bool shadGtMask = false;  // flatUiPasses: a colourless pass inside the UI: its quads are masks
    VkRenderingInfo dynInfo = {};
    std::vector<VkRenderingAttachmentInfo> dynColor;
    VkRenderingAttachmentInfo dynDepth = {}, dynStencil = {};
    // A mixed pass stays mono; what it wrote into its doubled attachments is copied into their siblings at its end.
    struct MixedCopy { VkImage image, sibling; VkImageSubresourceRange range; VkImageLayout layout; uint32_t w, h; VkImageLayout siblingFrom; };
    std::vector<MixedCopy> mixedCopies;
    GfxState passStart;
    // Recording (secondary buffers record always: a primary may replay them)
    std::vector<Rec> recs;
    std::vector<uint8_t> blob;
    // Sets bound since the last reset, for the submit-time constants scan
    std::vector<UsedSet> usedSets;
    // Shadow per set the replay will bind, resolved once per pass rather than under the device lock per bind.
    std::unordered_map<VkDescriptorSet, VkDescriptorSet> replayShadow;
    // Lock-free lookups for the hot hooks, valid while objGen is unchanged
    struct LayoutFacts { bool patched; uint32_t gameSets; std::vector<uint32_t> dyn; };
    std::unordered_map<VkPipelineLayout, LayoutFacts> layoutCache;
    struct PipelineFacts { VkPipelineLayout layout; bool shifted; bool posUsesMatrix; uint32_t posSet; uint64_t posMask; bool depthEqual; bool remapCS; uint32_t csCrc; bool depthOn; bool spriteShape; uint32_t posBinding = UINT32_MAX;
                           bool repointPos = false;      // its claims re-point cached right-eye camera bindings
                           bool worldQuad = false; };    // world quad: self-shifted, never pinned as small
    std::unordered_map<VkPipeline, PipelineFacts>     pipelineCache;
    bool gfxShifted = false;                          // bound graphics pipeline carries the patch
    bool lampPipe = false;
    float replayDb[2] = { 0.f, 1.f };
    bool gfxByConstants = false;                     // ... or is served by patching its matrix for this draw
    bool gfxEqualPass = false;                        // depth test EQUAL: must stay invariant with its prepass
    // flatNoDepthDraws, NoteSpriteOrder: the bound pipeline's depth test, and depth off with a sprite's inputs
    bool gfxDepthOn = false, gfxSpriteShape = false;
    uint32_t passW = 0, passH = 0;                    // current framebuffer
    bool passAspectOk = true;                         // offAspectPasses: the target has the backbuffer's aspect
    uint32_t gfxPosSet = 0, gfxPosBinding = UINT32_MAX;
    uint64_t gfxPosMask = 0;
    bool gfxRepointPos = false;                       // PipelineFacts::repointPos of the bound graphics pipeline
    bool gfxWorldQuad = false;                        // PipelineFacts::worldQuad of the bound graphics pipeline
    uint64_t objGen = 0;
};

// A command buffer is recorded by one thread at a time, so a per-thread
// (cb -> state) memo stands in for the locked map lookup on every command.
static CmdState* State(DeviceData* d, VkCommandBuffer cb)
{
    struct Cache { DeviceData* d; VkCommandBuffer cb; CmdState* s; uint64_t gen; };
    thread_local Cache c = {};
    const uint64_t gen = d->cmdGen.load(std::memory_order_acquire);
    if (c.d == d && c.cb == cb && c.gen == gen) return c.s;
    CmdState* s = nullptr;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto it = d->cmdStates.find(cb);
        if (it != d->cmdStates.end()) s = it->second;
    }
    if (!s)
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->cmdStates.find(cb);
        if (it != d->cmdStates.end()) s = it->second;
        else
        {
            s = new CmdState();
            s->cb = cb;
            d->cmdStates[cb] = s;
        }
    }
    c.d = d; c.cb = cb; c.s = s; c.gen = gen;
    return s;
}

void CmdStateFree(DeviceData* d, VkCommandBuffer cb)
{
    auto it = d->cmdStates.find(cb);
    if (it == d->cmdStates.end()) return;
    delete it->second;
    d->cmdStates.erase(it);
    d->cmdGen.fetch_add(1, std::memory_order_release);
}

static void ResetState(CmdState* s)
{
    const VkCommandBuffer cb = s->cb;
    const bool sec = s->secondary, secLevel = s->secondaryLevel;
    const bool foreign = s->foreign;
    std::unordered_map<VkPipelineLayout, CmdState::LayoutFacts> layouts;
    std::unordered_map<VkPipeline, CmdState::PipelineFacts> pipelines;
    layouts.swap(s->layoutCache);
    pipelines.swap(s->pipelineCache);
    const uint64_t gen = s->objGen;
    *s = CmdState();
    s->cb = cb;
    s->secondary = sec;
    s->secondaryLevel = secLevel;
    s->foreign = foreign;
    s->layoutCache.swap(layouts);
    s->pipelineCache.swap(pipelines);
    s->objGen = gen;
}

// Hands the submit the sets a command buffer bound, clearing the list. Lock held.
void CmdStateTakeUsedSets(DeviceData* d, VkCommandBuffer cb, std::vector<UsedSet>& out)
{
    auto it = d->cmdStates.find(cb);
    if (it == d->cmdStates.end()) return;
    CmdState* s = it->second;
    if (out.empty()) out.swap(s->usedSets);
    else { out.insert(out.end(), s->usedSets.begin(), s->usedSets.end()); s->usedSets.clear(); }
}

// Facts about a pipeline layout from the command buffer's cache, filled from
// the device map (locked) on a miss. Null when the layout is unknown.
static const CmdState::LayoutFacts* LayoutFactsFor(DeviceData* d, CmdState* s, VkPipelineLayout layout)
{
    const uint64_t gen = d->objGen.load(std::memory_order_acquire);
    if (s->objGen != gen) { s->layoutCache.clear(); s->pipelineCache.clear(); s->objGen = gen; }
    auto it = s->layoutCache.find(layout);
    if (it != s->layoutCache.end()) return it->second.patched || it->second.gameSets ? &it->second : nullptr;
    CmdState::LayoutFacts f = { false, 0, {} };
    bool known = false;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto lit = d->pipelineLayouts.find(layout);
        if (lit != d->pipelineLayouts.end()) { known = true; f.patched = lit->second.patched; f.gameSets = lit->second.gameSetCount; f.dyn = lit->second.dynCounts; }
    }
    auto& slot = s->layoutCache[layout];
    slot = f;
    return known ? &slot : nullptr;
}

static VkPipelineLayout PipelineLayoutFor(DeviceData* d, CmdState* s, VkPipeline pipeline, bool* shifted = nullptr,
                                          const CmdState::PipelineFacts** outFacts = nullptr)
{
    const uint64_t gen = d->objGen.load(std::memory_order_acquire);
    if (s->objGen != gen) { s->layoutCache.clear(); s->pipelineCache.clear(); s->objGen = gen; }
    auto it = s->pipelineCache.find(pipeline);
    if (it != s->pipelineCache.end())
    {
        if (shifted) *shifted = it->second.shifted;
        if (outFacts) *outFacts = &it->second;
        return it->second.layout;
    }
    CmdState::PipelineFacts f = { VK_NULL_HANDLE, false, false, 0, 0, false, false, 0, false, false };
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto pit = d->pipelines.find(pipeline);
        if (pit != d->pipelines.end()) { f.layout = pit->second.layout; f.shifted = pit->second.shifted;
                                       f.posUsesMatrix = pit->second.posUsesMatrix; f.posSet = pit->second.posSet; f.posMask = pit->second.posMask;
                                       f.posBinding = pit->second.posBinding;
                                       f.depthEqual = pit->second.depthOp == (int)VK_COMPARE_OP_EQUAL; f.remapCS = pit->second.remapCS; f.csCrc = pit->second.csCrc;
                                       f.depthOn = pit->second.depthOp >= 0; f.spriteShape = pit->second.depthOp < 0 && pit->second.rsxInputs >= 4;
                                       f.repointPos = FamiliesRepointVS(pit->second.vsCrc);
                                       f.worldQuad = FamiliesWorldQuadVS(pit->second.vsCrc); }
    }
    auto& slot = s->pipelineCache[pipeline];
    slot = f;
    if (shifted) *shifted = f.shifted;
    if (outFacts) *outFacts = &slot;
    return f.layout;
}

static bool Recording(const CmdState* s) { return (s->inPass && s->stereoPass) || s->secondary; }

static Rec& Push(CmdState* s, Op op)
{
    s->recs.emplace_back();
    s->recs.back().op = op;
    return s->recs.back();
}

static void Payload(CmdState* s, Rec& r, const void* p, size_t n)
{
    r.dataOff = (uint32_t)s->blob.size();
    r.dataLen = (uint32_t)n;
    s->blob.insert(s->blob.end(), (const uint8_t*)p, (const uint8_t*)p + n);
}

// Stereo set binding
static bool LayoutPatched(DeviceData* d, CmdState* s, VkPipelineLayout layout, uint32_t* gameSets)
{
    const CmdState::LayoutFacts* f = LayoutFactsFor(d, s, layout);
    if (!f) return false;
    if (gameSets) *gameSets = f->gameSets;
    return f->patched;
}

// A draw served by constants claims the set its position matrix lives in, naming the registers it multiplies by; a
// claim reaches only that draw's ring slice, so a screen pass reading the block through an unclaimed set stays put.
static void ClaimCameraSet(DeviceData* d, CmdState* s, bool fullScreen)
{
    if (!s->gfxByConstants || s->replaying || !gInfo.DetectInverseMatrix) return;
    if (s->gfxPosSet >= 32) return;
    const SetBind& b = s->gfx.sets[s->gfxPosSet];
    if (!b.set) return;
    UsedSet u;
    u.set = b.set;
    u.dynCount = (uint32_t)(b.dyn.size() > 8 ? 8 : b.dyn.size());
    for (uint32_t k = 0; k < u.dynCount; ++k) u.dyn[k] = b.dyn[k];
    u.fs = 0;
    // An EQUAL pass (invariant with its prepass, self-shifted) and a screen-covering quad (its patched projection would
    // move it) never claim; both record the range unclaimed, so the submit sees another draw's claim patching it.
    if ((s->gfxEqualPass && Vk().VKEqualPassSelfShift) || fullScreen)
    {
        // Past 8 dynamic offsets the bind-time entry was never queued; a truncated one here would guess a range.
        if (b.dyn.size() > 8) return;
        u.consumerMask = 0;
        u.noClaimPipe = s->gfx.pipeline;
        u.noClaimW = s->passW; u.noClaimH = s->passH;
        if (fullScreen) u.noClaimBinding = s->gfxPosBinding;
        s->usedSets.push_back(u);
        return;
    }
    u.consumerMask = s->gfxPosMask;
    u.pipeline = s->gfx.pipeline;   // the submit reports its verdict back to this pipeline
    if (s->gfxRepointPos) u.repointBinding = s->gfxPosBinding;
    if (Traits().positionBindings) u.posBinding = s->gfxPosBinding;
    s->usedSets.push_back(u);
}

// What a direct draw is, for both eyes: World (over 6 vertices, instanced, or a family's world quad); Screen (claims as
// full-screen, replays pinned); ScreenTriangle (screenTriangles: a camera-positioned triangle, ScreenTriangleSet).
enum class DrawKind { World, Screen, ScreenTriangle };
static DrawKind KindOfDraw(bool worldQuad, bool posUsesMatrix, uint32_t verts, uint32_t instances)
{
    if (verts > 6 || instances != 1 || worldQuad) return DrawKind::World;
    return (Traits().screenTriangles && verts == 3 && posUsesMatrix) ? DrawKind::ScreenTriangle : DrawKind::Screen;
}

// offAspectPasses: a target not shaped like the backbuffer (shadow atlases, light maps) replays unshifted, as a
// shifted caster there would move the shadow.
static void NotePassAspect(const DeviceData* d, CmdState* s)
{
    if (Traits().offAspectPasses && s->passW && s->passH && d->bbSize.width && d->bbSize.height) s->passAspectOk = !OffAspect(s->passW, s->passH, d->bbSize);
}

// A full-screen draw's bound sets, flagged: reconstruction bases are learned only in ranges these consume. Small
// matrix-positioned draws are decals and billboards, not screen passes, so they do not flag.
static void ClaimFullScreenSets(DeviceData* d, CmdState* s)
{
    if (s->replaying || !gInfo.DetectInverseMatrix || !d->constants) return;
    if (s->gfxByConstants) return;
    for (uint32_t i = 0; i < 32; ++i)
    {
        const SetBind& b = s->gfx.sets[i];
        if (!b.set) continue;
        UsedSet u;
        u.set = b.set;
        u.dynCount = (uint32_t)(b.dyn.size() > 8 ? 8 : b.dyn.size());
        for (uint32_t k = 0; k < u.dynCount; ++k) u.dyn[k] = b.dyn[k];
        u.consumerMask = 0;
        u.fs = 1;
        u.pipeline = VK_NULL_HANDLE;
        s->usedSets.push_back(u);
    }
}

// VKSmallDrawUnshift: a <=6-vertex non-instanced replay draw gets the sep=0 set, so screen-covering triangles never
// shift whatever their w; it cannot tell them from UI quads, so a game whose UI boxes are quads turns it off.
static bool SmallDrawUnshift(DeviceData* d, uint32_t verts, uint32_t instances)
{
    if (!Vk().VKSmallDrawUnshift) return false;
    const bool hit = verts <= 6 && instances == 1;
    if (hit) ++d->statSmallUnshift;
    return hit;
}

// <VKScreenDepth>: a draw whose vertex shader the profile lists gets the sep=0 set whatever its size.
static bool ScreenDepthDraw(DeviceData* d, CmdState* s)
{
    const std::vector<uint32_t>& list = ScreenDepthShaders();
    if (list.empty()) return false;
    std::shared_lock<std::shared_mutex> g(d->lock);
    auto it = d->pipelines.find(s->gfx.pipeline);
    return it != d->pipelines.end() && std::find(list.begin(), list.end(), it->second.vsCrc) != list.end();
}

// flatUiPasses: a one-colour pass into the display format (A2B10G10R10), or RGBA16F with no depth, is UI drawn with
// its own perspective, kept flat; a colourless pass inside the UI phase is its mask, whose quads stay flat too.
static bool ShadGtHudPass(DeviceData* d, CmdState* s, const VkRenderingInfo* p)
{
    if (!Traits().flatUiPasses || p->colorAttachmentCount > 1) return false;
    const bool attached = p->pDepthAttachment || p->pStencilAttachment;
    VkFormat f = VK_FORMAT_UNDEFINED;
    uint32_t w = p->renderArea.extent.width, h = p->renderArea.extent.height;
    if (p->colorAttachmentCount && p->pColorAttachments[0].imageView)
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto vit = d->views.find(p->pColorAttachments[0].imageView);
        auto iit = vit != d->views.end() ? d->images.find(vit->second.image) : d->images.end();
        if (iit == d->images.end()) return false;
        f = iit->second.ci.format;
        w = iit->second.ci.extent.width; h = iit->second.ci.extent.height;
    }
    if (f == VK_FORMAT_A2B10G10R10_UNORM_PACK32) d->shadGtHudPhase = true;
    else if (f == VK_FORMAT_B8G8R8A8_UNORM) d->shadGtHudPhase = false;
    s->shadGtMask = f == VK_FORMAT_UNDEFINED && d->shadGtHudPhase;
    const bool hud = f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || (f == VK_FORMAT_R16G16B16A16_SFLOAT && !attached);
    if (!hud) return false;
    static std::atomic<unsigned> s_hud{ 0 };
    const unsigned n = ++s_hud;
    if (n <= 3 || n % 60000 == 0) Log("[hud] %u UI passes kept flat, last %ux%u fmt=%d depth=%d\n", n, w, h, (int)f, (int)attached);
    return true;
}

// flatNoDepthDraws: a draw without the depth test lies flat on the screen (menus, post, UI), whatever its w;
// scene geometry tests depth.
static bool NoDepthUnshift(DeviceData* d, CmdState* s)
{
    if (!Traits().flatNoDepthDraws) return false;
    std::shared_lock<std::shared_mutex> g(d->lock);
    auto it = d->pipelines.find(s->gfx.pipeline);
    return it != d->pipelines.end() && it->second.depthOp < 0 && !it->second.midFrameSprite;
}

// A world sprite tests no depth either: it reads 4 or more attributes and is drawn before the frame's last
// depth-tested draw, where UI comes after; the verdict is last frame's, so a new pipeline starts pinned.
static void NoteSpriteOrder(DeviceData* d, CmdState* s)
{
    if (s->replaying || !s->stereoPass) return;
    if (s->gfxSpriteShape)
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (d->spritesPending.empty() || d->spritesPending.back() != s->gfx.pipeline) d->spritesPending.push_back(s->gfx.pipeline);
        return;
    }
    if (!s->gfxDepthOn || d->spritesPending.empty()) return;
    std::lock_guard<std::shared_mutex> g(d->lock);
    for (VkPipeline p : d->spritesPending)
    {
        auto it = d->pipelines.find(p);
        if (it == d->pipelines.end() || it->second.midFrameSprite) continue;
        it->second.midFrameSprite = true;
        static std::atomic<int> s_log{64};
        if (s_log.fetch_sub(1) > 0) Log("[sprite] VS %08X FS %08X (%u inputs): a world sprite, keeps its shift\n", it->second.vsCrc, it->second.fsCrc, it->second.rsxInputs);
    }
    d->spritesPending.clear();
}

// At present: whatever no depth-tested draw followed is UI this frame.
void SpriteFrameEnd(DeviceData* d)
{
    if (!Traits().flatNoDepthDraws) return;
    std::lock_guard<std::shared_mutex> g(d->lock);
    for (VkPipeline p : d->spritesPending)
    {
        auto it = d->pipelines.find(p);
        if (it == d->pipelines.end() || !it->second.midFrameSprite) continue;
        it->second.midFrameSprite = false;
        static std::atomic<int> s_log{64};
        if (s_log.fetch_sub(1) > 0) Log("[sprite] VS %08X FS %08X: after the last depth-tested draw, pinned again\n", it->second.vsCrc, it->second.fsCrc);
    }
    d->spritesPending.clear();
}

// portalScissor: an in-image portal (a mirror) is a small scissor whose first draw cuts a depth hole (depth ALWAYS);
// the rectangle is the left eye's bound and depth the real shape, so the right eye's scissor is widened.
static void RsxPortalScissor(DeviceData* d, CmdState* s, VkCommandBuffer cb, uint32_t verts)
{
    if (s->gfx.scissors.empty() || s->gfx.viewports.empty()) return;
    const VkRect2D rc = s->gfx.scissors[0];
    const VkViewport& vp = s->gfx.viewports[0];
    if ((float)rc.extent.width >= vp.width * 0.8f && (float)rc.extent.height >= vp.height * 0.5f) return;
    std::lock_guard<std::shared_mutex> g(d->lock);
    auto it = d->pipelines.find(s->gfx.pipeline);
    if (it == d->pipelines.end()) return;
    // A flat draw is pinned to the left's screen position, and so is its rectangle.
    if (it->second.depthOp < 0)
    {
        if (s->portalWide) { d->CmdSetScissor(cb, 0, 1, &rc); s->portalWide = false; }
        return;
    }
    DeviceData::RsxPortal* hit = nullptr;
    for (DeviceData::RsxPortal& p : d->rsxPortals)
        if (p.frame != ~0u && d->frames - p.frame <= 1 && p.rc.offset.x == rc.offset.x && p.rc.offset.y == rc.offset.y && p.rc.extent.width == rc.extent.width && p.rc.extent.height == rc.extent.height)
            hit = &p;
    if (!hit)
    {
        if (it->second.depthOp != (int)VK_COMPARE_OP_ALWAYS || !it->second.depthWrite || verts > 12) return;
        hit = &d->rsxPortals[d->rsxPortalNext++ % 4];
        hit->rc = rc;
        static std::atomic<int> s_log{ 8 };
        if (s_log.fetch_sub(1, std::memory_order_relaxed) > 0)
            Log("[portal] scissor %d,%d %ux%u, first draw VS %08X FS %08X n=%u cuts depth: the right eye's scissor is widened\n", rc.offset.x, rc.offset.y, rc.extent.width, rc.extent.height, it->second.vsCrc, it->second.fsCrc, verts);
    }
    hit->frame = d->frames;
    // The whole row, not a margin: a surface moves sep*(1 - conv/w), unbounded as w falls; depth still shapes it.
    const VkRect2D wide = { { 0, rc.offset.y }, { (uint32_t)vp.width, rc.extent.height } };
    d->CmdSetScissor(cb, 0, 1, &wide);
    s->portalWide = true;
}

// Bound lazily before each draw: set 8 is re-bound whenever the layout it was bound with is not the one about to draw.
static void EnsureEyeSet(DeviceData* d, CmdState* s, bool fullScreen = false, bool neverClaimed = false)
{
    if (Traits().flatNoDepthDraws) NoteSpriteOrder(d, s);
    if (s->replaying && !s->gfxShifted) ++d->statUnshiftedDraws;
    // A matrix-positioned pipeline the last submit patched gets the sep=0 set (its projection carries the eye), read
    // from the pipeline map, as the replay's BindPipeline refreshes only gfxShifted.
    bool served = false;
    // neverClaimed: a <=6-vertex one-instance draw never claims, so its range is never patched whatever its
    // pipeline's verdict.
    const bool claimable = !fullScreen && !neverClaimed;
    if (s->replaying && claimable && Vk().VKServeFallback && gInfo.DetectInverseMatrix)
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto it = d->pipelines.find(s->gfx.pipeline);
        if (it != d->pipelines.end() && it->second.posUsesMatrix)
        {
            const PipelineInfo& pi = it->second;
            // An EQUAL pass self-shifts, as its unserved prepass (a premultiplied MVP) does: one mechanism for both.
            served = pi.served && !(Vk().VKEqualPassSelfShift && pi.depthOp == (int)VK_COMPARE_OP_EQUAL);
            if (served) ++d->statServedDraws; else ++d->statFallbackDraws;
        }
    }
    if (s->replaying && !s->passAspectOk) ++d->statOffAspectDraws;
    // Pinned right-eye draws keep sep 0 but the eye offset: full-screen lighting reconstructs too (FsPatch.cpp).
    VkDescriptorSet want = !s->replaying ? d->stereoSetLeft
                         : (!fullScreen && !served && s->passAspectOk) ? d->stereoSetRight : d->stereoSetRightNoShift;
    if (Traits().rsxPositionTwins && s->replaying) want = StereoSetForWidth(d, want, s->gfx.viewports.empty() ? s->passW : (uint32_t)s->gfx.viewports[0].width);
    else if (Traits().widthSets && s->replaying)
    {
        // widthSets: the same per-width sets (member 5); the last answer is kept per thread.
        static thread_local uint32_t t_width = 0;
        static thread_local VkDescriptorSet t_base = VK_NULL_HANDLE, t_set = VK_NULL_HANDLE;
        const uint32_t vw = s->gfx.viewports.empty() ? s->passW : (uint32_t)s->gfx.viewports[0].width;
        if (vw != t_width || want != t_base) { t_set = StereoSetForWidth(d, want, vw); t_width = vw; t_base = want; }
        want = t_set;
    }
    if (s->eyeLayout == s->gfx.layout && s->eyeSet == want) return;
    if (!s->gfx.layout || !LayoutPatched(d, s, s->gfx.layout, nullptr)) { ++d->statUnpatchedDraws; return; }
    d->CmdBindDescriptorSets(s->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s->gfx.layout, kStereoSet, 1, &want, 0, nullptr);
    s->eyeLayout = s->gfx.layout;
    s->eyeSet = want;
    ++d->statEyeBinds;
}

// Push descriptors
static bool IsImageType(VkDescriptorType t)
{
    return t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
           t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT || t == VK_DESCRIPTOR_TYPE_SAMPLER;
}

// Blob: [uint32 count][PushRecord x count][payloads]. Payload is the
// descriptor array (image/buffer/texel view) or the inline uniform bytes.
static void SerializePushes(uint32_t count, const VkWriteDescriptorSet* w, std::vector<uint8_t>& blob)
{
    blob.clear();
    blob.resize(4 + count * sizeof(PushRecord));
    memcpy(blob.data(), &count, 4);
    for (uint32_t i = 0; i < count; ++i)
    {
        PushRecord r = { w[i].dstBinding, w[i].dstArrayElement, w[i].descriptorCount, (uint32_t)w[i].descriptorType, (uint32_t)blob.size(), 0 };
        const uint8_t* src = nullptr;
        if (w[i].descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
        {
            for (auto p = (const VkBaseInStructure*)w[i].pNext; p; p = p->pNext)
                if (p->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK)
                { src = (const uint8_t*)((const VkWriteDescriptorSetInlineUniformBlock*)p)->pData; r.dataLen = ((const VkWriteDescriptorSetInlineUniformBlock*)p)->dataSize; }
        }
        else if (IsImageType(w[i].descriptorType)) { src = (const uint8_t*)w[i].pImageInfo; r.dataLen = w[i].descriptorCount * sizeof(VkDescriptorImageInfo); }
        else if (w[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || w[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
        { src = (const uint8_t*)w[i].pTexelBufferView; r.dataLen = w[i].descriptorCount * sizeof(VkBufferView); }
        else { src = (const uint8_t*)w[i].pBufferInfo; r.dataLen = w[i].descriptorCount * sizeof(VkDescriptorBufferInfo); }
        if (src && r.dataLen) blob.insert(blob.end(), src, src + r.dataLen);
        else r.dataLen = 0;
        memcpy(blob.data() + 4 + i * sizeof(PushRecord), &r, sizeof(r));
    }
}

// VKPushDeltas: a push stands on the earlier ones under the same layout, for hosts that push only the changed bindings;
// the last call alone would leave the replay reading another draw's descriptors.
static void MergePushes(const uint8_t* in, size_t inLen, std::vector<uint8_t>& blob)
{
    uint32_t inCount, count;
    memcpy(&inCount, in, 4);
    memcpy(&count, blob.data(), 4);
    const size_t hdr = sizeof(PushRecord);
    if (count + inCount > 512) { blob.assign(in, in + inLen); return; }
    for (uint32_t i = 0; i < inCount; ++i)
    {
        PushRecord w; memcpy(&w, in + 4 + i * hdr, hdr);
        bool replaced = false;
        for (uint32_t k = 0; k < count && !replaced; ++k)
        {
            PushRecord r; memcpy(&r, blob.data() + 4 + k * hdr, hdr);
            if (r.binding != w.binding || r.arrayElement != w.arrayElement || r.count != w.count || r.type != w.type || r.dataLen != w.dataLen) continue;
            if (w.dataLen) memcpy(blob.data() + r.dataOff, in + w.dataOff, w.dataLen);
            replaced = true;
        }
        if (replaced) continue;
        // Writes apply in order, so a new shape goes last. Its record sits before the payloads, which all move along.
        for (uint32_t k = 0; k < count; ++k)
        {
            PushRecord r; memcpy(&r, blob.data() + 4 + k * hdr, hdr);
            r.dataOff += (uint32_t)hdr;
            memcpy(blob.data() + 4 + k * hdr, &r, hdr);
        }
        blob.insert(blob.begin() + 4 + count * hdr, hdr, 0);
        PushRecord added = w;
        added.dataOff = (uint32_t)blob.size();
        blob.insert(blob.end(), in + w.dataOff, in + w.dataOff + w.dataLen);
        memcpy(blob.data() + 4 + count * hdr, &added, hdr);
        ++count;
        memcpy(blob.data(), &count, 4);
    }
}

// A bind point's push state after this call.
static void KeepPushes(uint32_t count, const VkWriteDescriptorSet* w, std::vector<uint8_t>& blob, bool sameLayout)
{
    if (!Vk().VKPushDeltas || !sameLayout || blob.size() < 4) { SerializePushes(count, w, blob); return; }
    static thread_local std::vector<uint8_t> in;
    SerializePushes(count, w, in);
    MergePushes(in.data(), in.size(), blob);
}

// Lock held when right is set (sibling lookup).
static bool PushesHaveStereo(DeviceData* d, const std::vector<uint8_t>& blob, bool storageOnly = false)
{
    if (blob.size() < 4) return false;
    uint32_t count; memcpy(&count, blob.data(), 4);
    for (uint32_t i = 0; i < count; ++i)
    {
        PushRecord r; memcpy(&r, blob.data() + 4 + i * sizeof(PushRecord), sizeof(r));
        if (!IsImageType((VkDescriptorType)r.type) || r.type == VK_DESCRIPTOR_TYPE_SAMPLER) continue;
        if (storageOnly && r.type != (uint32_t)VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) continue;
        const auto* ii = (const VkDescriptorImageInfo*)(blob.data() + r.dataOff);
        for (uint32_t k = 0; k < r.count; ++k) if (ViewSibling(d, ii[k].imageView, nullptr)) return true;
    }
    return false;
}

// The right run's stand-in for a small written binding whose buffer has no twin (GDS, CPU-visible):
// a scratch slot seeded with the game's bytes, so the left eye's state never sees the right run's writes.
struct BufRedirect { VkBuffer from; VkDeviceSize off; VkBuffer to; VkDeviceSize toOff; };
static thread_local std::vector<BufRedirect> t_redirects;   // set around the right run's re-push
static thread_local bool t_leftBuffers = false;             // VKIndirectFromLeft: a right re-push keeps left buffers

static void IssuePushes(DeviceData* d, VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set,
                        const std::vector<uint8_t>& blob, bool right)
{
    if (blob.size() < 4 || !d->CmdPushDescriptorSetKHR) return;
    uint32_t count; memcpy(&count, blob.data(), 4);
    std::vector<VkWriteDescriptorSet> writes(count);
    std::vector<VkWriteDescriptorSetInlineUniformBlock> inl(count);
    std::vector<std::vector<VkDescriptorImageInfo>> imgs(count);
    // The right eye's buffers and texel views are the twins (VKBufferTwins).
    std::vector<std::vector<VkDescriptorBufferInfo>> bufs(count);
    std::vector<std::vector<VkBufferView>> texels(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        PushRecord r; memcpy(&r, blob.data() + 4 + i * sizeof(PushRecord), sizeof(r));
        VkWriteDescriptorSet& w = writes[i];
        w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstBinding = r.binding;
        w.dstArrayElement = r.arrayElement;
        w.descriptorCount = r.count;
        w.descriptorType = (VkDescriptorType)r.type;
        const uint8_t* data = blob.data() + r.dataOff;
        if (r.type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
        {
            inl[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK };
            inl[i].dataSize = r.dataLen;
            inl[i].pData = data;
            w.pNext = &inl[i];
        }
        else if (IsImageType(w.descriptorType))
        {
            imgs[i].assign((const VkDescriptorImageInfo*)data, (const VkDescriptorImageInfo*)data + r.count);
            if (right && w.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER)
            {
                std::shared_lock<std::shared_mutex> g(d->lock);
                for (auto& ii : imgs[i])
                {
                    // VKIndirectFromLeft=2: a draw of the left's particles keeps the left's 2D images not shaped like
                    // the screen (per-particle data read by particle index).
                    if (t_leftBuffers && Vk().VKIndirectFromLeft >= 2)
                    {
                        auto vit = d->views.find(ii.imageView);
                        auto iit = vit != d->views.end() ? d->images.find(vit->second.image) : d->images.end();
                        if (iit != d->images.end() && iit->second.ci.imageType == VK_IMAGE_TYPE_2D &&
                            OffAspect(iit->second.ci.extent.width, iit->second.ci.extent.height, d->bbSize.width ? d->bbSize : VkExtent2D{ 16, 9 }))
                            continue;
                    }
                    VkImageView sv;
                    if (ViewSibling(d, ii.imageView, &sv)) ii.imageView = sv;
                    else if (ViewOfDoubledImage(d, ii.imageView)) d->statLeftInRight.fetch_add(1, std::memory_order_relaxed);
                }
            }
            w.pImageInfo = imgs[i].data();
        }
        else if (w.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
        {
            texels[i].assign((const VkBufferView*)data, (const VkBufferView*)data + r.count);
            if (right && Vk().VKBufferTwins && !t_leftBuffers)
            {
                std::shared_lock<std::shared_mutex> g(d->lock);
                for (auto& v : texels[i]) v = BufferViewTwin(d, v);
            }
            w.pTexelBufferView = texels[i].data();
        }
        else
        {
            bufs[i].assign((const VkDescriptorBufferInfo*)data, (const VkDescriptorBufferInfo*)data + r.count);
            if (right && Vk().VKBufferTwins && !t_leftBuffers)
            {
                std::shared_lock<std::shared_mutex> g(d->lock);
                for (auto& b : bufs[i])
                {
                    bool redirected = false;
                    for (const BufRedirect& rd : t_redirects)
                        if (rd.from == b.buffer && rd.off == b.offset) { b.buffer = rd.to; b.offset = rd.toOff; redirected = true; break; }
                    if (redirected) continue;
                    if (const VkBuffer twin = BufferTwin(d, b.buffer)) b.buffer = twin;
                }
            }
            w.pBufferInfo = bufs[i].data();
        }
    }
    d->CmdPushDescriptorSetKHR(cb, bp, layout, set, count, writes.data());
}

// Lock held. What one serialised push binds for the right eye: doubled images or twinned buffers (`stereo`),
// and written bindings whose buffers have no twin -- a right run must not write the game's.
struct PushScan
{
    bool stereo = false;
    std::vector<VkDescriptorBufferInfo> untwinnedWrites;
    std::vector<VkDescriptorBufferInfo> twinnedWrites;   // written storage ranges the right run writes into twins
    bool untwinnedTexelWrite = false;
};

static void ScanPush(DeviceData* d, const std::vector<uint8_t>& blob, uint32_t set, const std::vector<uint64_t>* writes, PushScan& out)
{
    if (blob.size() < 4) return;
    uint32_t count; memcpy(&count, blob.data(), 4);
    for (uint32_t i = 0; i < count && 4 + (i + 1) * sizeof(PushRecord) <= blob.size(); ++i)
    {
        PushRecord r; memcpy(&r, blob.data() + 4 + i * sizeof(PushRecord), sizeof(r));
        const VkDescriptorType t = (VkDescriptorType)r.type;
        const uint8_t* p = blob.data() + r.dataOff;
        bool written = false;
        if (writes) for (uint64_t w : *writes) if (w == (((uint64_t)set << 32) | r.binding)) { written = true; break; }
        if (IsImageType(t))
        {
            if (t == VK_DESCRIPTOR_TYPE_SAMPLER) continue;
            for (uint32_t k = 0; k < r.count; ++k)
            {
                VkDescriptorImageInfo ii; memcpy(&ii, p + k * sizeof(ii), sizeof(ii));
                if (ViewSibling(d, ii.imageView, nullptr)) out.stereo = true;
            }
        }
        else if (t == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
        {
            for (uint32_t k = 0; k < r.count; ++k)
            {
                VkBufferView v; memcpy(&v, p + k * sizeof(v), sizeof(v));
                auto it = d->bufferViews.find(v);
                if (it == d->bufferViews.end()) continue;
                if (BufferTwin(d, it->second.ci.buffer)) out.stereo = true;
                else if (written && t == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) out.untwinnedTexelWrite = true;
            }
        }
        else if (t != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
        {
            const bool storage = t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
            for (uint32_t k = 0; k < r.count; ++k)
            {
                VkDescriptorBufferInfo bi; memcpy(&bi, p + k * sizeof(bi), sizeof(bi));
                if (BufferTwin(d, bi.buffer)) { out.stereo = true; if (written && storage) out.twinnedWrites.push_back(bi); }
                else if (written && storage) out.untwinnedWrites.push_back(bi);
            }
        }
    }
}

static void TrackPush(DeviceData* d, CmdState* s, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t count, const VkWriteDescriptorSet* pWrites)
{
    ++d->statPushSets;
    for (uint32_t i = 0; i < count; ++i)
        if (pWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) { ++d->statInlineBlocks; d->statInlineBytes += pWrites[i].descriptorCount; }
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE && set < 32)
    {
        KeepPushes(count, pWrites, s->compPushes[set], s->compPushLayout[set] == layout);
        s->compPushLayout[set] = layout;
        // As for graphics: a push replaces the slot's set, or BindComputeShadows would rebind its stale shadow
        // instead of re-pushing.
        s->compSets[set].set = VK_NULL_HANDLE;
    }
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS && set < 32)
    {
        KeepPushes(count, pWrites, s->gfx.pushSets[set], s->gfx.pushLayout[set] == layout);
        s->gfx.pushLayout[set] = layout;
        s->gfx.sets[set].set = VK_NULL_HANDLE;      // a push replaces whatever set was bound there
    }
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS && layout != s->eyeLayout) s->eyeLayout = VK_NULL_HANDLE;
    if (Recording(s) && !s->replaying)
    {
        std::vector<uint8_t> blob;
        SerializePushes(count, pWrites, blob);
        Rec& r = Push(s, Op::PushDescriptors);
        r.u[0] = bp; r.u[1] = set; r.h[0] = (uint64_t)layout;
        Payload(s, r, blob.data(), blob.size());
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdPushDescriptorSetKHR(
    VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t count, const VkWriteDescriptorSet* pWrites)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdPushDescriptorSetKHR(cb, bp, layout, set, count, pWrites);
    if (d->stereoEnabled) TrackPush(d, s, bp, layout, set, count, pWrites);
}

// Templates describe raw memory; expand to writes so one path serves both.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdPushDescriptorSetWithTemplateKHR(
    VkCommandBuffer cb, VkDescriptorUpdateTemplate t, VkPipelineLayout layout, uint32_t set, const void* pData)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdPushDescriptorSetWithTemplateKHR(cb, t, layout, set, pData);
    if (!d->stereoEnabled) return;
    DescriptorUpdateTemplateInfo info;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->templates.find(t);
        if (it == d->templates.end()) return;
        info = it->second;
    }
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkWriteDescriptorSetInlineUniformBlock> inl(info.entries.size());
    std::vector<std::vector<VkDescriptorImageInfo>> imgs;
    std::vector<std::vector<VkDescriptorBufferInfo>> bufs;
    std::vector<std::vector<VkBufferView>> texels;
    imgs.reserve(info.entries.size()); bufs.reserve(info.entries.size()); texels.reserve(info.entries.size());
    size_t ei = 0;
    for (const auto& e : info.entries)
    {
        VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstBinding = e.dstBinding;
        w.dstArrayElement = e.dstArrayElement;
        w.descriptorCount = e.descriptorCount;
        w.descriptorType = e.descriptorType;
        const char* base = (const char*)pData + e.offset;
        if (e.descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
        {
            inl[ei] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK };
            inl[ei].dataSize = e.descriptorCount;
            inl[ei].pData = base;
            w.pNext = &inl[ei];
        }
        else if (IsImageType(e.descriptorType))
        {
            imgs.emplace_back();
            for (uint32_t k = 0; k < e.descriptorCount; ++k) imgs.back().push_back(*(const VkDescriptorImageInfo*)(base + k * e.stride));
            w.pImageInfo = imgs.back().data();
        }
        else if (e.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || e.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
        {
            texels.emplace_back();
            for (uint32_t k = 0; k < e.descriptorCount; ++k) texels.back().push_back(*(const VkBufferView*)(base + k * e.stride));
            w.pTexelBufferView = texels.back().data();
        }
        else
        {
            bufs.emplace_back();
            for (uint32_t k = 0; k < e.descriptorCount; ++k) bufs.back().push_back(*(const VkDescriptorBufferInfo*)(base + k * e.stride));
            w.pBufferInfo = bufs.back().data();
        }
        writes.push_back(w);
        ++ei;
    }
    TrackPush(d, s, info.bindPoint, layout, set, (uint32_t)writes.size(), writes.data());
}

// Lifecycle
static VKAPI_ATTR VkResult VKAPI_CALL Hook_AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pInfo, VkCommandBuffer* pCbs)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->AllocateCommandBuffers(device, pInfo, pCbs);
    if (r != VK_SUCCESS) return r;
    const bool foreign = FgForeignCaller();
    std::lock_guard<std::shared_mutex> g(d->lock);
    for (uint32_t i = 0; i < pInfo->commandBufferCount; ++i)
    {
        d->cmdPools[pCbs[i]] = pInfo->commandPool;
        CmdStateFree(d, pCbs[i]);
        CmdState* s = new CmdState();
        s->cb = pCbs[i];
        s->foreign = foreign;
        s->secondary = s->secondaryLevel = pInfo->level == VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        d->cmdStates[pCbs[i]] = s;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_FreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t count, const VkCommandBuffer* pCbs)
{
    DeviceData* d = FindDevice(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (uint32_t i = 0; i < count; ++i) { CmdStateFree(d, pCbs[i]); d->cmdPools.erase(pCbs[i]); }
    }
    d->FreeCommandBuffers(device, pool, count, pCbs);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_ResetCommandPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags)
{
    DeviceData* d = FindDevice(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (auto& kv : d->cmdPools)
            if (kv.second == pool)
            {
                auto it = d->cmdStates.find(kv.first);
                if (it != d->cmdStates.end()) ResetState(it->second);
            }
    }
    return d->ResetCommandPool(device, pool, flags);
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyCommandPool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (auto it = d->cmdPools.begin(); it != d->cmdPools.end();)
        {
            if (it->second == pool) { CmdStateFree(d, it->first); it = d->cmdPools.erase(it); }
            else ++it;
        }
    }
    d->DestroyCommandPool(device, pool, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* pBeginInfo)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    ResetState(s);
    // VKDynamicRendering: secondaries that begin and end their own rendering are primaries to us; only one continuing
    // a pass inherits anything.
    if (Vk().VKDynamicRendering && s->secondaryLevel) s->secondary = (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT) != 0;
    return d->BeginCommandBuffer(cb, pBeginInfo);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_EndCommandBuffer(VkCommandBuffer cb)
{
    DeviceData* d = FindDevice(cb);
    return d->EndCommandBuffer(cb);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_ResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags flags)
{
    DeviceData* d = FindDevice(cb);
    ResetState(State(d, cb));
    return d->ResetCommandBuffer(cb, flags);
}

// State binds
static VKAPI_ATTR void VKAPI_CALL Hook_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipeline)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    d->CmdBindPipeline(cb, bp, pipeline);
    bool shifted = false;
    const CmdState::PipelineFacts* facts = nullptr;
    VkPipelineLayout layout = PipelineLayoutFor(d, s, pipeline, &shifted, &facts);
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS)
    {
        s->gfx.pipeline = pipeline; s->gfx.layout = layout; s->gfxShifted = shifted;
        s->gfxByConstants = facts && facts->posUsesMatrix;
        s->gfxPosSet = facts ? facts->posSet : 0;
        s->gfxPosBinding = facts ? facts->posBinding : UINT32_MAX;
        s->gfxPosMask = facts ? facts->posMask : 0;
        s->gfxRepointPos = facts && facts->repointPos;
        s->gfxWorldQuad = facts && facts->worldQuad;
        s->gfxEqualPass = facts && facts->depthEqual;
        s->gfxDepthOn = facts && facts->depthOn;
        s->gfxSpriteShape = facts && facts->spriteShape;
    }
    else if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) { s->compPipeline = pipeline; s->compLayout = layout; s->compRemap = facts && facts->remapCS; s->compCrc = facts ? facts->csCrc : 0; }
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::BindPipeline);
        r.u[0] = bp; r.h[0] = (uint64_t)pipeline; r.h[1] = (uint64_t)layout;
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBindDescriptorSets(
    VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t first, uint32_t count,
    const VkDescriptorSet* pSets, uint32_t dynCount, const uint32_t* pDyn)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    d->CmdBindDescriptorSets(cb, bp, layout, first, count, pSets, dynCount, pDyn);
    PerfScope pt(d->perf.binds);
    SetBind* table = bp == VK_PIPELINE_BIND_POINT_GRAPHICS ? s->gfx.sets : (bp == VK_PIPELINE_BIND_POINT_COMPUTE ? s->compSets : nullptr);
    if (table)
    {
        // Dynamic offsets are consumed in binding order across the sets: split per set (the layout's table serves, by
        // set compatibility), so each can be re-bound on its own.
        uint32_t used = 0;
        const CmdState::LayoutFacts* f = LayoutFactsFor(d, s, layout);
        for (uint32_t i = 0; i < count && first + i < 32; ++i)
        {
            SetBind& b = table[first + i];
            b.set = pSets[i];
            b.layout = layout;
            b.dyn.clear();
            if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) s->gfx.pushSets[first + i].clear();
            else s->compPushes[first + i].clear();
            const uint32_t n = (f && first + i < f->dyn.size()) ? f->dyn[first + i] : 0;
            if (n && used + n <= dynCount) b.dyn.assign(pDyn + used, pDyn + used + n);
            // More dynamic offsets than kept: never guess a range.
            if (gInfo.DetectInverseMatrix && !s->replaying && n <= 8)
            {
                UsedSet u;
                u.set = pSets[i];
                u.dynCount = n;
                u.consumerMask = 0;   // fs stays 0: a bind is no full-screen draw
                for (uint32_t k = 0; k < u.dynCount && used + k < dynCount; ++k) u.dyn[k] = pDyn[used + k];
                s->usedSets.push_back(u);
            }
            used += n;
        }
        if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS && layout != s->eyeLayout) s->eyeLayout = VK_NULL_HANDLE;
    }
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::BindSets);
        r.u[0] = bp; r.u[1] = first; r.u[2] = count; r.u[3] = dynCount; r.h[0] = (uint64_t)layout;
        std::vector<uint8_t> tmp(count * sizeof(VkDescriptorSet) + dynCount * sizeof(uint32_t));
        memcpy(tmp.data(), pSets, count * sizeof(VkDescriptorSet));
        if (dynCount) memcpy(tmp.data() + count * sizeof(VkDescriptorSet), pDyn, dynCount * sizeof(uint32_t));
        Payload(s, r, tmp.data(), tmp.size());
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkIndexType type)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    d->CmdBindIndexBuffer(cb, buffer, offset, type);
    s->gfx.indexBuf = buffer; s->gfx.indexOff = offset; s->gfx.indexType = type; s->gfx.hasIndex = true; s->gfx.index2 = false;
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::BindIndex);
        r.h[0] = (uint64_t)buffer; r.h[1] = offset; r.u[0] = type;
    }
}

// VK_KHR_maintenance5 / core 1.4: the same bind with a size.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdBindIndexBuffer2(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, VkIndexType type)
{
    DeviceData* d = FindDevice(cb);
    d->CmdBindIndexBuffer2(cb, buffer, offset, size, type);
    if (!Vk().VKDynamicRendering) return;
    CmdState* s = State(d, cb);
    s->gfx.indexBuf = buffer; s->gfx.indexOff = offset; s->gfx.indexSize = size; s->gfx.indexType = type;
    s->gfx.hasIndex = true; s->gfx.index2 = true;
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::BindIndex2);
        r.h[0] = (uint64_t)buffer; r.h[1] = offset; r.h[2] = size; r.u[0] = type;
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkBuffer* pBuffers, const VkDeviceSize* pOffsets)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    d->CmdBindVertexBuffers(cb, first, count, pBuffers, pOffsets);
    for (uint32_t i = 0; i < count && first + i < 32; ++i)
    {
        s->gfx.vb[first + i] = pBuffers[i]; s->gfx.vbOff[first + i] = pOffsets[i]; s->gfx.vbMask |= 1u << (first + i);
        s->gfx.vb2Mask &= ~(1u << (first + i));
        s->gfx.vbStrideMask &= ~(1u << (first + i));
    }
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::BindVertex);
        r.u[0] = first; r.u[1] = count;
        std::vector<uint8_t> tmp(count * (sizeof(VkBuffer) + sizeof(VkDeviceSize)));
        memcpy(tmp.data(), pBuffers, count * sizeof(VkBuffer));
        memcpy(tmp.data() + count * sizeof(VkBuffer), pOffsets, count * sizeof(VkDeviceSize));
        Payload(s, r, tmp.data(), tmp.size());
    }
}

// VK_EXT_extended_dynamic_state / core 1.3: hosts may bind every vertex buffer here, with the stride dynamic, so the
// replay restores these bindings through this call.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdBindVertexBuffers2(
    VkCommandBuffer cb, uint32_t first, uint32_t count, const VkBuffer* pBuffers, const VkDeviceSize* pOffsets,
    const VkDeviceSize* pSizes, const VkDeviceSize* pStrides)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    // Unresolved under both names here (the game got it from the loader): the core call keeps it rendering, less the
    // dynamic stride.
    if (!d->CmdBindVertexBuffers2)
    {
        static bool s_said = false;
        if (!s_said) { s_said = true; Log("!! vkCmdBindVertexBuffers2 unresolved, falling back to the core bind\n"); }
        Hook_CmdBindVertexBuffers(cb, first, count, pBuffers, pOffsets);
        return;
    }
    d->CmdBindVertexBuffers2(cb, first, count, pBuffers, pOffsets, pSizes, pStrides);
    for (uint32_t i = 0; i < count && first + i < 32; ++i)
    {
        const uint32_t b = first + i;
        s->gfx.vb[b] = pBuffers[i];
        s->gfx.vbOff[b] = pOffsets ? pOffsets[i] : 0;
        // VK_WHOLE_SIZE is what a null pSizes means, and 0 is what a null
        // pStrides means, so the restore is faithful either way.
        s->gfx.vbSize[b] = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
        s->gfx.vbStride[b] = pStrides ? pStrides[i] : 0;
        s->gfx.vbMask |= 1u << b;
        s->gfx.vb2Mask |= 1u << b;
        if (pStrides) s->gfx.vbStrideMask |= 1u << b; else s->gfx.vbStrideMask &= ~(1u << b);
    }
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::BindVertex2);
        r.u[0] = first; r.u[1] = count;
        r.u[2] = (pOffsets ? 1u : 0u) | (pSizes ? 2u : 0u) | (pStrides ? 4u : 0u);
        // buffers, then offsets, sizes and strides, each `count` long and each
        // written even when the game passed null so the replay indexes them flat.
        std::vector<uint8_t> tmp(count * (sizeof(VkBuffer) + 3 * sizeof(VkDeviceSize)));
        uint8_t* w = tmp.data();
        memcpy(w, pBuffers, count * sizeof(VkBuffer)); w += count * sizeof(VkBuffer);
        for (int part = 0; part < 3; ++part)
        {
            const VkDeviceSize* src = part == 0 ? pOffsets : part == 1 ? pSizes : pStrides;
            const VkDeviceSize dflt = part == 1 ? VK_WHOLE_SIZE : 0;
            for (uint32_t i = 0; i < count; ++i)
            {
                const VkDeviceSize v = src ? src[i] : dflt;
                memcpy(w, &v, sizeof(v)); w += sizeof(v);
            }
        }
        Payload(s, r, tmp.data(), tmp.size());
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* pValues)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    d->CmdPushConstants(cb, layout, stages, offset, size, pValues);
    ++d->statPushConst; d->statPushConstBytes += size;
    if (d->tiles && (stages & VK_SHADER_STAGE_COMPUTE_BIT) && offset == 0 && size <= sizeof(s->compPc))
    {
        memcpy(s->compPc, pValues, size);
        s->compPcSize = size; s->compPcLayout = layout; s->compPcStages = stages;
    }
    if (size <= 256)
    {
        PushRange* slot = nullptr;
        for (auto& p : s->gfx.pushes) if (p.offset == offset && p.size == size && p.stages == stages) { slot = &p; break; }
        const size_t cap = Vk().VKPushConstantOrder ? 64 : 16;
        if (!slot && s->gfx.pushes.size() < cap) { s->gfx.pushes.emplace_back(); slot = &s->gfx.pushes.back(); }
        // Full: the range written longest ago gives way, not the one being written now.
        if (!slot && Vk().VKPushConstantOrder)
            slot = &*std::min_element(s->gfx.pushes.begin(), s->gfx.pushes.end(), [](const PushRange& a, const PushRange& b) { return a.seq < b.seq; });
        if (slot) { slot->layout = layout; slot->stages = stages; slot->offset = offset; slot->size = size; memcpy(slot->data, pValues, size); slot->seq = ++s->gfx.pushSeq; }
    }
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::PushConstants);
        r.h[0] = (uint64_t)layout; r.u[0] = stages; r.u[1] = offset; r.u[2] = size;
        Payload(s, r, pValues, size);
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkViewport* p)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    d->CmdSetViewport(cb, first, count, p);
    s->gfx.viewports.assign(p, p + count); s->gfx.viewportFirst = first;
    if (!s->replaying) s->gfx.viewportSeq = ++s->gfx.dynSeq;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::Viewport); r.u[0] = first; r.u[1] = count; Payload(s, r, p, count * sizeof(VkViewport)); }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkRect2D* p)
{
    DeviceData* d = FindDevice(cb);
    CmdState* s = State(d, cb);
    d->CmdSetScissor(cb, first, count, p);
    s->gfx.scissors.assign(p, p + count); s->gfx.scissorFirst = first;
    if (!s->replaying) s->gfx.scissorSeq = ++s->gfx.dynSeq;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::Scissor); r.u[0] = first; r.u[1] = count; Payload(s, r, p, count * sizeof(VkRect2D)); }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetLineWidth(VkCommandBuffer cb, float w)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdSetLineWidth(cb, w);
    s->gfx.lineWidth = w; s->gfx.hasLineWidth = true;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::LineWidth); r.f[0] = w; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetDepthBias(VkCommandBuffer cb, float c, float clamp, float slope)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdSetDepthBias(cb, c, clamp, slope);
    s->gfx.depthBias[0] = c; s->gfx.depthBias[1] = clamp; s->gfx.depthBias[2] = slope; s->gfx.hasDepthBias = true;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::DepthBias); r.f[0] = c; r.f[1] = clamp; r.f[2] = slope; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetBlendConstants(VkCommandBuffer cb, const float k[4])
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdSetBlendConstants(cb, k);
    memcpy(s->gfx.blend, k, 16); s->gfx.hasBlend = true;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::BlendConstants); memcpy(r.f, k, 16); }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetDepthBounds(VkCommandBuffer cb, float mn, float mx)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdSetDepthBounds(cb, mn, mx);
    s->gfx.depthBounds[0] = mn; s->gfx.depthBounds[1] = mx; s->gfx.hasDepthBounds = true;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::DepthBounds); r.f[0] = mn; r.f[1] = mx; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetStencilCompareMask(VkCommandBuffer cb, VkStencilFaceFlags face, uint32_t v)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdSetStencilCompareMask(cb, face, v);
    if (face & VK_STENCIL_FACE_FRONT_BIT) s->gfx.stencilCompare.front = v;
    if (face & VK_STENCIL_FACE_BACK_BIT) s->gfx.stencilCompare.back = v;
    s->gfx.stencilCompare.has = true;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::StencilCompare); r.u[0] = face; r.u[1] = v; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetStencilWriteMask(VkCommandBuffer cb, VkStencilFaceFlags face, uint32_t v)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdSetStencilWriteMask(cb, face, v);
    if (face & VK_STENCIL_FACE_FRONT_BIT) s->gfx.stencilWrite.front = v;
    if (face & VK_STENCIL_FACE_BACK_BIT) s->gfx.stencilWrite.back = v;
    s->gfx.stencilWrite.has = true;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::StencilWrite); r.u[0] = face; r.u[1] = v; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetStencilReference(VkCommandBuffer cb, VkStencilFaceFlags face, uint32_t v)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdSetStencilReference(cb, face, v);
    if (face & VK_STENCIL_FACE_FRONT_BIT) s->gfx.stencilRef.front = v;
    if (face & VK_STENCIL_FACE_BACK_BIT) s->gfx.stencilRef.back = v;
    s->gfx.stencilRef.has = true;
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::StencilRef); r.u[0] = face; r.u[1] = v; }
}

// Extended dynamic state 1-3, vertex input, feedback loop
static bool DynOn(DeviceData* d) { return d->stereoEnabled && Vk().VKDynamicRendering; }

// The latest value per (state, key) for Reestablish, and a record inside a pass so the replay re-issues it.
static void TrackDyn(CmdState* s, Dyn fn, uint32_t key, const void* p, size_t n)
{
    if (s->replaying) return;
    DynState* slot = nullptr;
    for (DynState& e : s->gfx.dyn) if (e.fn == fn && e.key == key) { slot = &e; break; }
    if (!slot) { s->gfx.dyn.emplace_back(); slot = &s->gfx.dyn.back(); slot->fn = fn; slot->key = key; }
    slot->seq = ++s->gfx.dynSeq;
    slot->assign(p, n);
    if (Recording(s)) { Rec& r = Push(s, Op::DynState); r.u[0] = (uint32_t)fn; r.u[1] = key; Payload(s, r, p, n); }
}

// (function, state, argument): one value carried as 32 bits.
#define WIZ3D_DYN_SCALARS(X) \
    X(CmdSetCullMode, CullMode, VkCullModeFlags) \
    X(CmdSetFrontFace, FrontFace, VkFrontFace) \
    X(CmdSetPrimitiveTopology, PrimitiveTopology, VkPrimitiveTopology) \
    X(CmdSetDepthTestEnable, DepthTestEnable, VkBool32) \
    X(CmdSetDepthWriteEnable, DepthWriteEnable, VkBool32) \
    X(CmdSetDepthCompareOp, DepthCompareOp, VkCompareOp) \
    X(CmdSetDepthBoundsTestEnable, DepthBoundsTestEnable, VkBool32) \
    X(CmdSetStencilTestEnable, StencilTestEnable, VkBool32) \
    X(CmdSetRasterizerDiscardEnable, RasterizerDiscardEnable, VkBool32) \
    X(CmdSetDepthBiasEnable, DepthBiasEnable, VkBool32) \
    X(CmdSetPrimitiveRestartEnable, PrimitiveRestartEnable, VkBool32) \
    X(CmdSetPatchControlPointsEXT, PatchControlPoints, uint32_t) \
    X(CmdSetLogicOpEXT, LogicOp, VkLogicOp) \
    X(CmdSetTessellationDomainOriginEXT, TessellationDomainOrigin, VkTessellationDomainOrigin) \
    X(CmdSetDepthClampEnableEXT, DepthClampEnable, VkBool32) \
    X(CmdSetPolygonModeEXT, PolygonMode, VkPolygonMode) \
    X(CmdSetRasterizationSamplesEXT, RasterizationSamples, VkSampleCountFlagBits) \
    X(CmdSetAlphaToCoverageEnableEXT, AlphaToCoverageEnable, VkBool32) \
    X(CmdSetAlphaToOneEnableEXT, AlphaToOneEnable, VkBool32) \
    X(CmdSetLogicOpEnableEXT, LogicOpEnable, VkBool32) \
    X(CmdSetRasterizationStreamEXT, RasterizationStream, uint32_t) \
    X(CmdSetConservativeRasterizationModeEXT, ConservativeRasterizationMode, VkConservativeRasterizationModeEXT) \
    X(CmdSetDepthClipEnableEXT, DepthClipEnable, VkBool32) \
    X(CmdSetSampleLocationsEnableEXT, SampleLocationsEnable, VkBool32) \
    X(CmdSetProvokingVertexModeEXT, ProvokingVertexMode, VkProvokingVertexModeEXT) \
    X(CmdSetLineRasterizationModeEXT, LineRasterizationMode, VkLineRasterizationModeEXT) \
    X(CmdSetLineStippleEnableEXT, LineStippleEnable, VkBool32) \
    X(CmdSetDepthClipNegativeOneToOneEXT, DepthClipNegativeOneToOne, VkBool32) \
    X(CmdSetAttachmentFeedbackLoopEnableEXT, AttachmentFeedbackLoopEnable, VkImageAspectFlags)

// (first, count, values): keyed by the first attachment.
#define WIZ3D_DYN_RANGES(X) \
    X(CmdSetColorBlendEnableEXT, ColorBlendEnable, VkBool32) \
    X(CmdSetColorBlendEquationEXT, ColorBlendEquation, VkColorBlendEquationEXT) \
    X(CmdSetColorWriteMaskEXT, ColorWriteMask, VkColorComponentFlags) \
    X(CmdSetColorBlendAdvancedEXT, ColorBlendAdvanced, VkColorBlendAdvancedEXT)

// (count, values)
#define WIZ3D_DYN_COUNTED(X) \
    X(CmdSetViewportWithCount, ViewportWithCount, VkViewport) \
    X(CmdSetScissorWithCount, ScissorWithCount, VkRect2D) \
    X(CmdSetColorWriteEnableEXT, ColorWriteEnable, VkBool32)

#define X(Fn, Name, T) \
static VKAPI_ATTR void VKAPI_CALL Hook_##Fn(VkCommandBuffer cb, T v) \
{ \
    DeviceData* d = FindDevice(cb); \
    d->Fn(cb, v); \
    if (!DynOn(d)) return; \
    const uint32_t u = (uint32_t)v; \
    TrackDyn(State(d, cb), Dyn::Name, 0, &u, sizeof(u)); \
}
WIZ3D_DYN_SCALARS(X)
#undef X

#define X(Fn, Name, T) \
static VKAPI_ATTR void VKAPI_CALL Hook_##Fn(VkCommandBuffer cb, uint32_t first, uint32_t count, const T* p) \
{ \
    DeviceData* d = FindDevice(cb); \
    d->Fn(cb, first, count, p); \
    if (DynOn(d)) TrackDyn(State(d, cb), Dyn::Name, first, p, count * sizeof(T)); \
}
WIZ3D_DYN_RANGES(X)
#undef X

#define X(Fn, Name, T) \
static VKAPI_ATTR void VKAPI_CALL Hook_##Fn(VkCommandBuffer cb, uint32_t count, const T* p) \
{ \
    DeviceData* d = FindDevice(cb); \
    d->Fn(cb, count, p); \
    if (DynOn(d)) TrackDyn(State(d, cb), Dyn::Name, 0, p, count * sizeof(T)); \
}
WIZ3D_DYN_COUNTED(X)
#undef X

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetStencilOp(VkCommandBuffer cb, VkStencilFaceFlags face, VkStencilOp fail, VkStencilOp pass,
                                                       VkStencilOp depthFail, VkCompareOp op)
{
    DeviceData* d = FindDevice(cb);
    d->CmdSetStencilOp(cb, face, fail, pass, depthFail, op);
    const uint32_t v[4] = { (uint32_t)fail, (uint32_t)pass, (uint32_t)depthFail, (uint32_t)op };
    if (DynOn(d)) TrackDyn(State(d, cb), Dyn::StencilOp, face, v, sizeof(v));
}

// Payload: sample count, then its mask words (64 samples at most: two).
static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetSampleMaskEXT(VkCommandBuffer cb, VkSampleCountFlagBits samples, const VkSampleMask* pMask)
{
    DeviceData* d = FindDevice(cb);
    d->CmdSetSampleMaskEXT(cb, samples, pMask);
    if (!DynOn(d)) return;
    uint32_t v[3] = { (uint32_t)samples, 0, 0 };
    uint32_t words = ((uint32_t)samples + 31) / 32;
    if (words > 2) words = 2;
    for (uint32_t i = 0; i < words; ++i) v[1 + i] = pMask[i];
    TrackDyn(State(d, cb), Dyn::SampleMask, 0, v, (1 + words) * sizeof(uint32_t));
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetExtraPrimitiveOverestimationSizeEXT(VkCommandBuffer cb, float size)
{
    DeviceData* d = FindDevice(cb);
    d->CmdSetExtraPrimitiveOverestimationSizeEXT(cb, size);
    if (DynOn(d)) TrackDyn(State(d, cb), Dyn::ExtraPrimitiveOverestimationSize, 0, &size, sizeof(size));
}

// Payload: binding count, attribute count, then both arrays with pNext cleared.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdSetVertexInputEXT(VkCommandBuffer cb, uint32_t bc, const VkVertexInputBindingDescription2EXT* pB,
                                                            uint32_t ac, const VkVertexInputAttributeDescription2EXT* pA)
{
    DeviceData* d = FindDevice(cb);
    d->CmdSetVertexInputEXT(cb, bc, pB, ac, pA);
    if (!DynOn(d)) return;
    std::vector<uint8_t> tmp(8 + bc * sizeof(*pB) + ac * sizeof(*pA));
    memcpy(tmp.data(), &bc, 4);
    memcpy(tmp.data() + 4, &ac, 4);
    uint8_t* w = tmp.data() + 8;
    for (uint32_t i = 0; i < bc; ++i) { VkVertexInputBindingDescription2EXT b = pB[i]; b.pNext = nullptr; memcpy(w, &b, sizeof(b)); w += sizeof(b); }
    for (uint32_t i = 0; i < ac; ++i) { VkVertexInputAttributeDescription2EXT a = pA[i]; a.pNext = nullptr; memcpy(w, &a, sizeof(a)); w += sizeof(a); }
    TrackDyn(State(d, cb), Dyn::VertexInput, 0, tmp.data(), tmp.size());
}

static void IssueDyn(DeviceData* d, VkCommandBuffer cb, Dyn fn, uint32_t key, const uint8_t* p, uint32_t n)
{
    uint32_t v = 0;
    if (n >= 4) memcpy(&v, p, 4);
    switch (fn)
    {
#define X(Fn, Name, T) case Dyn::Name: if (d->Fn) d->Fn(cb, (T)v); break;
    WIZ3D_DYN_SCALARS(X)
#undef X
#define X(Fn, Name, T) case Dyn::Name: if (d->Fn) d->Fn(cb, key, n / (uint32_t)sizeof(T), (const T*)p); break;
    WIZ3D_DYN_RANGES(X)
#undef X
#define X(Fn, Name, T) case Dyn::Name: if (d->Fn) d->Fn(cb, n / (uint32_t)sizeof(T), (const T*)p); break;
    WIZ3D_DYN_COUNTED(X)
#undef X
    case Dyn::StencilOp:
    {
        uint32_t o[4] = {};
        memcpy(o, p, n < sizeof(o) ? n : sizeof(o));
        if (d->CmdSetStencilOp) d->CmdSetStencilOp(cb, key, (VkStencilOp)o[0], (VkStencilOp)o[1], (VkStencilOp)o[2], (VkCompareOp)o[3]);
        break;
    }
    case Dyn::SampleMask:
    {
        VkSampleMask m[2] = {};
        if (n > 4) memcpy(m, p + 4, n - 4 < sizeof(m) ? n - 4 : sizeof(m));
        if (d->CmdSetSampleMaskEXT) d->CmdSetSampleMaskEXT(cb, (VkSampleCountFlagBits)v, m);
        break;
    }
    case Dyn::ExtraPrimitiveOverestimationSize:
    {
        float f = 0.f;
        if (n >= 4) memcpy(&f, p, 4);
        if (d->CmdSetExtraPrimitiveOverestimationSizeEXT) d->CmdSetExtraPrimitiveOverestimationSizeEXT(cb, f);
        break;
    }
    case Dyn::VertexInput:
    {
        // Copied out: the record blob is unaligned and the driver reads whole structs.
        uint32_t bc = 0, ac = 0;
        if (n < 8) break;
        memcpy(&bc, p, 4); memcpy(&ac, p + 4, 4);
        thread_local std::vector<VkVertexInputBindingDescription2EXT> b;
        thread_local std::vector<VkVertexInputAttributeDescription2EXT> a;
        b.resize(bc); a.resize(ac);
        if (bc) memcpy(b.data(), p + 8, bc * sizeof(b[0]));
        if (ac) memcpy(a.data(), p + 8 + bc * sizeof(b[0]), ac * sizeof(a[0]));
        if (d->CmdSetVertexInputEXT) d->CmdSetVertexInputEXT(cb, bc, b.data(), ac, a.data());
        break;
    }
    }
}

// Draws
static VKAPI_ATTR void VKAPI_CALL Hook_CmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    // What the game issues, against statReplayDraws: the ratio says whether the
    // right eye is missing its draws or losing their result later.
    ++d->statGameDraws;
    if (s->stereoPass) ++d->statGameDrawsInStereoPass;
    PerfScope pt(d->perf.draws);
    // The first full-screen passes positioned through a uniform matrix are logged with their block, once a camera
    // has been learned (before that every answer is the same).
    if (vc <= 6 && ic == 1 && !s->replaying && d->constants && s->gfx.pipeline &&
        d->cameraActive.load(std::memory_order_relaxed))
    {
        static std::atomic<int> s_log{16};
        if (s_log.load(std::memory_order_relaxed) > 0)
        {
            bool usesMatrix = false; uint32_t set = 0, bind = 0, off = 0;
            bool camera = false, sibling = false, found = false, haveMat = false;
            VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize bo = 0, br = 0;
            float mat[16] = {};
            // The whole range, so LearnCamera's verdict on this block can be reproduced from the log.
            float blk[128] = {}; size_t blkFloats = 0;
            {
                std::lock_guard<std::shared_mutex> g(d->lock);
                auto it = d->pipelines.find(s->gfx.pipeline);
                if (it != d->pipelines.end())
                {
                    usesMatrix = it->second.posUsesMatrix;
                    set = it->second.posSet; bind = it->second.posBinding; off = it->second.posOffset;
                }
                // The block behind that binding, and whether it is a learned camera key we patch.
                if (usesMatrix && set < 32 && s->gfx.sets[set].set)
                {
                    const DescriptorSetInfo* sit = d->sets.Find(s->gfx.sets[set].set);
                    if (sit)
                    {
                        camera = ConstantsRangeIsStereo(d, sit->layout, bind);
                        for (const UboRef& u : sit->ubos)
                            if (u.binding == bind)
                            {
                                found = true; buf = u.buffer; bo = u.offset; br = u.range;
                                sibling = BufferSibling(d, u.buffer, nullptr);
                                // The matrix itself: a perspective row (m[11] == +-1) means w is view depth.
                                auto bit = d->buffers.find(u.buffer);
                                if (bit == d->buffers.end()) break;
                                auto mit = d->memory.find(bit->second.memory);
                                if (mit == d->memory.end() || !mit->second.mapped) break;
                                const VkDeviceSize memOff = bit->second.offset + u.offset;
                                if (memOff < mit->second.mapOff || memOff + off + 64 > mit->second.mapOff + mit->second.mapSize) break;
                                memcpy(mat, (const unsigned char*)mit->second.mapped + (memOff - mit->second.mapOff) + off, 64);
                                haveMat = true;
                                VkDeviceSize take = br < sizeof(blk) ? br : sizeof(blk);
                                if (memOff + take <= mit->second.mapOff + mit->second.mapSize)
                                {
                                    memcpy(blk, (const unsigned char*)mit->second.mapped + (memOff - mit->second.mapOff), (size_t)take);
                                    blkFloats = (size_t)take / 4;
                                }
                                break;
                            }
                    }
                }
            }
            if (s_log.fetch_sub(1, std::memory_order_relaxed) > 0)
            {
                Log("[fs] full-screen draw: %u verts, pipeline %p, matrix=%d (s%u:b%u+%u)"
                    " camera=%d sibling=%d shifted=%d found=%d off=%llu range=%llu\n",
                    vc, (void*)s->gfx.pipeline, (int)usesMatrix, set, bind, off,
                    (int)camera, (int)sibling, (int)s->gfxShifted, (int)found,
                    (unsigned long long)bo, (unsigned long long)br);
                if (haveMat)
                    Log("[fs]   m = [%.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f]\n",
                        mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7],
                        mat[8], mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15]);
                static std::atomic<int> s_blk{3};
                if (blkFloats && s_blk.fetch_sub(1, std::memory_order_relaxed) > 0)
                    for (size_t r2 = 0; r2 + 4 <= blkFloats; r2 += 4)
                        Log("[fs]   r%03zu %12.5f %12.5f %12.5f %12.5f\n",
                            r2 / 4, blk[r2], blk[r2 + 1], blk[r2 + 2], blk[r2 + 3]);
            }
        }
    }
    const bool screen = KindOfDraw(s->gfxWorldQuad, s->gfxByConstants, vc, ic) != DrawKind::World;
    if (screen) ClaimFullScreenSets(d, s);
    ClaimCameraSet(d, s, screen);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDraw(cb, vc, ic, fv, fi); pt.resume();
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::Draw); r.u[0] = vc; r.u[1] = ic; r.u[2] = fv; r.u[3] = fi; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo, uint32_t fInst)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    ++d->statGameDraws;
    if (s->stereoPass) ++d->statGameDrawsInStereoPass;
    PerfScope pt(d->perf.draws);
    const bool screen = KindOfDraw(s->gfxWorldQuad, s->gfxByConstants, ic, inst) != DrawKind::World;
    if (screen) ClaimFullScreenSets(d, s);
    ClaimCameraSet(d, s, screen);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDrawIndexed(cb, ic, inst, fi, vo, fInst); pt.resume();
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::DrawIndexed); r.u[0] = ic; r.u[1] = inst; r.u[2] = fi; r.u[3] = (uint32_t)vo; r.u[4] = fInst; }
}

// VK_EXT_multi_draw: one call per range, in both eyes, so the replay records each (gl_DrawID differs; hosts using
// this extension for multi-range draws do not read it).
static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawMultiEXT(VkCommandBuffer cb, uint32_t n, const VkMultiDrawInfoEXT* p, uint32_t inst, uint32_t firstInst, uint32_t stride)
{
    for (uint32_t i = 0; i < n; ++i)
    {
        const VkMultiDrawInfoEXT& e = *(const VkMultiDrawInfoEXT*)((const uint8_t*)p + i * stride);
        Hook_CmdDraw(cb, e.vertexCount, inst, e.firstVertex, firstInst);
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawMultiIndexedEXT(VkCommandBuffer cb, uint32_t n, const VkMultiDrawIndexedInfoEXT* p, uint32_t inst, uint32_t firstInst, uint32_t stride, const int32_t* pVertexOffset)
{
    for (uint32_t i = 0; i < n; ++i)
    {
        const VkMultiDrawIndexedInfoEXT& e = *(const VkMultiDrawIndexedInfoEXT*)((const uint8_t*)p + i * stride);
        Hook_CmdDrawIndexed(cb, e.indexCount, inst, e.firstIndex, pVertexOffset ? *pVertexOffset : e.vertexOffset, firstInst);
    }
}

// Conditional rendering opened and closed inside a pass: the right eye takes the same predicate.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginConditionalRenderingEXT(VkCommandBuffer cb, const VkConditionalRenderingBeginInfoEXT* p)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdBeginConditionalRenderingEXT(cb, p);
    if (s->replaying) return;
    ++d->statCondBegins;
    if (Recording(s)) { ++d->statCondInPass; Rec& r = Push(s, Op::CondBegin); r.h[0] = (uint64_t)p->buffer; r.h[1] = p->offset; r.u[0] = p->flags; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdEndConditionalRenderingEXT(VkCommandBuffer cb)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdEndConditionalRenderingEXT(cb);
    if (s->replaying) return;
    if (Recording(s)) Push(s, Op::CondEnd);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, uint32_t count, uint32_t stride)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    PerfScope pt(d->perf.draws);
    ClaimCameraSet(d, s, false);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDrawIndirect(cb, buf, off, count, stride); pt.resume();
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::DrawIndirect); r.h[0] = (uint64_t)buf; r.h[1] = off; r.u[0] = count; r.u[1] = stride; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, uint32_t count, uint32_t stride)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    PerfScope pt(d->perf.draws);
    ClaimCameraSet(d, s, false);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDrawIndexedIndirect(cb, buf, off, count, stride); pt.resume();
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::DrawIndexedIndirect); r.h[0] = (uint64_t)buf; r.h[1] = off; r.u[0] = count; r.u[1] = stride; }
}

// VK_KHR_draw_indirect_count: recorded like any draw.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawIndirectCount(
    VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkBuffer cntBuf, VkDeviceSize cntOff, uint32_t maxDraws, uint32_t stride)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    PerfScope pt(d->perf.draws);
    ClaimCameraSet(d, s, false);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDrawIndirectCount(cb, buf, off, cntBuf, cntOff, maxDraws, stride); pt.resume();
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::DrawIndirectCount);
        r.h[0] = (uint64_t)buf; r.h[1] = off; r.h[2] = (uint64_t)cntBuf; r.h[3] = cntOff;
        r.u[0] = maxDraws; r.u[1] = stride;
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawIndexedIndirectCount(
    VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkBuffer cntBuf, VkDeviceSize cntOff, uint32_t maxDraws, uint32_t stride)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    PerfScope pt(d->perf.draws);
    ClaimCameraSet(d, s, false);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDrawIndexedIndirectCount(cb, buf, off, cntBuf, cntOff, maxDraws, stride); pt.resume();
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::DrawIndexedIndirectCount);
        r.h[0] = (uint64_t)buf; r.h[1] = off; r.h[2] = (uint64_t)cntBuf; r.h[3] = cntOff;
        r.u[0] = maxDraws; r.u[1] = stride;
    }
}

// VK_EXT_mesh_shader: with VKMeshDraws the draws bind the eye's set and record like any draw; without it they are
// only counted, and the right eye has none of them.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawMeshTasksEXT(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z)
{
    DeviceData* d = FindDevice(cb);
    ++d->statMeshDraws;
    if (!Vk().VKMeshDraws) { d->CmdDrawMeshTasksEXT(cb, x, y, z); return; }
    CmdState* s = State(d, cb);
    PerfScope pt(d->perf.draws);
    ClaimCameraSet(d, s, false);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDrawMeshTasksEXT(cb, x, y, z); pt.resume();
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::DrawMeshTasks); r.u[0] = x; r.u[1] = y; r.u[2] = z; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, uint32_t count, uint32_t stride)
{
    DeviceData* d = FindDevice(cb);
    ++d->statMeshDraws;
    if (!Vk().VKMeshDraws) { d->CmdDrawMeshTasksIndirectEXT(cb, buf, off, count, stride); return; }
    CmdState* s = State(d, cb);
    PerfScope pt(d->perf.draws);
    ClaimCameraSet(d, s, false);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDrawMeshTasksIndirectEXT(cb, buf, off, count, stride); pt.resume();
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::DrawMeshTasksIndirect); r.h[0] = (uint64_t)buf; r.h[1] = off; r.u[0] = count; r.u[1] = stride; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDrawMeshTasksIndirectCountEXT(
    VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkBuffer cntBuf, VkDeviceSize cntOff, uint32_t maxDraws, uint32_t stride)
{
    DeviceData* d = FindDevice(cb);
    ++d->statMeshDraws;
    if (!Vk().VKMeshDraws) { d->CmdDrawMeshTasksIndirectCountEXT(cb, buf, off, cntBuf, cntOff, maxDraws, stride); return; }
    CmdState* s = State(d, cb);
    PerfScope pt(d->perf.draws);
    ClaimCameraSet(d, s, false);
    EnsureEyeSet(d, s);
    pt.pause(); d->CmdDrawMeshTasksIndirectCountEXT(cb, buf, off, cntBuf, cntOff, maxDraws, stride); pt.resume();
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::DrawMeshTasksIndirectCount);
        r.h[0] = (uint64_t)buf; r.h[1] = off; r.h[2] = (uint64_t)cntBuf; r.h[3] = cntOff;
        r.u[0] = maxDraws; r.u[1] = stride;
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdClearAttachments(VkCommandBuffer cb, uint32_t ac, const VkClearAttachment* pA, uint32_t rc, const VkClearRect* pR)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdClearAttachments(cb, ac, pA, rc, pR);
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::ClearAttachments);
        r.u[0] = ac; r.u[1] = rc;
        std::vector<uint8_t> tmp(ac * sizeof(VkClearAttachment) + rc * sizeof(VkClearRect));
        memcpy(tmp.data(), pA, ac * sizeof(VkClearAttachment));
        memcpy(tmp.data() + ac * sizeof(VkClearAttachment), pR, rc * sizeof(VkClearRect));
        Payload(s, r, tmp.data(), tmp.size());
    }
}

// Dispatch: doubled in place when a bound compute set has a shadow.

// `force`: the caller decided this dispatch runs per eye, whatever it writes (KytyTiles.cpp).
static bool BindComputeShadows(DeviceData* d, CmdState* s, bool shadows, bool force = false)
{
    uint32_t gameSets = 0;
    if (!s->compLayout) return false;
    LayoutPatched(d, s, s->compLayout, &gameSets);
    bool any = false;
    std::vector<uint32_t> pushSets;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        // A dispatch is doubled only when it touches a doubled image, and not when it writes a storage buffer with
        // no sibling (shared state, which a replay would step twice); read-only bindings never block.
        const std::vector<uint64_t>* writes = nullptr;
        if (!force)
        {
            auto pit = d->pipelines.find(s->compPipeline);
            if (pit != d->pipelines.end() && !pit->second.storageWrites.empty()) writes = &pit->second.storageWrites;
        }
        bool images = force;
        for (uint32_t i = 0; i < gameSets && i < 32; ++i)
        {
            const DescriptorSetInfo* it = s->compSets[i].set ? d->sets.Find(s->compSets[i].set) : nullptr;
            if (it)
            {
                // stereoStorage, not stereoImages: a read-only-sibling dispatch run
                // twice cross-coupled the eyes' auto-exposure (white/black).
                if (it->stereoStorage) images = true;
                if (writes)
                    for (const UboRef& u : it->ubos)
                    {
                        if (u.uniform) continue;
                        const uint64_t key = ((uint64_t)i << 32) | u.binding;
                        bool written = false;
                        for (uint64_t w : *writes) if (w == key) { written = true; break; }
                        if (!written) continue;
                        auto bit = d->buffers.find(u.buffer);
                        if (bit == d->buffers.end() || !bit->second.sibling)
                        {
                            // Request one, so later frames admit this dispatch.
                            if (bit != d->buffers.end())
                            {
                                bool queued = false;
                                for (VkBuffer q : d->pendingBufferSiblings) if (q == u.buffer) { queued = true; break; }
                                if (!queued && d->pendingBufferSiblings.size() < 256) d->pendingBufferSiblings.push_back(u.buffer);
                            }
                            return false;
                        }
                    }
            }
            if (!s->compPushes[i].empty() && s->compPushLayout[i] == s->compLayout && PushesHaveStereo(d, s->compPushes[i], true)) images = true;
        }
        if (!images) return false;
        for (uint32_t i = 0; i < gameSets && i < 32; ++i)
        {
            const SetBind& b = s->compSets[i];
            if (b.set)
            {
                VkDescriptorSet shadow = ShadowSetFor(d, b.set);
                if (!shadow) continue;
                any = true;
                VkDescriptorSet set = shadows ? shadow : b.set;
                d->CmdBindDescriptorSets(s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->compLayout, i, 1, &set,
                                         (uint32_t)b.dyn.size(), b.dyn.empty() ? nullptr : b.dyn.data());
            }
            else if (!s->compPushes[i].empty() && s->compPushLayout[i] == s->compLayout && PushesHaveStereo(d, s->compPushes[i]))
            {
                any = true;
                pushSets.push_back(i);
            }
        }
    }
    for (uint32_t i : pushSets)
        IssuePushes(d, s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->compLayout, i, s->compPushes[i], shadows);
    return any;
}

// Only sets pushed with the bound pipeline's layout: re-pushing one recorded for an
// earlier, incompatible layout would disturb the current pipeline's bindings.
static void RepushCompute(DeviceData* d, CmdState* s, bool right)
{
    for (uint32_t i = 0; i < 32; ++i)
        if (!s->compPushes[i].empty() && s->compPushLayout[i] == s->compLayout)
            IssuePushes(d, s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->compLayout, i, s->compPushes[i], right);
}

// VKBufferTwins: a dispatch binding any right-eye data runs again with every binding swapped to its right-eye
// copy; a small written binding with no twin goes to a seeded scratch slot, a large one stays bound.
template <class F> static bool RightCompute(DeviceData* d, CmdState* s, F dispatch)
{
    PushScan scan;
    bool setsStereo = false;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        const std::vector<uint64_t>* writes = nullptr;
        auto pit = d->pipelines.find(s->compPipeline);
        if (pit != d->pipelines.end() && !pit->second.storageWrites.empty()) writes = &pit->second.storageWrites;
        for (uint32_t i = 0; i < 32; ++i)
        {
            if (!s->compPushes[i].empty() && s->compPushLayout[i] == s->compLayout) ScanPush(d, s->compPushes[i], i, writes, scan);
            if (s->compSets[i].set)
            {
                const DescriptorSetInfo* it = d->sets.Find(s->compSets[i].set);
                if (it && it->stereoImages) setsStereo = true;
            }
        }
    }
    if (!scan.stereo && !setsStereo) return false;
    // The ranges this right run writes into twins: a later indirect dispatch with its args there takes the twin's.
    if (!scan.twinnedWrites.empty())
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (const VkDescriptorBufferInfo& bi : scan.twinnedWrites)
        {
            auto it = d->buffers.find(bi.buffer);
            const VkDeviceSize size = it != d->buffers.end() ? it->second.size : 0;
            const VkDeviceSize n = bi.offset >= size ? 0 : (bi.range == VK_WHOLE_SIZE || bi.offset + bi.range > size) ? size - bi.offset : bi.range;
            if (!n) continue;
            bool known = false;
            for (DeviceData::RightWrite& w : d->rightWrites)
                if (w.buffer == bi.buffer && w.offset == bi.offset && w.size == n) { w.frame = d->frames; known = true; break; }
            if (known) continue;
            if (d->rightWrites.size() >= 256) d->rightWrites.erase(d->rightWrites.begin());
            d->rightWrites.push_back({ bi.buffer, bi.offset, n, d->frames });
        }
    }
    std::vector<BufRedirect> redirects;
    std::vector<VkBufferCopy> seeds;
    for (const VkDescriptorBufferInfo& bi : scan.untwinnedWrites)
    {
        VkDeviceSize size = 0;
        {
            std::lock_guard<std::shared_mutex> g(d->lock);
            auto it = d->buffers.find(bi.buffer);
            if (it != d->buffers.end()) size = it->second.size;
        }
        const VkDeviceSize n = bi.offset >= size ? 0 : (bi.range == VK_WHOLE_SIZE || bi.offset + bi.range > size) ? size - bi.offset : bi.range;
        VkDeviceSize so = 0;
        const VkBuffer scratch = (n && n <= 64 * 1024) ? ScratchSlot(d, n, &so) : VK_NULL_HANDLE;
        if (!scratch)
        {
            std::lock_guard<std::shared_mutex> g(d->lock);
            ++d->statUntwinnedWrites;
            continue;
        }
        redirects.push_back({ bi.buffer, bi.offset, scratch, so });
        seeds.push_back({ bi.offset, so, n });
    }
    if (scan.untwinnedTexelWrite)
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        ++d->statUntwinnedWrites;
    }
    if (!seeds.empty())
    {
        VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT };
        d->CmdPipelineBarrier(s->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        for (size_t i = 0; i < seeds.size(); ++i) d->CmdCopyBuffer(s->cb, redirects[i].from, redirects[i].to, 1, &seeds[i]);
        VkMemoryBarrier mb2 = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
        d->CmdPipelineBarrier(s->cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);
    }
    const bool sets = BindComputeShadows(d, s, true);
    t_redirects.swap(redirects);
    RepushCompute(d, s, true);
    t_redirects.clear();
    dispatch();
    if (sets) BindComputeShadows(d, s, false);
    RepushCompute(d, s, false);
    std::lock_guard<std::shared_mutex> g(d->lock);
    ++d->statReplayDispatch;
    ++d->statRightDispatches;
    return true;
}

// VKComputeRemap: a remapped pipeline reads its run's camera from set 8 (sep = 0 for the left run).
static void BindRemapSet(DeviceData* d, CmdState* s, bool right)
{
    if (!s->compRemap || !s->compLayout) return;
    const VkDescriptorSet set = right ? d->stereoSetRight : d->stereoSetLeft;
    if (!set) return;
    d->CmdBindDescriptorSets(s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->compLayout, kStereoSet, 1, &set, 0, nullptr);
    if (right) ++d->statRemapRight;
}

// KytyTiles.cpp: a right run whose constant block names tile lists pushes a copy naming the right eye's;
// TilePop restores the left's.
static bool TilePush(DeviceData* d, CmdState* s)
{
    if (!d->tiles || !s->compPcSize || s->compPcLayout != s->compLayout) return false;
    uint32_t pc[32];
    memcpy(pc, s->compPc, s->compPcSize);
    if (!TileRightBlock(d, s->compPipeline, s->compCrc, pc, s->compPcSize / 4)) return false;
    d->CmdPushConstants(s->cb, s->compPcLayout, s->compPcStages, 0, s->compPcSize, pc);
    return true;
}

static void TilePop(DeviceData* d, CmdState* s)
{
    d->CmdPushConstants(s->cb, s->compPcLayout, s->compPcStages, 0, s->compPcSize, s->compPc);
}

// The host pushes before every dispatch that has a block; one that pushed nothing must not take the last one's.
struct TilePcDone { CmdState* s; ~TilePcDone() { s->compPcSize = 0; } };

// The tile classifier writes no image, so only this runs it for the right eye, and only with a block to re-point;
// its barrier orders the lists for the next step and the indirect args, as the game's name the left's memory only.
static bool RightTileChain(DeviceData* d, CmdState* s, uint32_t x, uint32_t y, uint32_t z)
{
    if (!TilePush(d, s)) return false;
    const bool sets = BindComputeShadows(d, s, true, true);
    BindRemapSet(d, s, true);
    d->CmdDispatch(s->cb, x, y, z);
    if (sets) BindComputeShadows(d, s, false, true);
    TilePop(d, s);
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT };
    d->CmdPipelineBarrier(s->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                          0, 1, &mb, 0, nullptr, 0, nullptr);
    std::lock_guard<std::shared_mutex> g(d->lock);
    ++d->statReplayDispatch;
    return true;
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    TilePcDone pcDone{ s };
    BindRemapSet(d, s, false);
    d->CmdDispatch(cb, x, y, z);
    if (Traits().tripKernels && s->tripBuffer)
    {
        // Two kernels (a depth gather, then a swap) would need the first one's binds back: that trip stays the left's.
        if (s->tripDispatches.empty() || s->tripPipeline == s->compPipeline) { s->tripPipeline = s->compPipeline; s->tripDispatches.push_back({ x, y, z }); }
        else
        {
            static std::atomic<int> s_kernels{4};
            if (s_kernels.fetch_sub(1) > 0) Log("[trip] more than one kernel between the copies: the right image keeps the left's bytes\n");
            s->tripBuffer = VK_NULL_HANDLE;
        }
    }
    if (!d->stereoEnabled || s->inPass || s->foreign || t_foreignCalls) return;
    if (TileChainShader(d, s->compCrc))
    {
        RightTileChain(d, s, x, y, z);
        return;
    }
    if (Vk().VKBufferTwins)
    {
        RightCompute(d, s, [&] {
            BindRemapSet(d, s, true);
            d->CmdDispatch(cb, x, y, z);
        });
        return;
    }
    if (BindComputeShadows(d, s, true))
    {
        BindRemapSet(d, s, true);
        const bool tile = TilePush(d, s);
        d->CmdDispatch(cb, x, y, z);
        if (tile) TilePop(d, s);
        BindComputeShadows(d, s, false);
        ++d->statReplayDispatch;
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    TilePcDone pcDone{ s };
    BindRemapSet(d, s, false);
    d->CmdDispatchIndirect(cb, buf, off);
    if (!d->stereoEnabled || s->inPass || s->foreign || t_foreignCalls) return;
    if (Vk().VKBufferTwins)
    {
        // Right-run args come from the args buffer's twin; at VKIndirectFromLeft=2 with camera windows, the left's
        // checked counts, unless a right run wrote these args there in the last two frames.
        VkBuffer args = buf;
        {
            std::lock_guard<std::shared_mutex> g(d->lock);
            if (const VkBuffer t = BufferTwin(d, buf))
            {
                bool own = Vk().VKIndirectFromLeft < 2 || !Vk().VKFsCameraWindows;
                for (size_t k = 0; !own && k < d->rightWrites.size(); ++k)
                {
                    const DeviceData::RightWrite& w = d->rightWrites[k];
                    own = w.buffer == buf && off >= w.offset && off < w.offset + w.size && w.frame + 2 >= d->frames;
                }
                if (own) { args = t; ++d->statTwinArgs; }
            }
        }
        RightCompute(d, s, [&] {
            BindRemapSet(d, s, true);
            d->CmdDispatchIndirect(cb, args, off);
        });
        return;
    }
    if (BindComputeShadows(d, s, true))
    {
        BindRemapSet(d, s, true);
        const bool tile = TilePush(d, s);
        VkBuffer args = buf;
        VkDeviceSize argsOff = off;
        TileRightArgs(d, buf, off, &args, &argsOff);
        d->CmdDispatchIndirect(cb, args, argsOff);
        if (tile) TilePop(d, s);
        BindComputeShadows(d, s, false);
        ++d->statReplayDispatch;
    }
}

// Transfers and clears outside passes: mirrored to siblings.

// Buffers the game fills on the GPU bypass the mapped memory the constants path reads: the update is repeated into
// their sibling and twin.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize off, VkDeviceSize size, const void* pData)
{
    DeviceData* d = FindDevice(cb);
    d->CmdUpdateBuffer(cb, dst, off, size, pData);
    ++d->statUpdateBuf; d->statUpdateBufBytes += (unsigned)size;
    VkBuffer sib = VK_NULL_HANDLE, twin = VK_NULL_HANDLE;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        BufferSibling(d, dst, &sib);
        if (Vk().VKBufferTwins) twin = BufferTwin(d, dst);
    }
    if (sib) d->CmdUpdateBuffer(cb, sib, off, size, pData);
    if (twin) d->CmdUpdateBuffer(cb, twin, off, size, pData);
}

// Lock NOT held. The right eye's copy of a buffer (VKBufferTwins), or null.
static VkBuffer TwinOf(DeviceData* d, VkBuffer b)
{
    if (!Vk().VKBufferTwins) return VK_NULL_HANDLE;
    std::shared_lock<std::shared_mutex> g(d->lock);
    return BufferTwin(d, b);
}

// Every transfer into a twinned buffer is repeated into its twin, from the source's
// twin when it has one: the right eye's guest memory follows the left's step for step.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t n, const VkBufferCopy* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdCopyBuffer(cb, src, dst, n, p);
    // New bytes in the trip buffer: not a trip any more.
    if (Traits().tripKernels) { CmdState* s = State(d, cb); if (s->tripBuffer == dst) s->tripBuffer = VK_NULL_HANDLE; }
    VkBuffer sib = VK_NULL_HANDLE;
    { std::shared_lock<std::shared_mutex> g(d->lock); BufferSibling(d, dst, &sib); }
    if (sib) d->CmdCopyBuffer(cb, src, sib, n, p);
    if (const VkBuffer dt = TwinOf(d, dst))
    {
        const VkBuffer st = TwinOf(d, src);
        d->CmdCopyBuffer(cb, st ? st : src, dt, n, p);
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyBuffer2(VkCommandBuffer cb, const VkCopyBufferInfo2* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdCopyBuffer2(cb, p);
    const VkBuffer dt = TwinOf(d, p->dstBuffer);
    if (!dt) return;
    const VkBuffer st = TwinOf(d, p->srcBuffer);
    VkCopyBufferInfo2 c = *p;
    c.srcBuffer = st ? st : p->srcBuffer;
    c.dstBuffer = dt;
    d->CmdCopyBuffer2(cb, &c);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdFillBuffer(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize off, VkDeviceSize size, uint32_t v)
{
    DeviceData* d = FindDevice(cb);
    d->CmdFillBuffer(cb, dst, off, size, v);
    if (const VkBuffer dt = TwinOf(d, dst)) d->CmdFillBuffer(cb, dt, off, size, v);
}

// Query results copied into guest memory: under VKPushDeltas the twin takes the left's counts too, as no query runs in
// the right replay.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyQueryPoolResults(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t count, VkBuffer dst, VkDeviceSize off, VkDeviceSize stride, VkQueryResultFlags flags)
{
    DeviceData* d = FindDevice(cb);
    d->CmdCopyQueryPoolResults(cb, pool, first, count, dst, off, stride, flags);
    if (Vk().VKOcclusionBothEyes)
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        if (d->occTwins.count(pool)) d->statOccGpuCopies += count;
    }
    if (!Vk().VKPushDeltas) return;
    if (const VkBuffer dt = TwinOf(d, dst)) d->CmdCopyQueryPoolResults(cb, pool, first, count, dt, off, stride, flags);
}

// VKOcclusionBothEyes: the right replay counts into a twin pool and the read adds it, so an object either eye sees
// passes a host's occlusion culling and one both miss stays culled.
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo* ci, const VkAllocationCallbacks* alloc, VkQueryPool* pool)
{
    DeviceData* d = FindDevice(device);
    const VkResult r = d->CreateQueryPool(device, ci, alloc, pool);
    if (r != VK_SUCCESS || !Vk().VKOcclusionBothEyes || ci->queryType != VK_QUERY_TYPE_OCCLUSION) return r;
    VkQueryPoolCreateInfo tci = *ci;
    tci.pNext = nullptr;
    VkQueryPool twin = VK_NULL_HANDLE;
    if (d->CreateQueryPool(device, &tci, alloc, &twin) != VK_SUCCESS) { Log("[occlusion] twin of pool %p (%u queries) NOT created\n", (void*)*pool, ci->queryCount); return r; }
    std::lock_guard<std::shared_mutex> g(d->lock);
    DeviceData::OccTwin& t = d->occTwins[*pool];
    t.twin = twin; t.count = ci->queryCount;
    t.begun.reset(new std::atomic<uint8_t>[ci->queryCount]);
    for (uint32_t i = 0; i < ci->queryCount; ++i) t.begun[i].store(0, std::memory_order_relaxed);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyQueryPool(VkDevice device, VkQueryPool pool, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = FindDevice(device);
    VkQueryPool twin = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->occTwins.find(pool);
        if (it != d->occTwins.end()) { twin = it->second.twin; d->occTwins.erase(it); }
    }
    if (twin) d->DestroyQueryPool(device, twin, alloc);
    d->DestroyQueryPool(device, pool, alloc);
}

static VKAPI_ATTR void VKAPI_CALL Hook_ResetQueryPool(VkDevice device, VkQueryPool pool, uint32_t first, uint32_t count)
{
    DeviceData* d = FindDevice(device);
    if (!d->ResetQueryPool) return;
    d->ResetQueryPool(device, pool, first, count);
    VkQueryPool twin = VK_NULL_HANDLE;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto it = d->occTwins.find(pool);
        if (it != d->occTwins.end() && first < it->second.count)
        {
            twin = it->second.twin;
            count = (std::min)(count, it->second.count - first);
            for (uint32_t i = 0; i < count; ++i) it->second.begun[first + i].store(0, std::memory_order_relaxed);
        }
    }
    if (twin) d->ResetQueryPool(device, twin, first, count);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdResetQueryPool(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t count)
{
    DeviceData* d = FindDevice(cb);
    d->CmdResetQueryPool(cb, pool, first, count);
    VkQueryPool twin = VK_NULL_HANDLE;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto it = d->occTwins.find(pool);
        if (it != d->occTwins.end() && first < it->second.count)
        {
            twin = it->second.twin;
            count = (std::min)(count, it->second.count - first);
            for (uint32_t i = 0; i < count; ++i) it->second.begun[first + i].store(0, std::memory_order_relaxed);
        }
    }
    if (twin) d->CmdResetQueryPool(cb, twin, first, count);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query, VkQueryControlFlags flags)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdBeginQuery(cb, pool, query, flags);
    if (!Vk().VKOcclusionBothEyes || s->replaying || !Recording(s)) return;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        if (!d->occTwins.count(pool)) return;
    }
    Rec& r = Push(s, Op::QueryBegin);
    r.h[0] = (uint64_t)pool; r.u[0] = query; r.u[1] = flags;
    ++d->statOccRecorded;
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdEndQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdEndQuery(cb, pool, query);
    if (!Vk().VKOcclusionBothEyes || s->replaying || !Recording(s)) return;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        if (!d->occTwins.count(pool)) return;
    }
    Rec& r = Push(s, Op::QueryEnd);
    r.h[0] = (uint64_t)pool; r.u[0] = query;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_GetQueryPoolResults(VkDevice device, VkQueryPool pool, uint32_t first, uint32_t count, size_t size, void* data, VkDeviceSize stride, VkQueryResultFlags flags)
{
    DeviceData* d = FindDevice(device);
    const VkResult r = d->GetQueryPoolResults(device, pool, first, count, size, data, stride, flags);
    if (!Vk().VKOcclusionBothEyes || !data || (r != VK_SUCCESS && r != VK_NOT_READY)) return r;
    const bool avail = (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0, wide = (flags & VK_QUERY_RESULT_64_BIT) != 0;
    if (r == VK_NOT_READY && !avail) return r;   // which values were written is unknown
    PerfScope pt(d->occReadTicks);
    VkQueryPool twin = VK_NULL_HANDLE;
    std::vector<uint8_t> begun;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto it = d->occTwins.find(pool);
        if (it == d->occTwins.end() || first >= it->second.count) return r;
        twin = it->second.twin;
        count = (std::min)(count, it->second.count - first);
        begun.resize(count);
        for (uint32_t i = 0; i < count; ++i) begun[i] = it->second.begun[first + i].load(std::memory_order_relaxed);
    }
    // The twin's slots in one read, availability per slot: a slot the right eye never began adds nothing.
    std::vector<uint64_t> tv((size_t)count * 2);
    const VkResult tr = d->GetQueryPoolResults(device, twin, first, count, tv.size() * 8, tv.data(), 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (tr != VK_SUCCESS && tr != VK_NOT_READY) return r;
    const size_t word = wide ? 8 : 4;
    unsigned reads = 0, rightOnly = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (!begun[i]) continue;
        uint8_t* p = (uint8_t*)data + (size_t)i * stride;
        if ((size_t)i * stride + word * (avail ? 2 : 1) > size) break;
        if (avail) { uint64_t a = 0; memcpy(&a, p + word, word); if (!a) continue; }
        uint64_t rv = tv[(size_t)i * 2];
        // Begun in a submitted pass, so it completes: wait for this one alone.
        if (!tv[(size_t)i * 2 + 1])
        {
            ++d->statOccWaits;
            if (d->GetQueryPoolResults(device, twin, first + i, 1, 8, &rv, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) continue;
        }
        ++reads;
        if (!rv) continue;
        uint64_t lv = 0;
        memcpy(&lv, p, word);
        if (!lv) ++rightOnly;
        lv += rv;
        if (!wide && lv > 0xFFFFFFFFull) lv = 0xFFFFFFFFull;
        memcpy(p, &lv, word);
    }
    d->statOccReads += reads;
    d->statOccRightOnly += rightOnly;
    return r;
}

// Lock held. readColorBuffers: the host's transfer helper (usage exactly transfer src+dst), which reloads and
// read-backs go through above native resolution.
static bool IsRsxHelper(const ImageInfo& i)
{
    return i.ci.usage == (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

static bool IsRsxLinearTwin(const ImageInfo& i)   // MSAA's resolve surface
{
    return i.ci.usage == (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

static bool RcbRecent(const DeviceData* d, unsigned f)
{
    return f != ~0u && d->frames - f <= 3;
}

static bool RcbHoldsRightFrame(const DeviceData* d, const ImageInfo& i)
{
    return IsRsxLinearTwin(i) || ((i.ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && RcbRecent(d, i.rightDrawnFrame));
}

// Whose read-back a reload holds (its own target's, or another's) differs per game, so it is measured on the left:
// three rows at each read-back and reload.
static bool RcbRowFormat(VkFormat f)
{
    return f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_R8G8B8A8_SRGB;
}

static DeviceData::RcbStash* RcbShape(DeviceData* d, const VkImageCreateInfo& ci)
{
    for (DeviceData::RcbStash& e : d->rcbStashes)
        if (e.w == ci.extent.width && e.h == ci.extent.height && e.fmt == ci.format) return &e;
    return nullptr;
}

// Lock held. A transfer image in device memory, to be moved to GENERAL by the first command buffer that uses it.
static VkImage RcbMakeImage(DeviceData* d, VkFormat fmt, uint32_t w, uint32_t h, VkDeviceMemory* mem)
{
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = fmt; ci.extent = { w, h, 1 }; ci.mipLevels = 1; ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage img = VK_NULL_HANDLE;
    if (d->CreateImage(d->device, &ci, nullptr, &img) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkMemoryRequirements req;
    d->GetImageMemoryRequirements(d->device, img, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->device, &mai, nullptr, mem) != VK_SUCCESS || d->BindImageMemory(d->device, img, *mem, 0) != VK_SUCCESS)
    {
        if (*mem) d->FreeMemory(d->device, *mem, nullptr);
        *mem = VK_NULL_HANDLE;
        d->DestroyImage(d->device, img, nullptr);
        return VK_NULL_HANDLE;
    }
    return img;
}

static void RcbToGeneral(DeviceData* d, VkCommandBuffer cb, VkImage img)
{
    VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = img;
    ib.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
}

static void RcbTransferBarrier(DeviceData* d, VkCommandBuffer cb)
{
    const VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

static void RcbLayout(DeviceData* d, VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    ib.srcAccessMask = ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.oldLayout = from; ib.newLayout = to;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = img;
    ib.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
}

// Lock held. The next row probe's place in host memory.
static VkBuffer RcbNextProbe(DeviceData* d, DeviceData::RcbStash& e, uint64_t read, uint64_t reloadNo, VkImage image, VkDeviceSize* off)
{
    if (!RcbRowFormat(e.fmt)) return VK_NULL_HANDLE;
    const VkDeviceSize one = 3ull * e.w * 4;
    if (!e.probeBuf)
    {
        if (e.probeFailed) return VK_NULL_HANDLE;
        e.probeFailed = true;
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = one * DeviceData::RcbStash::kProbes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (d->CreateBuffer(d->device, &bci, nullptr, &e.probeBuf) != VK_SUCCESS) { e.probeBuf = VK_NULL_HANDLE; return VK_NULL_HANDLE; }
        VkMemoryRequirements req;
        d->GetBufferMemoryRequirements(d->device, e.probeBuf, &req);
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = req.size;
        const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        // Cached if possible: uncached ReBAR reads are slow.
        mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, UINT32_MAX);
        if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, host, UINT32_MAX);
        void* map = nullptr;
        if (mai.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->device, &mai, nullptr, &e.probeMem) != VK_SUCCESS || d->BindBufferMemory(d->device, e.probeBuf, e.probeMem, 0) != VK_SUCCESS ||
            d->MapMemory(d->device, e.probeMem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS)
        {
            if (e.probeMem) d->FreeMemory(d->device, e.probeMem, nullptr);
            d->DestroyBuffer(d->device, e.probeBuf, nullptr);
            e.probeBuf = VK_NULL_HANDLE; e.probeMem = VK_NULL_HANDLE;
            return VK_NULL_HANDLE;
        }
        e.probeMap = (const uint8_t*)map;
        e.probeFailed = false;
    }
    const uint32_t i = e.probeNext++ % DeviceData::RcbStash::kProbes;
    DeviceData::RcbStash::Probe& p = e.probes[i];
    p.frame = d->frames; p.read = read; p.reloadNo = reloadNo; p.image = image; p.reload = reloadNo != 0; p.checked = false;
    *off = one * i;
    return e.probeBuf;
}

static void RcbCopyRows(DeviceData* d, VkCommandBuffer cb, VkImage img, VkImageLayout l, VkBuffer to, VkDeviceSize off, uint32_t w, uint32_t h)
{
    VkBufferImageCopy r[3] = {};
    for (uint32_t k = 0; k < 3; ++k)
    {
        r[k].bufferOffset = off + (VkDeviceSize)k * w * 4;
        r[k].imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r[k].imageOffset = { 0, (int32_t)(h * (3 + 2 * k) / 10), 0 };
        r[k].imageExtent = { w, 1, 1 };
    }
    d->CmdCopyImageToBuffer(cb, img, l, to, 3, r);
    const VkMemoryBarrier toHost = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &toHost, 0, nullptr, 0, nullptr);
}

// A read-back of a target in a shape a reload was seen in: numbered, the left's rows probed, and the right image kept
// if a verdict needs it.
static void RcbOnReadBack(DeviceData* d, VkCommandBuffer cb, VkImage src, VkImage ss, VkImageLayout sl)
{
    VkBuffer probe = VK_NULL_HANDLE;
    VkDeviceSize off = 0;
    VkImage slot = VK_NULL_HANDLE;
    bool fresh = false;
    uint32_t w = 0, h = 0;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->images.find(src);
        if (it == d->images.end() || !it->second.sibling || it->second.ci.samples != VK_SAMPLE_COUNT_1_BIT || !RcbHoldsRightFrame(d, it->second)) return;
        // A finished picture read back again is no CPU input: the next reload takes it as it is, HUD included.
        if (it->second.finished && !IsRsxLinearTwin(it->second)) return;
        DeviceData::RcbStash* e = RcbShape(d, it->second.ci);
        if (!e || e->cpuIsFilter) return;
        w = e->w; h = e->h;
        const uint64_t n = ++e->reads;
        const int k = (int)(n % DeviceData::RcbStash::kSlots);
        e->readFrom[k] = src;
        probe = RcbNextProbe(d, *e, n, 0, src, &off);
        if (e->wantSlots)
        {
            if (!e->img[k])
            {
                e->img[k] = RcbMakeImage(d, e->fmt, w, h, &e->mem[k]);
                fresh = e->img[k] != VK_NULL_HANDLE;
                if (fresh) Log("[rcb] %ux%u fmt=%d: slot %d keeps the right frame at read-backs\n", w, h, (int)e->fmt, k);
            }
            if (e->img[k]) { slot = e->img[k]; e->slotRead[k] = n; }
        }
    }
    if (probe) RcbCopyRows(d, cb, src, sl, probe, off, w, h);
    if (!slot) return;
    if (fresh) RcbToGeneral(d, cb, slot);
    else RcbTransferBarrier(d, cb);
    const VkImageCopy r = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { w, h, 1 } };
    d->CmdCopyImage(cb, ss, sl, slot, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}

struct RcbReload { VkImage slot = VK_NULL_HANDLE; VkBuffer probe = VK_NULL_HANDLE; VkDeviceSize off = 0; uint32_t w = 0, h = 0; bool dropLeft = false; };

// Lock held. A reload at native scale both eyes skip: the CPU was measured to add only edge filtering to this shape,
// and the target holds its own frame.
static bool RcbLeftKeepsOwn(DeviceData* d, VkImage dst, const VkBufferImageCopy& r)
{
    auto it = d->images.find(dst);
    if (it == d->images.end() || !it->second.sibling) return false;
    ImageInfo& i = it->second;
    DeviceData::RcbStash* e = RcbShape(d, i.ci);
    if (!e || !e->cpuIsFilter || d->frames - e->lastReload > 30) return false;
    if (r.imageOffset.x || r.imageOffset.y || r.imageExtent.width != i.ci.extent.width || r.imageExtent.height != i.ci.extent.height || r.imageSubresource.mipLevel) return false;
    if (!RcbRecent(d, i.readBackFrame) || !RcbHoldsRightFrame(d, i)) return false;
    e->lastReload = d->frames;
    i.finished = true;
    i.reloadFrame = d->frames;
    return true;
}

// Lock held. What a reload into `dst` needs: unless measured to hold another target's read-back, the right image
// already is that frame. An unseen shape is learned here.
static RcbReload RcbForReload(DeviceData* d, VkImage dst)
{
    RcbReload out;
    auto it = d->images.find(dst);
    if (it == d->images.end()) return out;
    it->second.finished = true;
    it->second.reloadFrame = d->frames;
    if (it->second.ci.samples != VK_SAMPLE_COUNT_1_BIT) return out;
    DeviceData::RcbStash* e = RcbShape(d, it->second.ci);
    if (!e)
    {
        if (d->rcbStashes.size() < 8)
        {
            d->rcbStashes.emplace_back();
            DeviceData::RcbStash& n = d->rcbStashes.back();
            n.w = it->second.ci.extent.width; n.h = it->second.ci.extent.height; n.fmt = it->second.ci.format;
        }
        return out;
    }
    out.w = e->w; out.h = e->h;
    // A gap in reloads (a menu, a load) resets the verdicts: the next stretch may run another way, and a stale verdict
    // would freeze the right eye on old frames.
    if (e->lastReload != ~0u && d->frames - e->lastReload > 30) { e->source = -1; e->votedSource = -1; e->votes = 0; e->cpuIsFilter = false; e->filterPasses = 0; }
    e->lastReload = d->frames;
    if (e->cpuIsFilter)
    {
        e->wantSlots = false;
        out.dropLeft = true;
        return out;
    }
    out.probe = RcbNextProbe(d, *e, e->reads, ++e->reloads, dst, &out.off);
    e->wantSlots = e->source == 1;
    if (!e->wantSlots) return out;
    for (uint64_t n = e->reads; n && e->reads - n < DeviceData::RcbStash::kSlots; --n)
    {
        const int k = (int)(n % DeviceData::RcbStash::kSlots);
        if (e->readFrom[k] == dst) continue;
        if (e->img[k] && e->slotRead[k] == n) out.slot = e->img[k];
        break;
    }
    return out;
}

// After the left's bytes landed in `dst`, in `dl`: its rows, then the right frame the verdict names.
static void RcbRecordReload(DeviceData* d, VkCommandBuffer cb, const RcbReload& r, VkImage dst, VkImage ds, VkImageLayout dl)
{
    if (r.probe)
    {
        RcbLayout(d, cb, dst, dl, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        RcbCopyRows(d, cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.probe, r.off, r.w, r.h);
        RcbLayout(d, cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dl);
    }
    if (!r.slot) return;
    RcbTransferBarrier(d, cb);
    const VkImageCopy c = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { r.w, r.h, 1 } };
    d->CmdCopyImage(cb, r.slot, VK_IMAGE_LAYOUT_GENERAL, ds, dl, 1, &c);
}

// Above native resolution the left's frame is the CPU's 1x image blitted up: R takes the same road, down to 1x in two
// box-like steps so its 3x samples average, then the left's own blit up.
static void RcbMatchScale(DeviceData* d, VkCommandBuffer cb, VkImage dst, VkImage ds, VkImageLayout dl, const VkImageBlit& up, VkFilter f)
{
    const uint32_t sw = (uint32_t)abs(up.srcOffsets[1].x - up.srcOffsets[0].x), sh = (uint32_t)abs(up.srcOffsets[1].y - up.srcOffsets[0].y);
    VkImage mid = VK_NULL_HANDLE, low = VK_NULL_HANDLE;
    bool freshMid = false, freshLow = false;
    uint32_t w = 0, h = 0;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->images.find(dst);
        if (it == d->images.end()) return;
        DeviceData::RcbStash* e = RcbShape(d, it->second.ci);
        if (!e || !sw || !sh || sw >= e->w) return;
        w = e->w; h = e->h;
        if (!e->low)
        {
            e->low = RcbMakeImage(d, e->fmt, sw, sh, &e->lowMem);
            if (!e->low) return;
            e->lowW = sw; e->lowH = sh;
            freshLow = true;
            Log("[rcb] %ux%u fmt=%d: the right frame goes through %ux%u, as the left's did\n", w, h, (int)e->fmt, sw, sh);
        }
        if (e->lowW != sw || e->lowH != sh) return;
        if (w / 2 > sw && h / 2 > sh && !e->mid)
        {
            e->mid = RcbMakeImage(d, e->fmt, w / 2, h / 2, &e->midMem);
            freshMid = e->mid != VK_NULL_HANDLE;
        }
        mid = e->mid; low = e->low;
    }
    RcbTransferBarrier(d, cb);
    if (freshLow) RcbToGeneral(d, cb, low);
    if (freshMid) RcbToGeneral(d, cb, mid);
    RcbLayout(d, cb, ds, dl, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const VkImageSubresourceLayers color = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    const VkOffset3D o = { 0, 0, 0 }, full = { (int32_t)w, (int32_t)h, 1 }, half = { (int32_t)(w / 2), (int32_t)(h / 2), 1 }, one = { (int32_t)sw, (int32_t)sh, 1 };
    if (mid)
    {
        const VkImageBlit a = { color, { o, full }, color, { o, half } }, b = { color, { o, half }, color, { o, one } };
        d->CmdBlitImage(cb, ds, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mid, VK_IMAGE_LAYOUT_GENERAL, 1, &a, VK_FILTER_LINEAR);
        RcbTransferBarrier(d, cb);
        d->CmdBlitImage(cb, mid, VK_IMAGE_LAYOUT_GENERAL, low, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_LINEAR);
    }
    else
    {
        const VkImageBlit a = { color, { o, full }, color, { o, one } };
        d->CmdBlitImage(cb, ds, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, low, VK_IMAGE_LAYOUT_GENERAL, 1, &a, VK_FILTER_LINEAR);
    }
    RcbTransferBarrier(d, cb);
    RcbLayout(d, cb, ds, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dl);
    d->CmdBlitImage(cb, low, VK_IMAGE_LAYOUT_GENERAL, ds, dl, 1, &up, f);
}

// The host's read-back is one bilinear blit from 3x to 1x (4 of each 9 samples): above 2:1 the left goes down in R's
// two averaging steps instead, the CPU's work stays. Lock free; true when it took the blit's place.
static bool RcbAverageReadBack(DeviceData* d, VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, const VkImageBlit& down)
{
    VkImage mid = VK_NULL_HANDLE;
    bool fresh = false;
    VkOffset3D half = { 0, 0, 1 };
    int32_t w = 0, h = 0;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto si = d->images.find(src), di = d->images.find(dst);
        if (si == d->images.end() || di == d->images.end() || !si->second.sibling || !IsRsxHelper(di->second) || si->second.ci.samples != VK_SAMPLE_COUNT_1_BIT) return false;
        DeviceData::RcbStash* e = RcbShape(d, si->second.ci);
        if (!e) return false;
        w = (int32_t)e->w; h = (int32_t)e->h;
        const int32_t dw = abs(down.dstOffsets[1].x - down.dstOffsets[0].x), dh = abs(down.dstOffsets[1].y - down.dstOffsets[0].y);
        if (down.srcOffsets[0].x || down.srcOffsets[0].y || down.srcOffsets[1].x != w || down.srcOffsets[1].y != h || w / 2 <= dw || h / 2 <= dh) return false;
        if (!e->mid)
        {
            e->mid = RcbMakeImage(d, e->fmt, e->w / 2, e->h / 2, &e->midMem);
            if (!e->mid) return false;
            fresh = true;
        }
        if (!e->averageLogged)
        {
            e->averageLogged = true;
            Log("[rcb] %ux%u fmt=%d: the left's read-back to %dx%d is averaged\n", e->w, e->h, (int)e->fmt, dw, dh);
        }
        mid = e->mid;
        half.x = w / 2; half.y = h / 2;
    }
    RcbTransferBarrier(d, cb);
    if (fresh) RcbToGeneral(d, cb, mid);
    const VkImageSubresourceLayers color = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    const VkOffset3D o = { 0, 0, 0 }, full = { w, h, 1 };
    const VkImageBlit a = { color, { o, full }, color, { o, half } };
    d->CmdBlitImage(cb, src, sl, mid, VK_IMAGE_LAYOUT_GENERAL, 1, &a, VK_FILTER_LINEAR);
    RcbTransferBarrier(d, cb);
    VkImageBlit b = down;
    b.srcOffsets[0] = o; b.srcOffsets[1] = half;
    d->CmdBlitImage(cb, mid, VK_IMAGE_LAYOUT_GENERAL, dst, dl, 1, &b, VK_FILTER_LINEAR);
    return true;
}

// Normalised correlation of two row probes, luma of every third texel; -2 when either is flat.
static float RcbRowsMatch(const uint8_t* a, const uint8_t* b, uint32_t w)
{
    double sa = 0, sb = 0, aa = 0, bb = 0, ab = 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < 3 * w; i += 3, ++n)
    {
        const uint8_t* p = a + i * 4; const uint8_t* q = b + i * 4;
        const double x = p[0] + 2.0 * p[1] + p[2], y = q[0] + 2.0 * q[1] + q[2];
        sa += x; sb += y; aa += x * x; bb += y * y; ab += x * y;
    }
    const double va = aa - sa * sa / n, vb = bb - sb * sb / n;
    if (va < n * 4.0 || vb < n * 4.0) return -2.f;
    return (float)((ab - sa * sb / n) / sqrt(va * vb));
}

// Mean and peak difference of 1/48-of-the-width blocks of two row probes, luma 0..255: what survives edge averaging.
static float RcbRowsLowDiff(const uint8_t* a, const uint8_t* b, uint32_t w, float* peak)
{
    const uint32_t block = (std::max)(w / 48, 8u);
    float sum = 0.f;
    uint32_t n = 0;
    *peak = 0.f;
    for (uint32_t row = 0; row < 3; ++row)
        for (uint32_t x = 0; x + block <= w; x += block)
        {
            int64_t da = 0, db = 0;
            for (uint32_t k = 0; k < block; ++k)
            {
                const uint8_t* p = a + (row * w + x + k) * 4; const uint8_t* q = b + (row * w + x + k) * 4;
                da += p[0] + 2 * p[1] + p[2];
                db += q[0] + 2 * q[1] + q[2];
            }
            const float diff = fabsf((float)(da - db)) / (4.f * block);
            sum += diff;
            if (diff > *peak) *peak = diff;
            ++n;
        }
    return n ? sum / n : 255.f;
}

// At present, run reloads vote on whose read-back they hold (moving pictures only; another target's needs a wide win,
// four in a row move it), and on whether the CPU only edge-filters: after 30 such reloads both eyes keep their own.
void RcbFrameEnd(DeviceData* d)
{
    if (!Traits().readColorBuffers) return;
    std::lock_guard<std::shared_mutex> g(d->lock);
    using S = DeviceData::RcbStash;
    for (S& e : d->rcbStashes)
    {
        if (!e.probeMap) continue;
        const VkDeviceSize one = 3ull * e.w * 4;
        for (uint32_t o = 0; o < S::kProbes; ++o)
        {
            const uint32_t i = (e.probeNext + o) % S::kProbes;
            S::Probe& p = e.probes[i];
            if (!p.reload || p.checked || p.frame == ~0u || d->frames - p.frame < 4) continue;
            p.checked = true;
            float moved = -2.f, own = -2.f, other = -2.f, best = -2.f;
            int bestAt = -1;
            for (uint32_t j = 0; j < S::kProbes; ++j)
            {
                const S::Probe& q = e.probes[j];
                if (q.frame == ~0u || j == i) continue;
                if (q.reload)
                {
                    if (q.reloadNo + 1 == p.reloadNo && q.image != VK_NULL_HANDLE) moved = RcbRowsMatch(e.probeMap + one * i, e.probeMap + one * j, e.w);
                    continue;
                }
                if (q.read > p.read || q.read + 5 < p.read) continue;
                const float v = RcbRowsMatch(e.probeMap + one * i, e.probeMap + one * j, e.w);
                float& side = q.image == p.image ? own : other;
                if (v > side) side = v;
                if (v > best) { best = v; bestAt = (int)j; }
            }
            if (bestAt >= 0 && !e.cpuIsFilter)
            {
                float peak = 0.f;
                const float low = RcbRowsLowDiff(e.probeMap + one * i, e.probeMap + one * (uint32_t)bestAt, e.w, &peak);
                if (best >= 0.9f && low <= 4.f && peak <= 30.f)
                {
                    if (++e.filterPasses >= 30)
                    {
                        e.cpuIsFilter = true;
                        Log("[rcb] %ux%u fmt=%d: the CPU's frame differs from its source only at edges (match %.3f, low-passed %.2f, peak %.2f): both eyes keep their own, frame %u\n", e.w, e.h, (int)e.fmt, best, low, peak, d->frames);
                    }
                }
                else
                {
                    static std::atomic<int> s_content{ 12 };
                    if (s_content.fetch_sub(1, std::memory_order_relaxed) > 0)
                        Log("[rcb] %ux%u fmt=%d: the CPU's frame stays in the left eye after %u clean reloads (match %.3f, low-passed %.2f, peak %.2f), frame %u\n", e.w, e.h, (int)e.fmt, e.filterPasses, best, low, peak, d->frames);
                    e.filterPasses = 0;
                }
            }
            if (moved < -1.f || moved >= 0.985f || own < -1.f || other < -1.f) continue;
            int vote = -1;
            if (other - own >= 0.05f) vote = 1;
            else if (own - other >= 0.02f) vote = 0;
            if (vote < 0) continue;
            if (vote == e.votedSource) ++e.votes;
            else { e.votedSource = vote; e.votes = 1; }
            if (e.votes >= 4 && e.source != vote)
            {
                e.source = vote;
                Log("[rcb] %ux%u fmt=%d: a reload holds %s read-back (own %.3f, other %.3f, against the last reload %.3f) at frame %u\n", e.w, e.h, (int)e.fmt, vote ? "another target's" : "its own target's", own, other, moved, d->frames);
            }
        }
    }
}

// Lock held. readColorBuffers: the host reloads a target from guest memory once the CPU (which saw the left eye only)
// changed it; a target both eyes drew and the CPU read keeps its right frame, a film with no read-back reaches both.
static bool KeepRightFrame(DeviceData* d, VkImage dst)
{
    if (!Traits().readColorBuffers) return false;
    auto it = d->images.find(dst);
    if (it == d->images.end() || !it->second.sibling) return false;
    const ImageInfo& i = it->second;
    const bool target = (i.ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0, twin = IsRsxLinearTwin(i);
    const bool keep = RcbRecent(d, i.readBackFrame) && RcbHoldsRightFrame(d, i);
    static std::atomic<unsigned> s_kept{ 0 };
    static std::atomic<int> s_mirrored{ 16 };
    if (keep)
    {
        const unsigned t = ++s_kept;
        if (t <= 3 || t % 6000 == 0) Log("[rcb] %u CPU reloads left the right eye's frame in place, last %ux%u fmt=%d\n", t, i.ci.extent.width, i.ci.extent.height, (int)i.ci.format);
    }
    else if ((target || twin) && s_mirrored.fetch_sub(1, std::memory_order_relaxed) > 0)
        Log("[rcb] CPU reload reached both eyes: %ux%u fmt=%d read back %d frames ago, drawn %d frames ago (-1 never)\n", i.ci.extent.width, i.ci.extent.height, (int)i.ci.format,
            i.readBackFrame == ~0u ? -1 : (int)(d->frames - i.readBackFrame), i.rightDrawnFrame == ~0u ? -1 : (int)(d->frames - i.rightDrawnFrame));
    return keep;
}

// An image into a buffer (a host's image download): the twin takes the right eye's image.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkBuffer dst, uint32_t n, const VkBufferImageCopy* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdCopyImageToBuffer(cb, src, sl, dst, n, p);
    VkImage ss = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (ImageIsStereo(d, src, &ss))
        {
            ++d->statDoubledToBuffer;
            if (Traits().readColorBuffers) d->images[src].readBackFrame = d->frames;
        }
    }
    if (ss && Traits().readColorBuffers) RcbOnReadBack(d, cb, src, ss, sl);
    const VkBuffer dt = TwinOf(d, dst);
    if (!dt)
    {
        // bufferTrips: an image -> buffer -> image copy (e.g. depth D32S8 -> R32F) whose buffer has no twin: the copy
        // out of this buffer takes the right eye's own road (Hook_CmdCopyBufferToImage).
        if (ss && Traits().bufferTrips)
        {
            CmdState* s = State(d, cb);
            s->tripBuffer = dst; s->tripSrcR = ss; s->tripLayout = sl;
            s->tripRegions.assign(p, p + n);
            s->tripDispatches.clear();
        }
        return;
    }
    d->CmdCopyImageToBuffer(cb, ss ? ss : src, sl, dt, n, p);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyImageToBuffer2(VkCommandBuffer cb, const VkCopyImageToBufferInfo2* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdCopyImageToBuffer2(cb, p);
    VkImage ss = VK_NULL_HANDLE;
    { std::lock_guard<std::shared_mutex> g(d->lock); if (ImageIsStereo(d, p->srcImage, &ss)) ++d->statDoubledToBuffer; }
    const VkBuffer dt = TwinOf(d, p->dstBuffer);
    if (!dt) return;
    VkCopyImageToBufferInfo2 c = *p;
    c.srcImage = ss ? ss : p->srcImage;
    c.dstBuffer = dt;
    d->CmdCopyImageToBuffer2(cb, &c);
}

// Lock held. A sibling that is only ever read is the right eye reading cleared
// memory, so every path that gives one content says so here.
static void MarkSiblingWritten(DeviceData* d, VkImage img)
{
    auto it = d->images.find(img);
    if (it != d->images.end()) it->second.siblingWritten = true;
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const VkImageCopy* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdCopyImage(cb, src, sl, dst, dl, n, p);
    VkImage ss = VK_NULL_HANDLE, ds = VK_NULL_HANDLE;
    { std::lock_guard<std::shared_mutex> g(d->lock); ImageIsStereo(d, src, &ss); if (ImageIsStereo(d, dst, &ds)) MarkSiblingWritten(d, dst); }
    if (!ds) return;
    d->CmdCopyImage(cb, ss ? ss : src, sl, ds, dl, n, p);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const VkImageBlit* p, VkFilter f)
{
    DeviceData* d = FindDevice(cb);
    VkImage ss = VK_NULL_HANDLE, ds = VK_NULL_HANDLE;
    RcbReload plan;
    bool keep = false, readBack = false;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        ImageIsStereo(d, src, &ss);
        if (ImageIsStereo(d, dst, &ds)) MarkSiblingWritten(d, dst);
        // Above native resolution a read-back is target -> helper -> buffer, a reload buffer -> helper -> target.
        if (Traits().readColorBuffers)
        {
            auto si = d->images.find(src), di = d->images.find(dst);
            if (si != d->images.end() && di != d->images.end())
            {
                if (ss && !ds && IsRsxHelper(di->second)) { si->second.readBackFrame = d->frames; readBack = true; }
                const bool whole = n == 1 && p[0].dstOffsets[0].x == 0 && p[0].dstOffsets[0].y == 0 &&
                                   p[0].dstOffsets[1].x == (int32_t)di->second.ci.extent.width && p[0].dstOffsets[1].y == (int32_t)di->second.ci.extent.height;
                if (!ss && ds && whole && si->second.cpuLoaded && KeepRightFrame(d, dst)) { keep = true; plan = RcbForReload(d, dst); }
                si->second.cpuLoaded = false;
                di->second.cpuLoaded = false;
            }
        }
    }
    if (keep)
    {
        if (!plan.dropLeft) d->CmdBlitImage(cb, src, sl, dst, dl, n, p, f);
        RcbRecordReload(d, cb, plan, dst, ds, dl);
        if (plan.dropLeft) RcbMatchScale(d, cb, dst, dst, dl, p[0], f);
        RcbMatchScale(d, cb, dst, ds, dl, p[0], f);
        return;
    }
    if (!(readBack && n == 1 && RcbAverageReadBack(d, cb, src, sl, dst, dl, p[0]))) d->CmdBlitImage(cb, src, sl, dst, dl, n, p, f);
    if (readBack) RcbOnReadBack(d, cb, src, ss, sl);
    if (!ds) return;
    d->CmdBlitImage(cb, ss ? ss : src, sl, ds, dl, n, p, f);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const VkImageResolve* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdResolveImage(cb, src, sl, dst, dl, n, p);
    VkImage ss = VK_NULL_HANDLE, ds = VK_NULL_HANDLE;
    { std::lock_guard<std::shared_mutex> g(d->lock); ImageIsStereo(d, src, &ss); if (ImageIsStereo(d, dst, &ds)) MarkSiblingWritten(d, dst); }
    if (ds) d->CmdResolveImage(cb, ss ? ss : src, sl, ds, dl, n, p);
}


static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer src, VkImage dst, VkImageLayout dl, uint32_t n, const VkBufferImageCopy* p)
{
    DeviceData* d = FindDevice(cb);
    if (Traits().readColorBuffers && n == 1 && State(d, cb)->tripBuffer != src)
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (RcbLeftKeepsOwn(d, dst, p[0])) return;
    }
    d->CmdCopyBufferToImage(cb, src, dst, dl, n, p);
    VkImage ds = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (ImageIsStereo(d, dst, &ds)) { MarkSiblingWritten(d, dst); ++d->statBufferToDoubled; ++d->bufferToDoubledImages[dst]; }
    }
    // bufferTrips: an image's trip through a buffer (depth <-> R32F, or a typeless copy with a byte swap between) is
    // repeated from the right image, or the right copy gets the left's bytes and ghosts at the left's position.
    if (Traits().bufferTrips)
    {
        CmdState* s = State(d, cb);
        const bool trip = ds && n && s->tripBuffer == src && !s->tripRegions.empty() && s->tripRegions[0].bufferOffset == p[0].bufferOffset;
        if (trip)
        {
            const VkMemoryBarrier toWrite = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT };
            d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toWrite, 0, nullptr, 0, nullptr);
            d->CmdCopyImageToBuffer(cb, s->tripSrcR, s->tripLayout, src, (uint32_t)s->tripRegions.size(), s->tripRegions.data());
            // The kernel, its set and push constants are still bound: only a barrier stood between it and that copy.
            if (!s->tripDispatches.empty())
            {
                const VkMemoryBarrier toShader = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &toShader, 0, nullptr, 0, nullptr);
                for (const auto& g : s->tripDispatches) d->CmdDispatch(cb, g[0], g[1], g[2]);
                const VkMemoryBarrier fromShader = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT };
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &fromShader, 0, nullptr, 0, nullptr);
                s->tripDispatches.clear();
            }
            else
            {
                const VkMemoryBarrier toRead = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT };
                d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toRead, 0, nullptr, 0, nullptr);
            }
            d->CmdCopyBufferToImage(cb, src, ds, dl, n, p);
            s->tripBuffer = VK_NULL_HANDLE;
            static std::atomic<unsigned> s_trips{ 0 };
            const unsigned t = ++s_trips;
            if (t <= 3 || t % 6000 == 0) Log("[trip] %u right-eye image trips through a buffer\n", t);
            return;
        }
        if (s->tripBuffer == src) s->tripBuffer = VK_NULL_HANDLE;
    }
    if (Traits().readColorBuffers && n == 1)
    {
        RcbReload plan;
        bool keep = false;
        {
            std::lock_guard<std::shared_mutex> g(d->lock);
            auto it = d->images.find(dst);
            if (it != d->images.end())
            {
                it->second.cpuLoaded = !ds && IsRsxHelper(it->second);
                const VkExtent3D& e = it->second.ci.extent;
                const VkBufferImageCopy& r = p[0];
                if (ds && !r.imageOffset.x && !r.imageOffset.y && r.imageExtent.width == e.width && r.imageExtent.height == e.height && !r.imageSubresource.mipLevel && KeepRightFrame(d, dst))
                {
                    keep = true;
                    plan = RcbForReload(d, dst);
                }
            }
        }
        if (keep)
        {
            RcbRecordReload(d, cb, plan, dst, ds, dl);
            return;
        }
    }
    if (!ds) return;
    const VkBuffer st = TwinOf(d, src);   // an image upload: the right eye's staging
    d->CmdCopyBufferToImage(cb, st ? st : src, ds, dl, n, p);
}

// Copy commands 2 (core 1.3): the same mirrors through the info structs.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdCopyImage2(cb, p);
    if (!Vk().VKDynamicRendering) return;
    VkImage ss = VK_NULL_HANDLE, ds = VK_NULL_HANDLE;
    { std::lock_guard<std::shared_mutex> g(d->lock); ImageIsStereo(d, p->srcImage, &ss); if (ImageIsStereo(d, p->dstImage, &ds)) MarkSiblingWritten(d, p->dstImage); }
    if (!ds) return;
    VkCopyImageInfo2 c = *p;
    c.srcImage = ss ? ss : p->srcImage; c.dstImage = ds;
    d->CmdCopyImage2(cb, &c);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdBlitImage2(cb, p);
    if (!Vk().VKDynamicRendering) return;
    VkImage ss = VK_NULL_HANDLE, ds = VK_NULL_HANDLE;
    { std::lock_guard<std::shared_mutex> g(d->lock); ImageIsStereo(d, p->srcImage, &ss); if (ImageIsStereo(d, p->dstImage, &ds)) MarkSiblingWritten(d, p->dstImage); }
    if (!ds) return;
    VkBlitImageInfo2 c = *p;
    c.srcImage = ss ? ss : p->srcImage; c.dstImage = ds;
    d->CmdBlitImage2(cb, &c);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdResolveImage2(VkCommandBuffer cb, const VkResolveImageInfo2* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdResolveImage2(cb, p);
    if (!Vk().VKDynamicRendering) return;
    VkImage ss = VK_NULL_HANDLE, ds = VK_NULL_HANDLE;
    { std::lock_guard<std::shared_mutex> g(d->lock); ImageIsStereo(d, p->srcImage, &ss); if (ImageIsStereo(d, p->dstImage, &ds)) MarkSiblingWritten(d, p->dstImage); }
    if (!ds) return;
    VkResolveImageInfo2 c = *p;
    c.srcImage = ss ? ss : p->srcImage; c.dstImage = ds;
    d->CmdResolveImage2(cb, &c);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdCopyBufferToImage2(VkCommandBuffer cb, const VkCopyBufferToImageInfo2* p)
{
    DeviceData* d = FindDevice(cb);
    d->CmdCopyBufferToImage2(cb, p);
    if (!Vk().VKDynamicRendering) return;
    VkImage ds = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (ImageIsStereo(d, p->dstImage, &ds)) { MarkSiblingWritten(d, p->dstImage); ++d->statBufferToDoubled; ++d->bufferToDoubledImages[p->dstImage]; }
    }
    if (!ds) return;
    VkCopyBufferToImageInfo2 c = *p;
    const VkBuffer st = TwinOf(d, p->srcBuffer);
    if (st) c.srcBuffer = st;
    c.dstImage = ds;
    d->CmdCopyBufferToImage2(cb, &c);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdClearColorImage(VkCommandBuffer cb, VkImage img, VkImageLayout l, const VkClearColorValue* c, uint32_t n, const VkImageSubresourceRange* r)
{
    DeviceData* d = FindDevice(cb);
    d->CmdClearColorImage(cb, img, l, c, n, r);
    VkImage s = VK_NULL_HANDLE;
    { std::lock_guard<std::shared_mutex> g(d->lock); if (ImageIsStereo(d, img, &s)) MarkSiblingWritten(d, img); }
    if (s) d->CmdClearColorImage(cb, s, l, c, n, r);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdClearDepthStencilImage(VkCommandBuffer cb, VkImage img, VkImageLayout l, const VkClearDepthStencilValue* c, uint32_t n, const VkImageSubresourceRange* r)
{
    DeviceData* d = FindDevice(cb);
    d->CmdClearDepthStencilImage(cb, img, l, c, n, r);
    VkImage s = VK_NULL_HANDLE;
    { std::lock_guard<std::shared_mutex> g(d->lock); if (ImageIsStereo(d, img, &s)) MarkSiblingWritten(d, img); }
    if (s) d->CmdClearDepthStencilImage(cb, s, l, c, n, r);
}

// Image barriers on doubled images get a twin for the sibling, so both track one layout; inside a pass (where the
// sibling is no attachment) the barrier is recorded and replayed instead.
static VKAPI_ATTR void VKAPI_CALL Hook_CmdPipelineBarrier(
    VkCommandBuffer cb, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkDependencyFlags flags,
    uint32_t mc, const VkMemoryBarrier* pM, uint32_t bc, const VkBufferMemoryBarrier* pB, uint32_t ic, const VkImageMemoryBarrier* pI)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    if (!d->stereoEnabled || (!ic && (!bc || !Vk().VKBufferTwins)))
    {
        d->CmdPipelineBarrier(cb, srcStage, dstStage, flags, mc, pM, bc, pB, ic, pI);
        return;
    }
    if (s->inPass)
    {
        d->CmdPipelineBarrier(cb, srcStage, dstStage, flags, mc, pM, bc, pB, ic, pI);
        if (Recording(s) && !s->replaying)
        {
            Rec& r = Push(s, Op::Barrier);
            r.u[0] = srcStage; r.u[1] = dstStage; r.u[2] = flags; r.u[3] = mc; r.u[4] = bc; r.u[5] = ic;
            std::vector<uint8_t> tmp(mc * sizeof(VkMemoryBarrier) + bc * sizeof(VkBufferMemoryBarrier) + ic * sizeof(VkImageMemoryBarrier));
            uint8_t* w = tmp.data();
            memcpy(w, pM, mc * sizeof(VkMemoryBarrier)); w += mc * sizeof(VkMemoryBarrier);
            memcpy(w, pB, bc * sizeof(VkBufferMemoryBarrier)); w += bc * sizeof(VkBufferMemoryBarrier);
            memcpy(w, pI, ic * sizeof(VkImageMemoryBarrier));
            Payload(s, r, tmp.data(), tmp.size());
        }
        return;
    }
    std::vector<VkImageMemoryBarrier> img(pI, pI + ic);
    std::vector<VkBufferMemoryBarrier> buf(pB, pB + bc);
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        for (uint32_t i = 0; i < ic; ++i)
        {
            VkImage sib;
            if (ImageIsStereo(d, pI[i].image, &sib))
            {
                VkImageMemoryBarrier b = pI[i];
                b.image = sib;
                img.push_back(b);
            }
        }
        if (Vk().VKBufferTwins)
            for (uint32_t i = 0; i < bc; ++i)
                if (const VkBuffer twin = BufferTwin(d, pB[i].buffer)) { VkBufferMemoryBarrier b = pB[i]; b.buffer = twin; buf.push_back(b); }
    }
    d->CmdPipelineBarrier(cb, srcStage, dstStage, flags, mc, pM, (uint32_t)buf.size(), buf.data(), (uint32_t)img.size(), img.data());
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* pDep)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    if (!d->stereoEnabled || (s->inPass && !Vk().VKDynamicRendering) ||
        (!s->inPass && !pDep->imageMemoryBarrierCount && (!pDep->bufferMemoryBarrierCount || !Vk().VKBufferTwins)))
    {
        d->CmdPipelineBarrier2(cb, pDep);
        return;
    }
    // Inside a pass (feedback loops), recorded like Op::Barrier, pNext dropped.
    if (s->inPass)
    {
        d->CmdPipelineBarrier2(cb, pDep);
        if (Recording(s) && !s->replaying)
        {
            const uint32_t mc = pDep->memoryBarrierCount, bc = pDep->bufferMemoryBarrierCount, ic = pDep->imageMemoryBarrierCount;
            Rec& r = Push(s, Op::Barrier2);
            r.u[0] = pDep->dependencyFlags; r.u[1] = mc; r.u[2] = bc; r.u[3] = ic;
            std::vector<uint8_t> tmp(mc * sizeof(VkMemoryBarrier2) + bc * sizeof(VkBufferMemoryBarrier2) + ic * sizeof(VkImageMemoryBarrier2));
            uint8_t* w = tmp.data();
            for (uint32_t i = 0; i < mc; ++i) { VkMemoryBarrier2 b = pDep->pMemoryBarriers[i]; b.pNext = nullptr; memcpy(w, &b, sizeof(b)); w += sizeof(b); }
            for (uint32_t i = 0; i < bc; ++i) { VkBufferMemoryBarrier2 b = pDep->pBufferMemoryBarriers[i]; b.pNext = nullptr; memcpy(w, &b, sizeof(b)); w += sizeof(b); }
            for (uint32_t i = 0; i < ic; ++i) { VkImageMemoryBarrier2 b = pDep->pImageMemoryBarriers[i]; b.pNext = nullptr; memcpy(w, &b, sizeof(b)); w += sizeof(b); }
            Payload(s, r, tmp.data(), tmp.size());
        }
        return;
    }
    std::vector<VkImageMemoryBarrier2> img(pDep->pImageMemoryBarriers, pDep->pImageMemoryBarriers + pDep->imageMemoryBarrierCount);
    std::vector<VkBufferMemoryBarrier2> buf(pDep->pBufferMemoryBarriers, pDep->pBufferMemoryBarriers + pDep->bufferMemoryBarrierCount);
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        for (uint32_t i = 0; i < pDep->imageMemoryBarrierCount; ++i)
        {
            VkImage sib;
            if (ImageIsStereo(d, pDep->pImageMemoryBarriers[i].image, &sib))
            {
                VkImageMemoryBarrier2 b = pDep->pImageMemoryBarriers[i];
                b.image = sib;
                img.push_back(b);
            }
        }
        if (Vk().VKBufferTwins)
            for (uint32_t i = 0; i < pDep->bufferMemoryBarrierCount; ++i)
                if (const VkBuffer twin = BufferTwin(d, pDep->pBufferMemoryBarriers[i].buffer))
                {
                    VkBufferMemoryBarrier2 b = pDep->pBufferMemoryBarriers[i];
                    b.buffer = twin;
                    buf.push_back(b);
                }
    }
    VkDependencyInfo di = *pDep;
    di.imageMemoryBarrierCount = (uint32_t)img.size();
    di.pImageMemoryBarriers = img.data();
    di.bufferMemoryBarrierCount = (uint32_t)buf.size();
    di.pBufferMemoryBarriers = buf.data();
    d->CmdPipelineBarrier2(cb, &di);
}

// Render passes
static void BeginPass(DeviceData* d, CmdState* s, const VkRenderPassBeginInfo* pInfo, VkSubpassContents contents)
{
    s->inPass = true;
    s->stereoPass = false;
    s->passSibling = VK_NULL_HANDLE;
    s->passAspectOk = true;
    s->mixedCopies.clear();
    // A trip runs outside passes: a buffer met after a pass is no trip's second half.
    if (Traits().tripKernels) s->tripBuffer = VK_NULL_HANDLE;
    if (!d->stereoEnabled || s->secondary || s->foreign || t_foreignCalls) return;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->framebuffers.find(pInfo->framebuffer);
        if (it != d->framebuffers.end())
        {
            s->passSibling = it->second.sibling;
            // monoPassCopies: a mixed pass stays mono, and its doubled attachments' results are copied to their
            // siblings at its end, or the right eye samples them unwritten.
            auto rit = Traits().monoPassCopies && !it->second.mixed.empty() ? d->renderPassAtts.find(pInfo->renderPass) : d->renderPassAtts.end();
            if (rit != d->renderPassAtts.end())
                for (const MixedAttachment& m : it->second.mixed)
                    if (m.index < rit->second.size())
                    {
                        s->mixedCopies.push_back({ m.image, m.sibling, m.range, rit->second[m.index].finalLayout, m.w, m.h, rit->second[m.index].initialLayout });
                        static std::atomic<int> s_log{ 32 };
                        if (s_log.fetch_sub(1, std::memory_order_relaxed) > 0) Log("[mixed] %ux%u attachment %u: mono pass, its result copied to the right image\n", m.w, m.h, m.index);
                    }
            s->passW = it->second.width; s->passH = it->second.height;
            NotePassAspect(d, s);
            // The game renders into these and the replay below into their siblings (a blank sibling of a written
            // image means a path we do not mirror).
            for (VkImage img : it->second.doubled)
            {
                auto ii = d->images.find(img);
                if (ii == d->images.end()) continue;
                ii->second.everWritten = true;
                if (it->second.sibling) { ii->second.siblingWritten = true; ii->second.rightDrawnFrame = d->frames; }
                if (ii->second.reloadFrame != d->frames) ii->second.finished = false;
            }
        }
    }
    if (!s->passSibling) { ++d->statMonoPasses; return; }
    s->stereoPass = true;
    s->passBegin = *pInfo;
    s->passBegin.pNext = nullptr;
    s->clears.assign(pInfo->pClearValues, pInfo->pClearValues + pInfo->clearValueCount);
    s->passBegin.pClearValues = s->clears.data();
    s->passBegin.framebuffer = s->passSibling;
    s->passStart = s->gfx;
    s->recs.clear();
    s->blob.clear();
    ++d->statStereoPasses;
    (void)contents;
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* pInfo, VkSubpassContents contents)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdBeginRenderPass(cb, pInfo, contents);
    BeginPass(d, s, pInfo, contents);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* pInfo, const VkSubpassBeginInfo* pSub)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdBeginRenderPass2(cb, pInfo, pSub);
    BeginPass(d, s, pInfo, pSub->contents);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdNextSubpass(VkCommandBuffer cb, VkSubpassContents contents)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdNextSubpass(cb, contents);
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::NextSubpass); r.u[0] = contents; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdNextSubpass2(VkCommandBuffer cb, const VkSubpassBeginInfo* pB, const VkSubpassEndInfo* pE)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdNextSubpass2(cb, pB, pE);
    if (Recording(s) && !s->replaying) { Rec& r = Push(s, Op::NextSubpass); r.u[0] = pB->contents; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdExecuteCommands(VkCommandBuffer cb, uint32_t count, const VkCommandBuffer* pCbs)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdExecuteCommands(cb, count, pCbs);
    if (gInfo.DetectInverseMatrix && !s->replaying)
    {
        // The secondaries' sets are used by this primary's submit.
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (uint32_t i = 0; i < count; ++i)
        {
            auto it = d->cmdStates.find(pCbs[i]);
            if (it != d->cmdStates.end()) s->usedSets.insert(s->usedSets.end(), it->second->usedSets.begin(), it->second->usedSets.end());
        }
    }
    if (Recording(s) && !s->replaying)
    {
        Rec& r = Push(s, Op::Execute);
        r.u[0] = count;
        Payload(s, r, pCbs, count * sizeof(VkCommandBuffer));
        ++d->statSecondaryPasses;
    }
    // Secondaries leave the primary's state undefined; force rebinds.
    s->eyeLayout = VK_NULL_HANDLE;
}

// Re-issues the graphics state captured at pass begin.
// The right eye's replay reads vertex, index and indirect data from twins (VKBufferTwins).
static VkBuffer RightBuf(DeviceData* d, VkBuffer b)
{
    const VkBuffer t = TwinOf(d, b);
    return t ? t : b;
}

// Lamp tiles (LampTileVS): the right eye opens the per-tile depth bounds the game cut from the left eye's depth.
static bool LampPipeline(DeviceData* d, VkPipeline p)
{
    std::shared_lock<std::shared_mutex> g(d->lock);
    auto it = d->pipelines.find(p);
    return it != d->pipelines.end() && LampTileVS(it->second.vsCrc);
}

static void LampDraw(DeviceData* d, CmdState* s, bool before)
{
    if (!s->lampPipe) return;
    if (before) d->CmdSetDepthBounds(s->cb, 0.f, 1.f);
    else d->CmdSetDepthBounds(s->cb, s->replayDb[0], s->replayDb[1]);
}

// The order a replay re-pushes constants in: the game's write order with VKPushConstantOrder (overlapping ranges
// then resolve as they did for it), else first-seen order as before.
static std::vector<const PushRange*> PushesInOrder(const std::vector<PushRange>& pushes)
{
    std::vector<const PushRange*> order;
    order.reserve(pushes.size());
    for (const PushRange& p : pushes) order.push_back(&p);
    if (Vk().VKPushConstantOrder)
        std::sort(order.begin(), order.end(), [](const PushRange* a, const PushRange* b) { return a->seq < b->seq; });
    return order;
}

static void Reestablish(DeviceData* d, CmdState* s, const GfxState& g)
{
    const VkCommandBuffer cb = s->cb;
    if (g.pipeline) { d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g.pipeline); PipelineLayoutFor(d, s, g.pipeline, &s->gfxShifted); s->lampPipe = Vk().VKFsCameraWindows && LampPipeline(d, g.pipeline); }
    for (uint32_t i = 0; i < 32; ++i)
    {
        const SetBind& b = g.sets[i];
        if (!b.set || !b.layout) continue;
        VkDescriptorSet set = b.set;
        if (s->replaying && i < 8) { s->rGameSet[i] = b.set; s->rGameSetLayout[i] = b.layout; s->rBind[i] = b; s->rBindKnown[i] = true; }
        if (s->replaying)
        {
            auto it = s->replayShadow.find(set);
            if (it != s->replayShadow.end() && it->second) set = it->second;
        }
        d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, b.layout, i, 1, &set,
                                 (uint32_t)b.dyn.size(), b.dyn.empty() ? nullptr : b.dyn.data());
    }
    for (uint32_t i = 0; i < 32; ++i)
        if (!g.pushSets[i].empty()) IssuePushes(d, cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g.pushLayout[i], i, g.pushSets[i], s->replaying);
    const bool twins = s->replaying && Vk().VKBufferTwins;
    if (g.hasIndex)
    {
        const VkBuffer ib = twins ? RightBuf(d, g.indexBuf) : g.indexBuf;
        if (g.index2 && d->CmdBindIndexBuffer2) d->CmdBindIndexBuffer2(cb, ib, g.indexOff, g.indexSize, g.indexType);
        else d->CmdBindIndexBuffer(cb, ib, g.indexOff, g.indexType);
    }
    VkBuffer vb[32];
    for (uint32_t i = 0; i < 32; ++i) vb[i] = (twins && (g.vbMask & (1u << i))) ? RightBuf(d, g.vb[i]) : g.vb[i];
    for (uint32_t i = 0; i < 32; ++i)
    {
        if (!(g.vbMask & (1u << i))) continue;
        // A run is restored by the call that bound it: the 2-variant also carries the size and the dynamic stride.
        const bool two = (g.vb2Mask & (1u << i)) != 0;
        const bool strides = (g.vbStrideMask & (1u << i)) != 0;
        uint32_t count = 1;
        while (i + count < 32 && (g.vbMask & (1u << (i + count)))
               && (((g.vb2Mask >> (i + count)) & 1u) != 0) == two
               && (((g.vbStrideMask >> (i + count)) & 1u) != 0) == strides) ++count;
        if (two && d->CmdBindVertexBuffers2)
            d->CmdBindVertexBuffers2(cb, i, count, &vb[i], &g.vbOff[i], &g.vbSize[i], strides ? &g.vbStride[i] : nullptr);
        else
            d->CmdBindVertexBuffers(cb, i, count, &vb[i], &g.vbOff[i]);
        i += count - 1;
    }
    for (const PushRange* p : PushesInOrder(g.pushes)) d->CmdPushConstants(cb, p->layout, p->stages, p->offset, p->size, p->data);
    // staticPipelineState: core state the pipeline holds static stays as its bind set it (the core half of
    // VKDynByPipeline), so re-issued dynamic values cannot override a pipeline's stencil reference or masks.
    uint32_t core = ~0u;
    if (Traits().staticPipelineState && g.pipeline)
    {
        std::shared_lock<std::shared_mutex> gl(d->lock);
        auto it = d->pipelines.find(g.pipeline);
        if (it != d->pipelines.end()) core = it->second.coreDynDeclared;
    }
    auto dynamicIn = [core](VkDynamicState st) { return ((core >> (uint32_t)st) & 1) != 0; };
    if (!g.viewports.empty() && dynamicIn(VK_DYNAMIC_STATE_VIEWPORT)) d->CmdSetViewport(cb, g.viewportFirst, (uint32_t)g.viewports.size(), g.viewports.data());
    if (!g.scissors.empty() && dynamicIn(VK_DYNAMIC_STATE_SCISSOR)) d->CmdSetScissor(cb, g.scissorFirst, (uint32_t)g.scissors.size(), g.scissors.data());
    if (g.hasLineWidth && dynamicIn(VK_DYNAMIC_STATE_LINE_WIDTH)) d->CmdSetLineWidth(cb, g.lineWidth);
    if (g.hasDepthBias && dynamicIn(VK_DYNAMIC_STATE_DEPTH_BIAS)) d->CmdSetDepthBias(cb, g.depthBias[0], g.depthBias[1], g.depthBias[2]);
    if (g.hasBlend && dynamicIn(VK_DYNAMIC_STATE_BLEND_CONSTANTS)) d->CmdSetBlendConstants(cb, g.blend);
    if (g.hasDepthBounds && dynamicIn(VK_DYNAMIC_STATE_DEPTH_BOUNDS)) { d->CmdSetDepthBounds(cb, g.depthBounds[0], g.depthBounds[1]); s->replayDb[0] = g.depthBounds[0]; s->replayDb[1] = g.depthBounds[1]; }
    if (g.stencilCompare.has && dynamicIn(VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK)) { d->CmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, g.stencilCompare.front); d->CmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, g.stencilCompare.back); }
    if (g.stencilWrite.has && dynamicIn(VK_DYNAMIC_STATE_STENCIL_WRITE_MASK)) { d->CmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, g.stencilWrite.front); d->CmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, g.stencilWrite.back); }
    if (g.stencilRef.has && dynamicIn(VK_DYNAMIC_STATE_STENCIL_REFERENCE)) { d->CmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, g.stencilRef.front); d->CmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, g.stencilRef.back); }
    // After the pipeline bind, in the order the game last set them: overlapping ranges resolve as they did.
    if (!g.dyn.empty())
    {
        std::vector<const DynState*> order;
        order.reserve(g.dyn.size());
        for (const DynState& e : g.dyn) order.push_back(&e);
        std::sort(order.begin(), order.end(), [](const DynState* a, const DynState* b) { return a->seq < b->seq; });
        uint32_t vpSeq = 0, scSeq = 0;
        // VKDynByPipeline: a state this pipeline holds static stays as its bind set it.
        uint64_t declared = ~0ull;
        uint32_t vsCrc = 0, fsCrc = 0;
        if (Vk().VKDynByPipeline && g.pipeline)
        {
            std::shared_lock<std::shared_mutex> gl(d->lock);
            auto it = d->pipelines.find(g.pipeline);
            if (it != d->pipelines.end()) { declared = it->second.dynDeclared; vsCrc = it->second.vsCrc; fsCrc = it->second.fsCrc; }
        }
        std::string skipped;
        for (const DynState* e : order)
        {
            if (!((declared >> (unsigned)e->fn) & 1))
            {
                if (s->replaying)
                {
                    uint32_t v = 0;
                    memcpy(&v, e->data(), e->len < 4 ? e->len : 4);
                    char one[32];
                    sprintf_s(one, " %u=%X", (unsigned)e->fn, v);
                    skipped += one;
                }
                continue;
            }
            IssueDyn(d, cb, e->fn, e->key, e->data(), e->len);
            if (e->fn == Dyn::ViewportWithCount) vpSeq = e->seq;
            if (e->fn == Dyn::ScissorWithCount) scSeq = e->seq;
        }
        // A core viewport/scissor set after the WithCount one is the live state.
        if (!g.viewports.empty() && g.viewportSeq > vpSeq) d->CmdSetViewport(cb, g.viewportFirst, (uint32_t)g.viewports.size(), g.viewports.data());
        if (!g.scissors.empty() && g.scissorSeq > scSeq) d->CmdSetScissor(cb, g.scissorFirst, (uint32_t)g.scissors.size(), g.scissors.data());
        static std::atomic<int> s_skipLog{24};
        if (!skipped.empty() && s_skipLog.fetch_sub(1, std::memory_order_relaxed) > 0)
            Log("[dynpipe] R pass start: VS %08X FS %08X holds static, not re-issued (Dyn=first dword):%s\n", vsCrc, fsCrc, skipped.c_str());
    }
    s->eyeLayout = VK_NULL_HANDLE;
}

static void ReplayRecords(DeviceData* d, CmdState* s, const std::vector<Rec>& recs, const std::vector<uint8_t>& blob);

// VKIndirectFromLeft: the binds a right replay has in effect, by the game's handles.
static void NoteReplayBind(CmdState* s, const Rec& r, const uint8_t* data)
{
    switch (r.op)
    {
    case Op::BindIndex: case Op::BindIndex2:
        s->rIb = (VkBuffer)r.h[0]; s->rIbOff = r.h[1]; s->rIbType = (VkIndexType)r.u[0];
        s->rIbTwo = r.op == Op::BindIndex2; s->rIbSize = s->rIbTwo ? r.h[2] : VK_WHOLE_SIZE;
        break;
    case Op::BindVertex: case Op::BindVertex2:
    {
        const uint32_t first = r.u[0], count = r.u[1];
        const bool two = r.op == Op::BindVertex2;
        const VkBuffer* bufs = (const VkBuffer*)data;
        const VkDeviceSize* offs = (const VkDeviceSize*)(data + count * sizeof(VkBuffer));
        for (uint32_t k = 0; k < count && first + k < 32; ++k)
        {
            const uint32_t i = first + k, bit = 1u << i;
            s->rVb[i] = bufs[k]; s->rVbOff[i] = offs[k];
            s->rVbMask |= bit;
            s->rVb2Mask = two ? (s->rVb2Mask | bit) : (s->rVb2Mask & ~bit);
            if (two) { s->rVbSize[i] = offs[count + k]; s->rVbStride[i] = offs[2 * count + k]; }
            s->rVbStrideMask = (two && (r.u[2] & 4)) ? (s->rVbStrideMask | bit) : (s->rVbStrideMask & ~bit);
        }
        break;
    }
    case Op::PushDescriptors:
        if (r.u[0] == VK_PIPELINE_BIND_POINT_GRAPHICS && r.u[1] < 8)
        {
            if (Vk().VKPushDeltas && s->rPushLayout[r.u[1]] == (VkPipelineLayout)r.h[0] && s->rPush[r.u[1]].size() >= 4) MergePushes(data, r.dataLen, s->rPush[r.u[1]]);
            else s->rPush[r.u[1]].assign(data, data + r.dataLen);
            s->rPushLayout[r.u[1]] = (VkPipelineLayout)r.h[0];
        }
        break;
    default: break;
    }
}

// VKIndirectFromLeft: the noted binds again, with the left's buffers (`left`) or the right's; images stay the right's.
static void ReissueReplayBinds(DeviceData* d, CmdState* s, bool left)
{
    const VkCommandBuffer cb = s->cb;
    t_leftBuffers = left;
    for (uint32_t i = 0; i < 8; ++i)
        if (!s->rPush[i].empty() && s->rPushLayout[i] == s->gfx.layout)
            IssuePushes(d, cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s->rPushLayout[i], i, s->rPush[i], true);
    t_leftBuffers = false;
    if (s->rIb)
    {
        const VkBuffer ib = left ? s->rIb : RightBuf(d, s->rIb);
        if (s->rIbTwo && d->CmdBindIndexBuffer2) d->CmdBindIndexBuffer2(cb, ib, s->rIbOff, s->rIbSize, s->rIbType);
        else d->CmdBindIndexBuffer(cb, ib, s->rIbOff, s->rIbType);
    }
    for (uint32_t i = 0; i < 32; ++i)
    {
        if (!(s->rVbMask & (1u << i))) continue;
        const VkBuffer vb = left ? s->rVb[i] : RightBuf(d, s->rVb[i]);
        if ((s->rVb2Mask & (1u << i)) && d->CmdBindVertexBuffers2)
            d->CmdBindVertexBuffers2(cb, i, 1, &vb, &s->rVbOff[i], &s->rVbSize[i], (s->rVbStrideMask & (1u << i)) ? &s->rVbStride[i] : nullptr);
        else d->CmdBindVertexBuffers(cb, i, 1, &vb, &s->rVbOff[i]);
    }
}

// screenTriangles: `slot` bound again with the game's own set (`left`), or back with its shadow. False when it is not
// on a shadow, or (counted) its bind is not known or its set holds a doubled image, which the right eye needs.
static bool ScreenTriangleSet(DeviceData* d, CmdState* s, uint32_t slot, bool left)
{
    if (slot >= 8 || !s->rBindKnown[slot]) { if (left) ++d->statScreenTriRefused; return false; }
    const SetBind& b = s->rBind[slot];
    auto it = s->replayShadow.find(b.set);
    if (it == s->replayShadow.end() || !it->second) return false;
    if (left)
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        const DescriptorSetInfo* si = d->sets.Find(b.set);
        if (!si || si->stereoImages)
        {
            ++d->statScreenTriRefused;
            // Named once per pipeline. The device lock is held shared here, so the flag has its own lock.
            static std::mutex s_named;
            std::lock_guard<std::mutex> named(s_named);
            auto pit = d->pipelines.find(s->gfx.pipeline);
            if (pit != d->pipelines.end() && !pit->second.triRefusedNamed)
            {
                pit->second.triRefusedNamed = true;
                Log("[screen] screen triangle VS %08X FS %08X stays on its shadow set: the set holds a doubled image\n", pit->second.vsCrc, pit->second.fsCrc);
            }
            return false;
        }
        ++d->statScreenTriLeft;
    }
    const VkDescriptorSet set = left ? b.set : it->second;
    d->CmdBindDescriptorSets(s->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, b.layout, slot, 1, &set, (uint32_t)b.dyn.size(), b.dyn.empty() ? nullptr : b.dyn.data());
    if (b.layout != s->eyeLayout) s->eyeLayout = VK_NULL_HANDLE;
    return true;
}

static void ReplayOne(DeviceData* d, CmdState* s, const Rec& r, const uint8_t* data)
{
    if (Vk().VKIndirectFromLeft && Vk().VKBufferTwins) NoteReplayBind(s, r, data);
    const VkCommandBuffer cb = s->cb;
    // screenTriangles: a 3-vertex draw positioned through a camera block covers the screen, but its bytes may be
    // another draw's patched claim in the shared copy: the right eye binds the game's own set, the unpatched camera.
    DrawKind kind = DrawKind::World;
    uint32_t leftSlot = UINT32_MAX;
    if (r.op == Op::Draw || r.op == Op::DrawIndexed)
    {
        const CmdState::PipelineFacts* f = nullptr;
        if (Traits().screenTriangles && r.u[0] <= 6) PipelineLayoutFor(d, s, s->gfx.pipeline, nullptr, &f);
        kind = KindOfDraw(f && f->worldQuad, f && f->posUsesMatrix, r.u[0], r.u[1]);
        if (kind == DrawKind::ScreenTriangle && ScreenTriangleSet(d, s, f->posSet, true)) leftSlot = f->posSet;
    }
    switch (r.op)
    {
    case Op::BindPipeline:
        d->CmdBindPipeline(cb, (VkPipelineBindPoint)r.u[0], (VkPipeline)r.h[0]);
        if (r.u[0] == VK_PIPELINE_BIND_POINT_GRAPHICS) { s->gfx.pipeline = (VkPipeline)r.h[0]; s->gfx.layout = (VkPipelineLayout)r.h[1]; PipelineLayoutFor(d, s, s->gfx.pipeline, &s->gfxShifted); s->lampPipe = Vk().VKFsCameraWindows && LampPipeline(d, s->gfx.pipeline); }
        break;
    case Op::BindSets:
    {
        // Compute binds made inside a pass serve the dispatch after it: the right eye's would stay bound for the
        // game's own (left) dispatch, and the right run binds its own.
        if (r.u[0] == VK_PIPELINE_BIND_POINT_COMPUTE) break;
        const uint32_t count = r.u[2], dyn = r.u[3];
        VkDescriptorSet sets[32];
        memcpy(sets, data, count * sizeof(VkDescriptorSet));
        // screenTriangles: each slot's own dynamic offsets, split by the layout's per-set counts as the bind hook does.
        const CmdState::LayoutFacts* lf = (Traits().screenTriangles && r.u[0] == VK_PIPELINE_BIND_POINT_GRAPHICS) ? LayoutFactsFor(d, s, (VkPipelineLayout)r.h[0]) : nullptr;
        const uint32_t* od = (const uint32_t*)(data + count * sizeof(VkDescriptorSet));
        uint32_t usedDyn = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            if (r.u[0] == VK_PIPELINE_BIND_POINT_GRAPHICS && r.u[1] + i < 8)
            {
                const uint32_t slot = r.u[1] + i;
                s->rGameSet[slot] = sets[i]; s->rGameSetLayout[slot] = (VkPipelineLayout)r.h[0];
                const uint32_t n = (lf && slot < lf->dyn.size()) ? lf->dyn[slot] : 0;
                s->rBindKnown[slot] = lf && usedDyn + n <= dyn;
                if (s->rBindKnown[slot])
                {
                    s->rBind[slot].set = sets[i]; s->rBind[slot].layout = (VkPipelineLayout)r.h[0];
                    s->rBind[slot].dyn.assign(od + usedDyn, od + usedDyn + n);
                }
                usedDyn += n;
            }
            auto it = s->replayShadow.find(sets[i]);
            if (it != s->replayShadow.end() && it->second) sets[i] = it->second;
        }
        const uint32_t* pDyn = (const uint32_t*)(data + count * sizeof(VkDescriptorSet));
        d->CmdBindDescriptorSets(cb, (VkPipelineBindPoint)r.u[0], (VkPipelineLayout)r.h[0], r.u[1], count, sets, dyn, dyn ? pDyn : nullptr);
        if (r.u[0] == VK_PIPELINE_BIND_POINT_GRAPHICS && (VkPipelineLayout)r.h[0] != s->eyeLayout) s->eyeLayout = VK_NULL_HANDLE;
        break;
    }
    case Op::BindIndex:  d->CmdBindIndexBuffer(cb, RightBuf(d, (VkBuffer)r.h[0]), r.h[1], (VkIndexType)r.u[0]); break;
    case Op::BindIndex2:
        if (d->CmdBindIndexBuffer2) d->CmdBindIndexBuffer2(cb, RightBuf(d, (VkBuffer)r.h[0]), r.h[1], r.h[2], (VkIndexType)r.u[0]);
        else d->CmdBindIndexBuffer(cb, RightBuf(d, (VkBuffer)r.h[0]), r.h[1], (VkIndexType)r.u[0]);
        break;
    case Op::DynState: IssueDyn(d, cb, (Dyn)r.u[0], r.u[1], data, r.dataLen); break;
    case Op::BindVertex:
    {
        std::vector<VkBuffer> bufs((const VkBuffer*)data, (const VkBuffer*)data + r.u[1]);
        for (auto& b : bufs) b = RightBuf(d, b);
        d->CmdBindVertexBuffers(cb, r.u[0], r.u[1], bufs.data(), (const VkDeviceSize*)(data + r.u[1] * sizeof(VkBuffer)));
        break;
    }
    case Op::BindVertex2:
    {
        const uint32_t count = r.u[1];
        std::vector<VkBuffer> right((const VkBuffer*)data, (const VkBuffer*)data + count);
        for (auto& b : right) b = RightBuf(d, b);
        const VkBuffer* bufs = right.data();
        const auto* offs = (const VkDeviceSize*)(data + count * sizeof(VkBuffer));
        const VkDeviceSize* sizes = offs + count;
        const VkDeviceSize* strides = sizes + count;
        if (!d->CmdBindVertexBuffers2) { d->CmdBindVertexBuffers(cb, r.u[0], count, bufs, offs); break; }
        // Null back where the game passed null: the defaults were serialised,
        // but the two are not interchangeable for every implementation.
        d->CmdBindVertexBuffers2(cb, r.u[0], count, bufs, (r.u[2] & 1) ? offs : nullptr,
                                 (r.u[2] & 2) ? sizes : nullptr, (r.u[2] & 4) ? strides : nullptr);
        break;
    }
    case Op::PushConstants: d->CmdPushConstants(cb, (VkPipelineLayout)r.h[0], r.u[0], r.u[1], r.u[2], data); break;
    case Op::Viewport: d->CmdSetViewport(cb, r.u[0], r.u[1], (const VkViewport*)data); break;
    case Op::Scissor:
        d->CmdSetScissor(cb, r.u[0], r.u[1], (const VkRect2D*)data);
        if (Traits().portalScissor) { s->gfx.scissors.assign((const VkRect2D*)data, (const VkRect2D*)data + r.u[1]); s->gfx.scissorFirst = r.u[0]; s->portalWide = false; }
        break;
    case Op::LineWidth: d->CmdSetLineWidth(cb, r.f[0]); break;
    case Op::DepthBias: d->CmdSetDepthBias(cb, r.f[0], r.f[1], r.f[2]); break;
    case Op::BlendConstants: d->CmdSetBlendConstants(cb, r.f); break;
    case Op::DepthBounds: d->CmdSetDepthBounds(cb, r.f[0], r.f[1]); s->replayDb[0] = r.f[0]; s->replayDb[1] = r.f[1]; break;
    case Op::StencilCompare: d->CmdSetStencilCompareMask(cb, r.u[0], r.u[1]); break;
    case Op::StencilWrite: d->CmdSetStencilWriteMask(cb, r.u[0], r.u[1]); break;
    case Op::StencilRef: d->CmdSetStencilReference(cb, r.u[0], r.u[1]); break;
    case Op::CondBegin:
    {
        VkConditionalRenderingBeginInfoEXT ci = { VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT };
        ci.buffer = (VkBuffer)r.h[0];
        ci.offset = r.h[1];
        ci.flags = r.u[0];
        d->CmdBeginConditionalRenderingEXT(cb, &ci);
        break;
    }
    case Op::CondEnd: d->CmdEndConditionalRenderingEXT(cb); break;
    case Op::QueryBegin:
    {
        // Each twin slot is begun at most once per reset (a pass replayed again keeps its first count).
        VkQueryPool twin = VK_NULL_HANDLE;
        {
            std::shared_lock<std::shared_mutex> g(d->lock);
            auto it = d->occTwins.find((VkQueryPool)r.h[0]);
            if (it != d->occTwins.end() && r.u[0] < it->second.count && !s->rQueryOpen && !it->second.begun[r.u[0]].exchange(1))
                twin = it->second.twin;
        }
        if (!twin) break;
        d->CmdBeginQuery(cb, twin, r.u[0], r.u[1]);
        s->rQueryOpen = twin; s->rQueryIdx = r.u[0];
        ++d->statOccRight;
        break;
    }
    case Op::QueryEnd:
        if (s->rQueryOpen) { d->CmdEndQuery(cb, s->rQueryOpen, s->rQueryIdx); s->rQueryOpen = VK_NULL_HANDLE; }
        break;
    case Op::Draw: case Op::DrawIndexed:
    {
        // A listed depth-only occluder (SkippedOccluderVS) is not replayed: no colour needs it, and its right-eye depth
        // could fail every draw behind it.
        if (r.op == Op::Draw && Vk().VKPushDeltas && r.u[0] > 6)
        {
            std::shared_lock<std::shared_mutex> gl(d->lock);
            auto pit = d->pipelines.find(s->gfx.pipeline);
            if (pit != d->pipelines.end() && SkippedOccluderVS(pit->second.vsCrc)) break;
        }
        // A screen pass is pinned and never claimed (KindOfDraw); world quads are neither. `screen` is also the UI
        // masks' <=6 test.
        const bool screen = kind != DrawKind::World;
        EnsureEyeSet(d, s, (screen && SmallDrawUnshift(d, r.u[0], r.u[1])) || ScreenDepthDraw(d, s) || NoDepthUnshift(d, s) || s->shadGtHud || (s->shadGtMask && screen), screen);
        // VKIndirectFromLeft=2: particles drawn indirectly in one scene may be drawn directly in the next; only
        // non-indexed instanced draws of a learned VS take the left's buffers.
        bool fromLeft = false;
        if (Vk().VKIndirectFromLeft >= 2 && Vk().VKBufferTwins && r.op == Op::Draw && r.u[1] > 1)
        {
            std::shared_lock<std::shared_mutex> g(d->lock);
            auto it = d->pipelines.find(s->gfx.pipeline);
            fromLeft = it != d->pipelines.end() && it->second.vsCrc && d->fromLeftVs.count(it->second.vsCrc);
        }
        if (fromLeft) { ReissueReplayBinds(d, s, true); std::lock_guard<std::shared_mutex> g(d->lock); ++d->statDirectFromLeft; }
        if (Traits().portalScissor) RsxPortalScissor(d, s, cb, r.u[0]);
        LampDraw(d, s, true);
        if (r.op == Op::Draw) d->CmdDraw(cb, r.u[0], r.u[1], r.u[2], r.u[3]);
        else d->CmdDrawIndexed(cb, r.u[0], r.u[1], r.u[2], (int32_t)r.u[3], r.u[4]);
        LampDraw(d, s, false);
        if (fromLeft) ReissueReplayBinds(d, s, false);
        if (leftSlot != UINT32_MAX) ScreenTriangleSet(d, s, leftSlot, false);
        ++d->statReplayDraws;
        break;
    }
    case Op::DrawIndirect: case Op::DrawIndexedIndirect: case Op::DrawIndirectCount: case Op::DrawIndexedIndirectCount:
    {
        EnsureEyeSet(d, s, s->shadGtHud);
        // VKIndirectFromLeft: GPU-written draws (particles) take the left's simulation; the right eye only moves them.
        const bool fromLeft = Vk().VKIndirectFromLeft && Vk().VKBufferTwins;
        if (fromLeft) { ReissueReplayBinds(d, s, true); ++d->statIndirectFromLeft; }
        if (fromLeft && Vk().VKIndirectFromLeft >= 2)
        {
            std::lock_guard<std::shared_mutex> g(d->lock);
            auto it = d->pipelines.find(s->gfx.pipeline);
            if (it != d->pipelines.end() && it->second.vsCrc) d->fromLeftVs.insert(it->second.vsCrc);
            // Particles bound through a set, which ReissueReplayBinds cannot reach: the set's next write keeps the
            // left's buffers, so the left's args meet the left's particles.
            if (Vk().VKPushDeltas && it != d->pipelines.end())
                for (uint32_t i = 0; i < 8; ++i)
                {
                    if (!s->rGameSet[i] || s->rGameSetLayout[i] != s->gfx.layout) continue;
                    const DescriptorSetInfo* st = d->sets.Find(s->rGameSet[i]);
                    if (st && d->leftBufferSetLayouts.insert(st->layout).second)
                        Log("[leftset] VS %08X FS %08X: set %u keeps the left's buffers in the right eye\n", it->second.vsCrc, it->second.fsCrc, i);
                }
        }
        const VkBuffer args = fromLeft ? (VkBuffer)r.h[0] : RightBuf(d, (VkBuffer)r.h[0]);
        const VkBuffer count = fromLeft ? (VkBuffer)r.h[2] : RightBuf(d, (VkBuffer)r.h[2]);
        if (r.op == Op::DrawIndirect) d->CmdDrawIndirect(cb, args, r.h[1], r.u[0], r.u[1]);
        else if (r.op == Op::DrawIndexedIndirect) d->CmdDrawIndexedIndirect(cb, args, r.h[1], r.u[0], r.u[1]);
        else if (r.op == Op::DrawIndirectCount) d->CmdDrawIndirectCount(cb, args, r.h[1], count, r.h[3], r.u[0], r.u[1]);
        else d->CmdDrawIndexedIndirectCount(cb, args, r.h[1], count, r.h[3], r.u[0], r.u[1]);
        if (fromLeft) ReissueReplayBinds(d, s, false);
        ++d->statReplayDraws;
        break;
    }
    case Op::DrawMeshTasks: case Op::DrawMeshTasksIndirect: case Op::DrawMeshTasksIndirectCount:
    {
        // The task/mesh stages read the left's buffers (culling lists included); the eye's set moves the mesh Position.
        EnsureEyeSet(d, s);
        if (r.op == Op::DrawMeshTasks) d->CmdDrawMeshTasksEXT(cb, r.u[0], r.u[1], r.u[2]);
        else if (r.op == Op::DrawMeshTasksIndirect) d->CmdDrawMeshTasksIndirectEXT(cb, RightBuf(d, (VkBuffer)r.h[0]), r.h[1], r.u[0], r.u[1]);
        else d->CmdDrawMeshTasksIndirectCountEXT(cb, RightBuf(d, (VkBuffer)r.h[0]), r.h[1], RightBuf(d, (VkBuffer)r.h[2]), r.h[3], r.u[0], r.u[1]);
        ++d->statReplayDraws;
        break;
    }
    case Op::ClearAttachments:
        d->CmdClearAttachments(cb, r.u[0], (const VkClearAttachment*)data, r.u[1], (const VkClearRect*)(data + r.u[0] * sizeof(VkClearAttachment)));
        break;
    case Op::NextSubpass: d->CmdNextSubpass(cb, (VkSubpassContents)r.u[0]); break;
    case Op::PushDescriptors:
    {
        if (r.u[0] == VK_PIPELINE_BIND_POINT_COMPUTE) break;   // as BindSets: the left's dispatch keeps the left's
        std::vector<uint8_t> blob(data, data + r.dataLen);
        IssuePushes(d, cb, (VkPipelineBindPoint)r.u[0], (VkPipelineLayout)r.h[0], r.u[1], blob, true);
        if (r.u[0] == VK_PIPELINE_BIND_POINT_GRAPHICS && (VkPipelineLayout)r.h[0] != s->eyeLayout) s->eyeLayout = VK_NULL_HANDLE;
        break;
    }
    case Op::Barrier:
    {
        const uint32_t mc = r.u[3], bc = r.u[4], ic = r.u[5];
        const VkMemoryBarrier* pM = (const VkMemoryBarrier*)data;
        std::vector<VkBufferMemoryBarrier> buf((const VkBufferMemoryBarrier*)(data + mc * sizeof(VkMemoryBarrier)),
                                               (const VkBufferMemoryBarrier*)(data + mc * sizeof(VkMemoryBarrier)) + bc);
        std::vector<VkImageMemoryBarrier> img((const VkImageMemoryBarrier*)(data + mc * sizeof(VkMemoryBarrier) + bc * sizeof(VkBufferMemoryBarrier)),
                                              (const VkImageMemoryBarrier*)(data + mc * sizeof(VkMemoryBarrier) + bc * sizeof(VkBufferMemoryBarrier)) + ic);
        {
            std::shared_lock<std::shared_mutex> g(d->lock);
            for (auto& b : img) { VkImage sib; if (ImageIsStereo(d, b.image, &sib)) b.image = sib; }
            if (Vk().VKBufferTwins) for (auto& b : buf) if (const VkBuffer t = BufferTwin(d, b.buffer)) b.buffer = t;
        }
        d->CmdPipelineBarrier(cb, r.u[0], r.u[1], r.u[2], mc, pM, bc, buf.data(), ic, img.data());
        break;
    }
    case Op::Barrier2:
    {
        const uint32_t mc = r.u[1], bc = r.u[2], ic = r.u[3];
        std::vector<VkMemoryBarrier2> mb(mc);
        std::vector<VkBufferMemoryBarrier2> bb(bc);
        std::vector<VkImageMemoryBarrier2> ib(ic);
        const uint8_t* q = data;
        if (mc) memcpy(mb.data(), q, mc * sizeof(VkMemoryBarrier2));
        q += mc * sizeof(VkMemoryBarrier2);
        if (bc) memcpy(bb.data(), q, bc * sizeof(VkBufferMemoryBarrier2));
        q += bc * sizeof(VkBufferMemoryBarrier2);
        if (ic) memcpy(ib.data(), q, ic * sizeof(VkImageMemoryBarrier2));
        {
            std::shared_lock<std::shared_mutex> g(d->lock);
            for (auto& b : ib) { VkImage sib; if (ImageIsStereo(d, b.image, &sib)) b.image = sib; }
            if (Vk().VKBufferTwins) for (auto& b : bb) if (const VkBuffer t = BufferTwin(d, b.buffer)) b.buffer = t;
        }
        VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        di.dependencyFlags = r.u[0];
        di.memoryBarrierCount = mc; di.pMemoryBarriers = mb.data();
        di.bufferMemoryBarrierCount = bc; di.pBufferMemoryBarriers = bb.data();
        di.imageMemoryBarrierCount = ic; di.pImageMemoryBarriers = ib.data();
        d->CmdPipelineBarrier2(cb, &di);
        break;
    }
    case Op::Execute:
    {
        // Inline the secondary's recording; a secondary starts from no state.
        const VkCommandBuffer* cbs = (const VkCommandBuffer*)data;
        for (uint32_t i = 0; i < r.u[0]; ++i)
        {
            CmdState* sec = nullptr;
            { std::shared_lock<std::shared_mutex> g(d->lock); auto it = d->cmdStates.find(cbs[i]); if (it != d->cmdStates.end()) sec = it->second; }
            if (!sec) continue;
            std::vector<Rec> recs = sec->recs;
            std::vector<uint8_t> blob = sec->blob;
            s->eyeLayout = VK_NULL_HANDLE;
            ReplayRecords(d, s, recs, blob);
        }
        break;
    }
    }
}

static void ReplayRecords(DeviceData* d, CmdState* s, const std::vector<Rec>& recs, const std::vector<uint8_t>& blob)
{
    for (const Rec& r : recs) ReplayOne(d, s, r, blob.data() + r.dataOff);
}

// Every set the coming replay will bind, resolved to its shadow in ONE lock
// acquisition. The counters are per distinct set now, not per bind.
static void ResolvePassShadows(DeviceData* d, CmdState* s, const std::vector<Rec>& recs, const std::vector<uint8_t>& blob)
{
    s->replayShadow.clear();
    for (uint32_t i = 0; i < 32; ++i)
        if (s->passStart.sets[i].set) s->replayShadow.emplace(s->passStart.sets[i].set, VK_NULL_HANDLE);
    for (const Rec& r : recs)
    {
        if (r.op != Op::BindSets) continue;
        const uint32_t count = r.u[2];
        if (!count || r.dataLen < count * sizeof(VkDescriptorSet)) continue;
        const VkDescriptorSet* sets = (const VkDescriptorSet*)(blob.data() + r.dataOff);
        for (uint32_t i = 0; i < count; ++i)
            if (sets[i]) s->replayShadow.emplace(sets[i], VK_NULL_HANDLE);
    }
    if (s->replayShadow.empty()) return;
    unsigned shadowed = 0, unshadowed = 0, unknown = 0;
    std::shared_lock<std::shared_mutex> rd(d->lock, std::defer_lock);
    std::unique_lock<std::shared_mutex> ex(d->lock, std::defer_lock);
    if (Traits().sharedCollect) rd.lock(); else ex.lock();   // set lookups only (ShadowSetFor)
    for (auto& kv : s->replayShadow)
    {
        VkDescriptorSet sh = ShadowSetFor(d, kv.first);
        if (sh) { kv.second = sh; ++shadowed; }
        else if (d->sets.Find(kv.first)) ++unshadowed;
        else ++unknown;
    }
    d->statReplayShadowed += shadowed;
    d->statReplayUnshadowed += unshadowed;
    d->statReplayUnknownSet += unknown;
}

// After a mixed (mono) pass: the left eye's result into each doubled attachment's sibling, which would
// otherwise stay unwritten (a light-space map is the same for both eyes).
static void CopyMixedToSiblings(DeviceData* d, CmdState* s)
{
    const VkCommandBuffer cb = s->cb;
    for (const auto& m : s->mixedCopies)
    {
        const bool general = m.layout == VK_IMAGE_LAYOUT_GENERAL;
        const VkImageLayout srcL = general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        const VkImageLayout dstL = general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        VkImageMemoryBarrier b[2] = {};
        for (int i = 0; i < 2; ++i)
        {
            b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b[i].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            b[i].dstAccessMask = i == 0 ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
            b[i].oldLayout = i == 0 ? m.layout : m.siblingFrom;   // the sibling sat the pass out
            b[i].newLayout = i == 0 ? srcL : dstL;
            b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b[i].image = i == 0 ? m.image : m.sibling;
            b[i].subresourceRange = m.range;
        }
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, b);
        std::vector<VkImageCopy> regions;
        for (uint32_t mip = 0; mip < m.range.levelCount; ++mip)
        {
            const uint32_t level = m.range.baseMipLevel + mip;
            VkImageCopy r = {};
            r.srcSubresource = { m.range.aspectMask, level, m.range.baseArrayLayer, m.range.layerCount };
            r.dstSubresource = r.srcSubresource;
            r.extent = { (m.w >> level) ? (m.w >> level) : 1, (m.h >> level) ? (m.h >> level) : 1, 1 };
            regions.push_back(r);
        }
        d->CmdCopyImage(cb, m.image, srcL, m.sibling, dstL, (uint32_t)regions.size(), regions.data());
        for (int i = 0; i < 2; ++i)
        {
            b[i].srcAccessMask = i == 0 ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
            b[i].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            b[i].oldLayout = b[i].newLayout;
            b[i].newLayout = m.layout;
        }
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, b);
    }
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (const auto& m : s->mixedCopies) MarkSiblingWritten(d, m.image);
    }
    s->mixedCopies.clear();
}

static void EndPass(DeviceData* d, CmdState* s)
{
    const bool dyn = s->dynPass;
    s->dynPass = false;
    if (!s->inPass) return;
    const bool stereo = s->stereoPass;
    s->inPass = false;
    s->stereoPass = false;
    if (!s->mixedCopies.empty()) CopyMixedToSiblings(d, s);
    if (!stereo) return;

    // The pass again, into the sibling framebuffer or sibling views, for the right eye.
    PerfScope pt(d->perf.replay);
    const VkCommandBuffer cb = s->cb;
    std::vector<Rec> recs;
    std::vector<uint8_t> blob;
    recs.swap(s->recs);
    blob.swap(s->blob);
    const GfxState after = s->gfx;

    ResolvePassShadows(d, s, recs, blob);
    s->replaying = true;
    s->gfx = s->passStart;
    s->portalWide = false;
    if (Vk().VKIndirectFromLeft && Vk().VKBufferTwins)
    {
        // The binds the replay starts with: the pass's starting state, which Reestablish issues below.
        const GfxState& g = s->passStart;
        for (uint32_t i = 0; i < 8; ++i) { s->rPush[i] = g.pushSets[i]; s->rPushLayout[i] = g.pushLayout[i]; }
        s->rIb = g.hasIndex ? g.indexBuf : VK_NULL_HANDLE;
        s->rIbOff = g.indexOff; s->rIbSize = g.indexSize; s->rIbType = g.indexType; s->rIbTwo = g.index2;
        memcpy(s->rVb, g.vb, sizeof(s->rVb)); memcpy(s->rVbOff, g.vbOff, sizeof(s->rVbOff));
        memcpy(s->rVbSize, g.vbSize, sizeof(s->rVbSize)); memcpy(s->rVbStride, g.vbStride, sizeof(s->rVbStride));
        s->rVbMask = g.vbMask; s->rVb2Mask = g.vb2Mask; s->rVbStrideMask = g.vbStrideMask;
    }
    if (dyn)
    {
        VkRenderingInfo ri = s->dynInfo;
        ri.pColorAttachments = s->dynColor.empty() ? nullptr : s->dynColor.data();
        ri.pDepthAttachment = s->dynHasDepth ? &s->dynDepth : nullptr;
        ri.pStencilAttachment = s->dynHasStencil ? &s->dynStencil : nullptr;
        d->CmdBeginRendering(cb, &ri);
    }
    else
        d->CmdBeginRenderPass(cb, &s->passBegin, VK_SUBPASS_CONTENTS_INLINE);
    Reestablish(d, s, s->passStart);
    ReplayRecords(d, s, recs, blob);
    // A query must end in the pass it began in: a recording that stopped mid-segment leaves the twin's open.
    if (s->rQueryOpen) { d->CmdEndQuery(cb, s->rQueryOpen, s->rQueryIdx); s->rQueryOpen = VK_NULL_HANDLE; }
    if (dyn) d->CmdEndRendering(cb);
    else d->CmdEndRenderPass(cb);
    s->replaying = false;

    // Put the left-eye bindings back so the game continues from its own state.
    s->gfx = after;
    Reestablish(d, s, after);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdEndRenderPass(VkCommandBuffer cb)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdEndRenderPass(cb);
    EndPass(d, s);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo* pEnd)
{
    DeviceData* d = FindDevice(cb); CmdState* s = State(d, cb);
    d->CmdEndRenderPass2(cb, pEnd);
    EndPass(d, s);
}

// Dynamic rendering (core 1.3, VKDynamicRendering)

// The render-pass unit with sibling views for a sibling framebuffer, under Hook_CreateFramebuffer's mixed-pass
// rules; pNext chains are dropped from the right eye's begin.
static void BeginDynamicPass(DeviceData* d, CmdState* s, const VkRenderingInfo* p)
{
    s->inPass = true;
    s->dynPass = true;
    s->stereoPass = false;
    s->passSibling = VK_NULL_HANDLE;
    s->passAspectOk = true;
    s->passW = s->passH = 0;
    s->shadGtHud = s->shadGtMask = false;
    if (s->secondary || s->foreign || t_foreignCalls) return;
    s->shadGtHud = ShadGtHudPass(d, s, p);
    ++d->statDynRendering;
    // A suspended instance continues in the next command buffer submitted; a replay between the halves would split it.
    if (p->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)) { ++d->statDynSuspended; ++d->statMonoPasses; return; }
    s->dynColor.assign(p->pColorAttachments, p->pColorAttachments + p->colorAttachmentCount);
    s->dynHasDepth = p->pDepthAttachment != nullptr;
    s->dynHasStencil = p->pStencilAttachment != nullptr;
    s->dynDepth = s->dynHasDepth ? *p->pDepthAttachment : VkRenderingAttachmentInfo{};
    s->dynStencil = s->dynHasStencil ? *p->pStencilAttachment : VkRenderingAttachmentInfo{};
    bool any = false, chained = p->pNext != nullptr;
    uint32_t missing = 0, att = 0;
    std::vector<VkImage> doubled;
    std::vector<CmdState::MixedCopy> copies;
    s->mixedCopies.clear();
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto swap = [&](VkImageView& v, VkImageLayout layout)
        {
            if (!v) return;
            ++att;
            auto vit = d->views.find(v);
            VkImageView sv;
            if (ViewSibling(d, v, &sv))
            {
                v = sv;
                any = true;
                if (vit == d->views.end()) return;
                doubled.push_back(vit->second.image);
                auto ii = d->images.find(vit->second.image);
                bool seen = false;   // depth and stencil usually share one view
                for (const auto& c : copies)
                    seen |= c.image == vit->second.image && c.range.baseMipLevel == vit->second.range.baseMipLevel &&
                            c.range.baseArrayLayer == vit->second.range.baseArrayLayer;
                if (!seen && ii != d->images.end() && ii->second.sibling)
                    copies.push_back({ vit->second.image, ii->second.sibling, vit->second.range, layout,
                                       ii->second.ci.extent.width, ii->second.ci.extent.height, layout });
                return;
            }
            auto iit = vit != d->views.end() ? d->images.find(vit->second.image) : d->images.end();
            if (iit == d->images.end() || !FamiliesSharedDepth(iit->second.ci, d->bbSize)) ++missing;
        };
        auto attach = [&](VkRenderingAttachmentInfo& a)
        {
            chained |= a.pNext != nullptr;
            a.pNext = nullptr;
            swap(a.imageView, a.imageLayout);
            if (a.resolveMode != VK_RESOLVE_MODE_NONE) swap(a.resolveImageView, a.resolveImageLayout);
        };
        for (auto& a : s->dynColor) attach(a);
        if (s->dynHasDepth) attach(s->dynDepth);
        if (s->dynHasStencil) attach(s->dynStencil);
        if (any && missing && Vk().VKMixedFramebufferMono)
        {
            any = false;
            s->mixedCopies.swap(copies);
        }
        if (any)
            for (VkImage img : doubled)
            {
                auto ii = d->images.find(img);
                if (ii == d->images.end()) continue;
                ii->second.everWritten = ii->second.siblingWritten = true;
                if (!s->passW) { s->passW = ii->second.ci.extent.width; s->passH = ii->second.ci.extent.height; }
            }
    }
    if (!any && !doubled.empty())
    {
        ++d->statDynMixed;
        static std::atomic<int> s_log{16};
        if (s_log.fetch_sub(1, std::memory_order_relaxed) > 0)
            Log("[dyn] mixed pass: %u of %u attachments undoubled, pass stays mono, %u doubled copied to siblings at its end (area %ux%u)\n",
                missing, att, (unsigned)s->mixedCopies.size(), p->renderArea.extent.width, p->renderArea.extent.height);
    }
    if (!any) { ++d->statDynMono; ++d->statMonoPasses; return; }
    if (chained)
    {
        static std::atomic<bool> s_said{false};
        if (!s_said.exchange(true)) Log("[dyn] a VkRenderingInfo/attachment pNext chain is dropped from the right eye's begin\n");
    }
    NotePassAspect(d, s);
    s->stereoPass = true;
    s->dynInfo = *p;
    s->dynInfo.pNext = nullptr;
    s->dynInfo.flags &= ~VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;   // the replay inlines secondaries
    s->passStart = s->gfx;
    s->recs.clear();
    s->blob.clear();
    ++d->statStereoPasses;
    ++d->statDynStereo;
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* pInfo)
{
    DeviceData* d = FindDevice(cb);
    d->CmdBeginRendering(cb, pInfo);
    if (!d->stereoEnabled || !pInfo) return;
    if (!Vk().VKDynamicRendering) { ++d->statDynRendering; return; }
    BeginDynamicPass(d, State(d, cb), pInfo);
}

static VKAPI_ATTR void VKAPI_CALL Hook_CmdEndRendering(VkCommandBuffer cb)
{
    DeviceData* d = FindDevice(cb);
    d->CmdEndRendering(cb);
    if (Vk().VKDynamicRendering) EndPass(d, State(d, cb));
}

const HookEntry g_commandHooks[] = {
    WIZ3D_HOOK(AllocateCommandBuffers), WIZ3D_HOOK(FreeCommandBuffers),
    WIZ3D_HOOK(ResetCommandPool), WIZ3D_HOOK(DestroyCommandPool),
    WIZ3D_HOOK(BeginCommandBuffer), WIZ3D_HOOK(EndCommandBuffer), WIZ3D_HOOK(ResetCommandBuffer),
    WIZ3D_HOOK(CmdBindPipeline), WIZ3D_HOOK(CmdBindDescriptorSets),
    WIZ3D_HOOK(CmdBindIndexBuffer), WIZ3D_HOOK(CmdBindIndexBuffer2), WIZ3D_HOOK(CmdBindVertexBuffers), WIZ3D_HOOK(CmdBindVertexBuffers2), WIZ3D_HOOK(CmdPushConstants),
    WIZ3D_HOOK(CmdSetViewport), WIZ3D_HOOK(CmdSetScissor), WIZ3D_HOOK(CmdSetLineWidth),
    WIZ3D_HOOK(CmdSetDepthBias), WIZ3D_HOOK(CmdSetBlendConstants), WIZ3D_HOOK(CmdSetDepthBounds),
    WIZ3D_HOOK(CmdSetStencilCompareMask), WIZ3D_HOOK(CmdSetStencilWriteMask), WIZ3D_HOOK(CmdSetStencilReference),
#define X(Fn, Name, T) WIZ3D_HOOK(Fn),
    WIZ3D_DYN_SCALARS(X) WIZ3D_DYN_RANGES(X) WIZ3D_DYN_COUNTED(X)
#undef X
    WIZ3D_HOOK(CmdSetStencilOp), WIZ3D_HOOK(CmdSetSampleMaskEXT), WIZ3D_HOOK(CmdSetExtraPrimitiveOverestimationSizeEXT),
    WIZ3D_HOOK(CmdSetVertexInputEXT),
    WIZ3D_HOOK(CmdDraw), WIZ3D_HOOK(CmdDrawIndexed), WIZ3D_HOOK(CmdDrawMultiEXT), WIZ3D_HOOK(CmdDrawMultiIndexedEXT),
    WIZ3D_HOOK(CmdBeginConditionalRenderingEXT), WIZ3D_HOOK(CmdEndConditionalRenderingEXT),
    WIZ3D_HOOK(CmdDrawIndirect), WIZ3D_HOOK(CmdDrawIndexedIndirect),
    WIZ3D_HOOK(CmdDrawIndirectCount), WIZ3D_HOOK(CmdDrawIndexedIndirectCount),
    WIZ3D_HOOK(CmdDrawMeshTasksEXT), WIZ3D_HOOK(CmdDrawMeshTasksIndirectEXT), WIZ3D_HOOK(CmdDrawMeshTasksIndirectCountEXT),
    WIZ3D_HOOK(CmdClearAttachments),
    WIZ3D_HOOK(CmdDispatch), WIZ3D_HOOK(CmdDispatchIndirect),
    WIZ3D_HOOK(CmdBeginRendering), WIZ3D_HOOK(CmdEndRendering),
    WIZ3D_HOOK(CmdUpdateBuffer), WIZ3D_HOOK(CmdCopyBuffer), WIZ3D_HOOK(CmdCopyImage), WIZ3D_HOOK(CmdBlitImage), WIZ3D_HOOK(CmdResolveImage), WIZ3D_HOOK(CmdCopyBufferToImage),
    WIZ3D_HOOK(CmdCopyImage2), WIZ3D_HOOK(CmdBlitImage2), WIZ3D_HOOK(CmdResolveImage2), WIZ3D_HOOK(CmdCopyBufferToImage2),
    WIZ3D_HOOK(CmdCopyImageToBuffer), WIZ3D_HOOK(CmdCopyImageToBuffer2), WIZ3D_HOOK(CmdCopyBuffer2), WIZ3D_HOOK(CmdFillBuffer), WIZ3D_HOOK(CmdCopyQueryPoolResults),
    WIZ3D_HOOK(CreateQueryPool), WIZ3D_HOOK(DestroyQueryPool), WIZ3D_HOOK(ResetQueryPool), WIZ3D_HOOK(CmdResetQueryPool),
    WIZ3D_HOOK(CmdBeginQuery), WIZ3D_HOOK(CmdEndQuery), WIZ3D_HOOK(GetQueryPoolResults),
    WIZ3D_HOOK(CmdClearColorImage), WIZ3D_HOOK(CmdClearDepthStencilImage),
    WIZ3D_HOOK(CmdPipelineBarrier), WIZ3D_HOOK(CmdPipelineBarrier2),
    WIZ3D_HOOK(CmdBeginRenderPass), WIZ3D_HOOK(CmdBeginRenderPass2),
    WIZ3D_HOOK(CmdNextSubpass), WIZ3D_HOOK(CmdNextSubpass2),
    WIZ3D_HOOK(CmdEndRenderPass), WIZ3D_HOOK(CmdEndRenderPass2),
    WIZ3D_HOOK(CmdExecuteCommands),
    WIZ3D_HOOK(CmdPushDescriptorSetKHR), WIZ3D_HOOK(CmdPushDescriptorSetWithTemplateKHR),
    { "vkCmdPushDescriptorSet", (PFN_vkVoidFunction)&Hook_CmdPushDescriptorSetKHR },
    { "vkCmdPushDescriptorSetWithTemplate", (PFN_vkVoidFunction)&Hook_CmdPushDescriptorSetWithTemplateKHR },
    // LookupDeviceHook strips a trailing KHR, not EXT, and hosts may resolve the EXT name.
    { "vkCmdBindVertexBuffers2EXT", (PFN_vkVoidFunction)&Hook_CmdBindVertexBuffers2 },
    { nullptr, nullptr },
};

}} // namespace wiz3d::vk
