/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/
// Per-game Vulkan switches: <VKName Value="..."/> elements of the game's profile, read once before first use.
#pragma once
#include <windows.h>
#include <stdint.h>
#include <map>
#include <string>
#include <vector>

namespace wiz3d { namespace vk {

struct VkSwitches
{
    float VKSeparationScale       = 1.0f;   // sep scale (clip units per unit w) until a projection gives one
    bool  VKSkipUnitW             = true;   // vertices with w within VKUnitWEpsilon of 1 keep the left eye's x
    float VKNearClamp             = 0.0f;   // floor on w in the shift, as a fraction of convergence; 0 = none
    float VKViewModelClipZ        = 0.0f;   // <VKViewModel ClipZ>: clip z above it is the first-person weapon; 0 = none
    float VKViewModelSep          = 1.0f;   // ... Scale (or Sep): its sep scale
    float VKViewModelConv         = 1.0f;   // ... Scale (or Conv): its convergence scale
    bool  VKTileListsLeft         = false;  // FS tile lookups (FragCoord / tile size) read the left eye's tile lists
    bool  VKScreenMarch           = false;  // FS screen marches toward a left-camera projected point end at this eye's
    bool  VKDoubleStorageImages   = true;   // storage-only images (compute outputs) get a right-eye sibling
    int   VKSiblingBudget         = 0;      // cap on the number of siblings; 0 = unlimited
    int   VKUiTargetMaxPixels     = 0;      // off-aspect colour targets up to this many pixels stay mono; 0 = none
    bool  VKSmallDrawUnshift      = true;   // non-instanced replay draws of 6 vertices or fewer take the sep=0 set
    float VKEdgeCrop              = 0.0f;   // crop the edge strip the eye shift uncovers, as a multiple of it
    bool  VKGprPositionChain      = false;  // position chains follow register stores; 4 uniform registers make a matrix
    bool  VKServeFallback         = true;   // matrix-positioned shaders self-shift where the constants path missed
    bool  VKValidateInverse       = true;   // a learned inverse slot is patched only if it inverts this range's P
    bool  VKRepointLearnedSets    = true;   // sets written before their camera was learned move to patched siblings
    bool  VKEqualPassSelfShift    = true;   // EQUAL depth passes self-shift, so they match their prepass bit for bit
    bool  VKMaskMatrixSpan        = true;   // a claimed register anywhere in a matrix claims the whole matrix
    bool  VKDoubleArrayImages     = true;   // layered images get a right-eye sibling
    bool  VKMixedFramebufferMono  = true;   // a framebuffer with some attachments refused a sibling stays mono
    bool  VKMajorityP11           = true;   // the self-shift scale is the P11 most claimed ranges use
    bool  VKChainThroughBitcast   = true;   // the position-chain walk passes through bitcasts and integer ops
    bool  VKDynamicRendering      = false;  // dynamic rendering, extended dynamic state and "2" commands replay
    bool  VKBufferTwins           = false;  // GPU-only and sparse buffers get right-eye twins (StereoBuffers.cpp)
    bool  VKFsCameraWindows       = false;  // FS camera matrices loaded as raw dwords take the right eye's shift
    bool  VKScreenReadVaryings    = false;  // clip varyings shift only where the FS reads a texture at their x/w
    bool  VKDoubleTransferImages  = false;  // transfer-only images get a right-eye sibling
    bool  VKShiftPrevClipVaryings = false;  // previous-frame clip varyings shift like gl_Position
    bool  VKPushConstantOrder     = false;  // a replay restores push constants in the game's write order
    bool  VKShiftNdcVaryings      = false;  // screen-position varyings (xy / w) shift with gl_Position
    int   VKIndirectFromLeft      = 0;      // 1: right indirect draws read the left's buffers; 2: off-screen images too
    int   VKComputeRemap          = 0;      // right-eye camera for shaders rebuilding positions on the left one
    bool  VKDoubleFroxelVolumes   = false;  // backbuffer-aspect 3D storage images 16+ slices deep get siblings
    bool  VKLightRaysCamera       = false;  // light-volume FS rays move to the surface (needs VKShiftNdcVaryings)
    bool  VKShadowSetTwins        = false;  // the right eye's shadow descriptor sets take buffer twins
    bool  VKDynByPipeline         = false;  // a right-eye pass start re-issues only its pipeline's dynamic state
    bool  VKPushDeltas            = false;  // push descriptors accumulate per layout
    bool  VKPinLayer              = false;  // the layer stays loaded for the process's life
    bool  VKHideDescriptorBuffer  = false;  // VK_EXT_descriptor_buffer is hidden from the device's extensions
    bool  VKHideDepthBiasControl  = false;  // VK_EXT_depth_bias_control is hidden from the device's extensions
    bool  VKMonoPassCopies        = false;  // a mixed (mono) pass's result is copied into its doubled attachments
    bool  VKMeshDraws             = false;  // mesh-shader draws replay and the mesh stage shifts Position at its stores
    bool  VKScreenUvVaryings      = false;  // screen-uv varyings fma(x * (1/w), k, c) shift with gl_Position
    bool  VKFroxelLookups         = false;  // FS froxel-volume lookups read at the right eye's uv
    bool  VKFragCoordRays         = false;  // uv-ray rule for the pixel's own uv (needs VKComputeRemap 4)
    bool  VKFroxelCornerRays      = false;  // volume builders placing cells by corner rays: the right run's own frustum
    bool  VKDepthSprites          = false;  // w == 1 sprites tested against depth move by their own depth's parallax
    bool  VKOcclusionBothEyes     = false;  // occlusion queries count both eyes through twin pools
    float VKUnitWEpsilon          = 0.001f; // VKSkipUnitW's band around w = 1
    bool  VKStereoDLSS            = true;   // a DLSS-SR feature twin per eye at the NGX boundary (NgxVK.cpp)
    bool  VKWeaveGpuSync          = true;   // the SR weave's stages wait on the GPU (a shared D3D11 fence), not the CPU
    bool  VKFrameGen              = true;   // DLSS-G over Streamline: both eyes squeezed into one image (FrameGenVK.cpp)
    bool  VKShadowAtlasByShape    = true;   // shadow atlases are picked by shape, not area, and stay mono
    int   VKDumpModifiedShaders   = 0;      // 1: dump original and patched SPIR-V; 2: also the twins
};

extern VkSwitches g_vkSwitches;
inline const VkSwitches& Vk() { return g_vkSwitches; }

// Reads the game's profile once (S3DAPI ReadCurrentProfile) and fills g_vkSwitches; later calls do nothing.
void LoadVkSwitches();
// One row: build, profile, its families, and every switch that differs from its default.
void LogVkSwitches();
// <VKScreenDepth><Draw VertexShader="0x.."/>: draws of the listed vertex shaders always take the unshifted set (a HUD
// drawn with its own perspective).
const std::vector<uint32_t>& ScreenDepthShaders();

// A profile element's attributes, name -> value.
typedef std::map<std::string, std::string> ProfileEntry;
// One attribute, "" when absent.
const std::string& EntryValue(const ProfileEntry& entry, const char* attr);
// A profile element's child elements in order, each with its own name under "@".
std::vector<const ProfileEntry*> ChildEntries(const char* element);
// The shaders an element's <child> entries list, in order: each entry's VertexShader, DomainShader, PixelShader and
// ComputeShader attribute (0x hex).
std::vector<uint32_t> ChildShaders(const char* element, const char* child);

}} // namespace wiz3d::vk
