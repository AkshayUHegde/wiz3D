/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Engine families: what one engine needs beyond the shared layer, one VkFamily per profile element (<VKRage>).
// Adding one: a class in its own <Family>VK.cpp, plus its element in kFamilies (FamilyVK.cpp).

#pragma once

#include <stdint.h>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include "ProfileVK.h"

namespace wiz3d { namespace vk {

struct DeviceData;

// The camera tests' limits: IsBareProjection's largest P00 / P11, IsRayBasis's smallest column length,
// IsUnitCameraBasis's smallest vertical tan, and whether a projection with no inverse still takes the shift.
struct CameraLimits { float zoom = 6.f, tele = 0.2f, teleY = 0.05f; bool singularP = false; };

// What the shared layer asks of the families, fixed at the profile latch. A trait holds when any family sets it.
struct FamilyTraits
{
    CameraLimits limits;
    bool clipSlotStraddle   = false;  // a clip slot a claiming chain's matrix starts inside is refused
    bool claimedForward     = false;  // the camera forward comes from claimed, non-upright views, whose rotations vote
    bool previousProjection = false;  // an unclaimed P beside a claimed one (the previous frame's) is patched too
    bool unpatchedKeys      = false;  // a skip-claim screen draw's unclaimed position key reads the game's buffer
    bool claimMirror        = false;  // a claimed range another key mirrors unclaimed reads its own claim sibling
    bool sharedCollect      = false;  // set collection and set lookups read the device maps under a shared lock
    bool claimedKeysNeedP   = false;  // a key geometry claims learns only with a projection; a basis alone is evicted
    bool claimedKeyRetry    = false;  // a claimed position key is never given up: it retries once a second
    bool unitCameraBlock    = false;  // a block carrying its own origin (row 3) learns as a unit camera basis
    bool positionBindings   = false;  // a claim names the binding its draw positions through
    bool screenTriangles    = false;  // a 3-vertex screen draw positioned through a camera block binds the game's set
    bool offAspectPasses    = false;  // passes on targets not shaped like the backbuffer replay unshifted
    bool widthSets          = false;  // right-eye stereo sets per viewport width (StereoSetForWidth)
    bool matrixOverScalar   = false;  // a matrix use replaces a scalar use at the same uniform offset
    bool fragmentRules      = false;  // fragment modules go through the families' PatchFragment
    bool computeRules       = false;  // compute modules go through the families' PatchCompute (a twin each run binds)
    bool pinLayer           = false;  // the layer stays loaded for the process's life (instances come and go)
    bool bindTimeDescriptors = false; // update-after-bind is hidden, so descriptors are written at bind
    bool rsxPositionTwins   = false;  // FS rebuilding position from the pixel get a twin reading the left's (member 5)
    bool flatNoDepthDraws   = false;  // a depth-off draw is flat on the screen unless the frame shows it a world sprite
    bool portalScissor      = false;  // a small scissor that cut a depth hole widens for the right eye
    bool readColorBuffers   = false;  // targets the CPU reads back and reloads keep each eye's frame
    bool bufferTrips        = false;  // an image's trip through a buffer and back carries the right eye's bytes
    bool tripKernels        = false;  // compute between a trip's copies replays on the right bytes; new data ends it
    bool monoPassCopies     = false;  // a mono pass's doubled attachments get its result copied to the right image
    bool staticPipelineState = false; // state a pipeline holds static is not re-issued over it in the right eye
    bool flatUiPasses       = false;  // a single-colour pass with no depth into the display format is UI: no shift
    bool clockFrameRate     = false;  // a frame-rate row every 5 s by the clock, with the time of day
    bool velocityTwins      = false;  // scene FS take the stereo shift off their motion vectors (velocity twin)
    bool glassTwins         = false;  // refractive glass reads the scene behind it at this eye's pixel
    bool decalTwins         = false;  // box decals rebuild their surface from the left eye's pixel
    bool skyTwins           = false;  // the family's listed skies take the sky-ray twin (GtsSkyVS / GtsSkyFS)
    bool fusedMad           = false;  // FMul + FAdd pairs the translator split are fused back before the remap rules
    bool tileLists          = false;  // the listed tile-lighting chain runs per eye on its own lists (KytyTiles.cpp)
};

// One module rule's result: the code to create in place of the module (fragment) or beside it (compute twin).
struct FamilyPatch
{
    std::vector<uint32_t> code;
    const char* rule = "";          // names the rule's log row and counts its modules
    const char* dumpTag = nullptr;  // vk_<crc>_<tag>.spv under DumpModifiedShaders=2; null: not dumped
    bool dumpOriginal = false;      // the "other" dump beside it
    bool eyeOriginStats = false;    // counts into statFsEyeOrigin / statFsProducts
    unsigned products = 0;
    bool always = false;            // a single shader: its row is never throttled
    char row[224] = {};             // what the rule did, for the log
};

class VkFamily
{
public:
    virtual ~VkFamily() {}
    virtual const char* Tag() const = 0;                  // its log rows' tag
    virtual void SetTraits(FamilyTraits&) const {}
    virtual void LoadProfile() {}                         // reads its profile children, after every profile read

    // Pipelines: a vertex shader positioning through a camera block was created; a pipeline's vertex shader draws
    // world quads (its <=6-vertex draws are geometry); its cached right-eye sets are re-pointed once learned.
    virtual void OnPositionShader(uint32_t) const {}
    virtual bool WorldQuadVS(uint32_t) const { return false; }
    virtual bool RepointVS(uint32_t) const { return false; }

    // Module rules (pure: no device). PatchCompute's listed: the shader is the family's even if nothing was found.
    virtual bool PatchFragment(const uint32_t*, size_t, uint32_t, FamilyPatch&) const { return false; }
    virtual bool PatchCompute(const uint32_t*, size_t, uint32_t, FamilyPatch&, bool* listed) const { *listed = false; return false; }
    virtual void LogRule(const char*, uint32_t, const FamilyPatch&, bool) const {}

    // Images: a shape with a fixed answer (true = decided: *doubled, and for mono *reason); a depth target that is
    // one image for both eyes, whose pass keeps the game's view of it.
    virtual bool ImageRule(const VkImageCreateInfo&, const VkExtent2D&, bool*, const char**) const { return false; }
    // The same, asked after the array, sparse and transfer-image checks: usage patterns; and first for colour targets.
    virtual bool ImageUsageRule(const VkImageCreateInfo&, const VkExtent2D&, bool*, const char**) const { return false; }
    virtual bool ColourTargetRule(const VkImageCreateInfo&, const VkExtent2D&, bool*, const char**) const { return false; }
    virtual bool SharedDepth(const VkImageCreateInfo&, const VkExtent2D&) const { return false; }

    virtual void LogFrame(const DeviceData&) const {}     // its rows of the per-frame report
};

extern FamilyTraits g_familyTraits;
inline const FamilyTraits& Traits() { return g_familyTraits; }

// Builds the families the profile names (registry order) and their traits; once, at the profile latch.
void LoadFamilies();
// The profile elements that built a family, as " VKRage ..."; "" for none.
std::string FamilyElements();
bool IsFamilyElement(const std::string& element);

// The families' answers, first family first.
void FamiliesLoadProfile();
void FamiliesOnPositionShader(uint32_t crc);
bool FamiliesWorldQuadVS(uint32_t crc);
bool FamiliesRepointVS(uint32_t crc);
const VkFamily* FamiliesPatchFragment(const uint32_t* code, size_t words, uint32_t crc, FamilyPatch& out);
// The family that lists the compute shader (its rule may have found nothing: *ok false), or null.
const VkFamily* FamiliesPatchCompute(const uint32_t* code, size_t words, uint32_t crc, FamilyPatch& out, bool* ok);
bool FamiliesImageRule(const VkImageCreateInfo& ci, const VkExtent2D& bb, bool* doubled, const char** reason);
bool FamiliesImageUsageRule(const VkImageCreateInfo& ci, const VkExtent2D& bb, bool* doubled, const char** reason);
bool FamiliesColourTargetRule(const VkImageCreateInfo& ci, const VkExtent2D& bb, bool* doubled, const char** reason);
bool FamiliesSharedDepth(const VkImageCreateInfo& ci, const VkExtent2D& bb);
void FamiliesLogFrame(const DeviceData& d);

// The factories kFamilies calls; null when the profile has no such element, or it is Value="0".
std::unique_ptr<VkFamily> MakeRage(const ProfileEntry* entry);
std::unique_ptr<VkFamily> MakeRpcs3(const ProfileEntry* entry);
std::unique_ptr<VkFamily> MakeShadPs4(const ProfileEntry* entry);
std::unique_ptr<VkFamily> MakeShadGt(const ProfileEntry* entry);
std::unique_ptr<VkFamily> MakeKyty(const ProfileEntry* entry);

// The <VKKyty> family's compute shaders the remap leaves alone (KytyVK.cpp).
bool ComputeRemapSkipped(uint32_t crc);

// The <VKShadGT> family's skies (ShadGtVK.cpp): their VS takes the sky-ray twin, their FS reads the screen at it.
bool GtsSkyVS(uint32_t crc);
bool GtsSkyFS(uint32_t crc);

// The <VKShadPS4> family's shader lists (ShadPs4VK.cpp), for the treatments of the shared switches: VKFsCameraWindows
// (camera windows, decals, sprites, lamp tiles, shafts), VKComputeRemap (the Remap lists), VKPushDeltas (occluders).
bool CameraWindowsAllowed(uint32_t crc);
bool CameraDecalFS(uint32_t crc);
bool ScreenReadTes(uint32_t crc);
bool CameraSpriteVS(uint32_t crc);
int  CameraSpriteVarying(uint32_t crc);     // the sprite varying its depth parallax also moves; -1 for none
bool CameraScreenXFS(uint32_t crc);
bool LampTileVS(uint32_t crc);
bool CameraShaftFS(uint32_t crc);
bool CameraShaftBlur(uint32_t crc);          // the shaft pass that blurs toward a centre (PatchShaftFS blur)
bool ComputeRemapStoresCamera(uint32_t crc);
bool ComputeRemapTiledLights(uint32_t crc);
bool ComputeRemapRayRows(uint32_t crc);
bool ComputeRemapScreenRead(uint32_t crc);
bool ComputeRemapSkyVS(uint32_t crc);
bool ComputeRemapKeepsScreenUv(uint32_t crc);
bool ComputeRemapSoftParticle(uint32_t crc);
bool ComputeRemapFlareFS(uint32_t crc);
bool SkippedOccluderVS(uint32_t crc);        // a depth-only occluder whose right-eye draw is skipped

}} // namespace wiz3d::vk
