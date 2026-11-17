/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// RAGE (Red Dead Redemption 2) on Vulkan, the <VKRage> family: shader rules, image shapes, the layer's traits.
// Which shader gets which rule, and the engine's block layout, come from the profile's <VKRage> children.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/ReadData.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <stdio.h>

namespace wiz3d { namespace vk {

// Every water material, and no other shader, carries the constant 1.17647: a shape, not an identity.
static const uint32_t kWaterConstant = 0x3F969696u;

struct RageProfile
{
    // Treatments, by the <VKRage> child element that lists their shaders.
    std::vector<uint32_t> worldQuad;        // <WorldQuad>: water planes and mirror glass, 6-index world quads
    std::vector<uint32_t> waterSurface;     // <WaterSurface>: the water VS whose cached sets are re-pointed
    std::vector<uint32_t> mirrorGlass;      // <MirrorGlass Clip=>: reads its reflection at a clip varying
    std::vector<uint32_t> fogApply;         // <FogApply>: full-screen read of the left-built froxel volume
    std::vector<uint32_t> lightViewRay;     // <LightViewRay Ray= Uv=>: lighting reflecting normalize(ray varying)
    std::vector<uint32_t> screenRayLight;   // <ScreenRayLight Ray=>: clustered lighting, position = ray * depth
    std::vector<uint32_t> occlusionVolume;  // <OcclusionVolume>: FS drawn with the occlusion-box VS
    std::vector<uint32_t> sunMaskFar;       // <SunMaskFar>: the sun mask's far half: skewed rays, no product
    std::vector<uint32_t> cloudRays;        // <CloudRays>: half-res cloud compute passes
    std::vector<uint32_t> ssrRays;          // <SsrRays>: reflection trace and its history resolve
    // Layout numbers. -1: the profile does not give it, and the rules needing it stay off.
    int cameraBinding = -1, cameraAltBinding = -1, projection = -1, previousProjection = -1;   // <CameraBlock>
    int waterBinding = -1, waterMember = -1, waterField = -1;                                    // <WaterBlock>: its VP
    int viewRay = -1, viewRayUv = -1, screenRay = -1, glassClip = -1;                            // input locations
    // Storage image shapes (VkFormat numbers). -1: no such rule.
    int exposureFormat = -1, exposureWidth = -1, exposureHeight = -1;   // <MonoExposureGrid>
    int dofFormat = -1, dofDivisor = -1;                                // <MonoDofTiles>: backbuffer / divisor
    int reflFormat = -1, reflLayers = -1, reflAspect = -1;              // <StereoReflectionArray>: w = aspect * h
};
static RageProfile s_rage;

static bool In(const std::vector<uint32_t>& list, uint32_t crc) { return std::find(list.begin(), list.end(), crc) != list.end(); }

// One numeric attribute of a <VKRage> child.
static bool TreatmentAttribute(const char* treatment, const char* attribute, int* value)
{
    const std::string* last = nullptr;   // the last child naming it wins, then it is parsed once
    for (const ProfileEntry* child : ChildEntries("VKRage"))
    {
        if (EntryValue(*child, "@") != treatment) continue;
        const auto v = child->find(attribute);
        if (v != child->end()) last = &v->second;
    }
    return last && sscanf_s(last->c_str(), "%d", value) == 1;
}

static void RageLoadProfile()
{
    s_rage = RageProfile();
    unsigned total = 0;
    auto shaders = [&](const char* name, std::vector<uint32_t>& out)
    {
        out = ChildShaders("VKRage", name);
        total += (unsigned)out.size();
    };
    auto attr = [](const char* name, const char* a, int& v) { int x = 0; if (TreatmentAttribute(name, a, &x)) v = x; };
    shaders("WorldQuad", s_rage.worldQuad);             shaders("WaterSurface", s_rage.waterSurface);
    shaders("MirrorGlass", s_rage.mirrorGlass);         shaders("FogApply", s_rage.fogApply);
    shaders("LightViewRay", s_rage.lightViewRay);       shaders("ScreenRayLight", s_rage.screenRayLight);
    shaders("OcclusionVolume", s_rage.occlusionVolume); shaders("SunMaskFar", s_rage.sunMaskFar);
    shaders("CloudRays", s_rage.cloudRays);             shaders("SsrRays", s_rage.ssrRays);
    attr("CameraBlock", "Binding", s_rage.cameraBinding);        attr("CameraBlock", "AltBinding", s_rage.cameraAltBinding);
    attr("CameraBlock", "Projection", s_rage.projection);        attr("CameraBlock", "PreviousProjection", s_rage.previousProjection);
    attr("WaterBlock", "Binding", s_rage.waterBinding);          attr("WaterBlock", "Member", s_rage.waterMember);
    attr("WaterBlock", "Field", s_rage.waterField);
    attr("LightViewRay", "Ray", s_rage.viewRay);                 attr("LightViewRay", "Uv", s_rage.viewRayUv);
    attr("ScreenRayLight", "Ray", s_rage.screenRay);             attr("MirrorGlass", "Clip", s_rage.glassClip);
    attr("MonoExposureGrid", "Format", s_rage.exposureFormat);   attr("MonoExposureGrid", "Width", s_rage.exposureWidth);
    attr("MonoExposureGrid", "Height", s_rage.exposureHeight);
    attr("MonoDofTiles", "Format", s_rage.dofFormat);            attr("MonoDofTiles", "BackbufferDivisor", s_rage.dofDivisor);
    attr("StereoReflectionArray", "Format", s_rage.reflFormat);  attr("StereoReflectionArray", "Layers", s_rage.reflLayers);
    attr("StereoReflectionArray", "Aspect", s_rage.reflAspect);
    // The projection members name bits of a 32-bit member mask.
    for (int* m : { &s_rage.projection, &s_rage.previousProjection })
        if (*m > 31)
        {
            Log("[rage] profile: <CameraBlock> projection member %d is out of range (0..31): the rules needing it are off\n", *m);
            *m = -1;
        }
    Log("[rage] profile: %u listed shader(s): worldQuad %zu, waterSurface %zu, mirrorGlass %zu, fogApply %zu, lightViewRay %zu, screenRayLight %zu, "
        "occlusionVolume %zu, sunMaskFar %zu, cloudRays %zu, ssrRays %zu | camera block b%d/b%d P m%d prevP m%d | water block b%d m%d.%d | "
        "locations: viewRay %d uv %d, screenRay %d, glassClip %d | images: exposure f%d %dx%d, dofTiles f%d /%d, reflection f%d x%d\n",
        total, s_rage.worldQuad.size(), s_rage.waterSurface.size(), s_rage.mirrorGlass.size(), s_rage.fogApply.size(), s_rage.lightViewRay.size(),
        s_rage.screenRayLight.size(), s_rage.occlusionVolume.size(), s_rage.sunMaskFar.size(), s_rage.cloudRays.size(), s_rage.ssrRays.size(),
        s_rage.cameraBinding, s_rage.cameraAltBinding, s_rage.projection, s_rage.previousProjection,
        s_rage.waterBinding, s_rage.waterMember, s_rage.waterField, s_rage.viewRay, s_rage.viewRayUv, s_rage.screenRay, s_rage.glassClip,
        s_rage.exposureFormat, s_rage.exposureWidth, s_rage.exposureHeight, s_rage.dofFormat, s_rage.dofDivisor, s_rage.reflFormat, s_rage.reflLayers);
    if (!total) Log("[rage] profile: VKRage is on but its entry lists NO shaders (<VKRage> children missing): only the shape rules run\n");
    if (s_rage.cameraBinding < 0 || s_rage.projection < 0) Log("[rage] profile: no <CameraBlock Binding Projection>: water, view-ray and reflection rules are OFF\n");
}

static void RageLogRule(const char* stage, uint32_t crc, const FamilyPatch& rp, bool patched)
{
    static std::mutex s_lock;
    static std::map<std::string, unsigned> s_count;
    unsigned n;
    { std::lock_guard<std::mutex> g(s_lock); n = ++s_count[rp.rule]; }
    if (rp.always || !patched || n <= 8 || n % 100 == 0)
        Log("[rage] %s %08X %s: %s%s | module %u\n", stage, crc, rp.rule, rp.row, patched ? "" : " - NOT patched, the right eye keeps the left's", n);
}

// The dither fade on top of another rule's output: that rule returns, so a material with both would lose it.
static unsigned ChainDither(std::vector<uint32_t>& fs)
{
    std::vector<uint32_t> dith;
    unsigned sites = 0;
    if (PatchScreenDitherFS(fs.data(), fs.size(), kStereoSet, dith, &sites)) fs.swap(dith);
    return sites;
}

// Fragment rules in order; the first that takes the module wins.
static bool RagePatchFragment(const uint32_t* code, size_t words, uint32_t crc, FamilyPatch& out)
{
    const RageProfile& rg = s_rage;
    const bool camera = rg.cameraBinding >= 0 && rg.projection >= 0;
    // Water and mirror glass; the glass reads its reflection at bias + scale*clip.xy/clip.w, drawn for the left's crop.
    const bool glass = rg.glassClip >= 0 && In(rg.mirrorGlass, crc);
    uint32_t waterBinding = 0;
    if (!glass && camera && std::find(code, code + words, kWaterConstant) != code + words)
    {
        std::vector<uint32_t> probe;
        unsigned n = 0;
        for (int b : { rg.cameraBinding, rg.cameraAltBinding })
            if (b >= 0 && PatchDirProjectionFS(code, words, kStereoSet, (uint32_t)b, 1u << rg.projection, probe, &n)) { waterBinding = (uint32_t)b; break; }
    }
    if (waterBinding || glass)
    {
        std::vector<uint32_t> fs, surf;
        unsigned dirs = 0;
        // The surface point rebuilt as rows . (ndc, 1) / gl_FragCoord.w, so foam, ripples and flow maps are looked up
        // at this eye's point.
        ReconstructionRules surfaceRules;
        surfaceRules.skewExact = true;
        surfaceRules.fragDepth = true;
        ReconstructionCounts surfaceCounts;
        const bool surface = !glass && PatchReconstructionFS(code, words, kStereoSet, surfaceRules, surf, &surfaceCounts);
        const unsigned surfProds = surfaceCounts.products;
        const uint32_t* wcode = surface ? surf.data() : code;
        const size_t wwords = surface ? surf.size() : words;
        // The reflection offset is a direction through the camera block's P: the right P moves it by fsSep.
        if (glass ? PatchClipVaryingLeftFS(code, words, kStereoSet, (uint32_t)rg.glassClip, fs, &dirs)
                  : PatchDirProjectionFS(wcode, wwords, kStereoSet, waterBinding, 1u << rg.projection, fs, &dirs))
        {
            // The refracted point goes through the shared water block's view-projection, the left camera's in both.
            unsigned refr = 0;
            std::vector<uint32_t> both;
            if (!glass && rg.waterBinding >= 0 && rg.waterMember >= 0 && rg.waterField >= 0 &&
                PatchBufferProjectionFS(fs.data(), fs.size(), kStereoSet, (uint32_t)rg.waterBinding, (uint32_t)rg.waterMember, (uint32_t)rg.waterField, both, &refr))
                fs.swap(both);
            out.code.swap(fs);
            out.rule = glass ? "glass" : "water";
            out.dumpTag = glass ? "glassleft" : "waterrefl";
            out.always = glass;
            if (glass) sprintf_s(out.row, "%u clip varying load(s) read at the left eye's pixel", dirs);
            else sprintf_s(out.row, "surface position %u, reflection offset %u (b%u), refraction projection %u", surface ? surfProds : 0, dirs, waterBinding, refr);
            return true;
        }
        Log("[rage] FS %08X %s: its lookup was not found - NOT patched\n", crc, glass ? "glass" : "water");
    }
    // The full-screen fog apply reads the froxel volume (built once, for the left camera) at its own screen uv.
    if (In(rg.fogApply, crc))
    {
        unsigned lookups = 0;
        if (PatchFogVolumeLeftFS(code, words, kStereoSet, out.code, &lookups))
        {
            out.rule = "fog"; out.dumpTag = "fogvol"; out.always = true;
            sprintf_s(out.row, "%u fog volume lookup(s) at the left eye's pixel", lookups);
            return true;
        }
        Log("[rage] FS %08X fog: its volume lookup was not found - NOT patched\n", crc);
    }
    // Full-screen lighting drawn with the left camera's ray in both eyes, listed for either rule or both: its view
    // direction (probe reflections) taken at the right eye's ray, its position ray * depth given the eye offset.
    {
        const bool view = camera && rg.viewRay >= 0 && rg.viewRayUv >= 0 && In(rg.lightViewRay, crc);
        const bool pos = rg.screenRay >= 0 && In(rg.screenRayLight, crc);
        std::vector<uint32_t> a, b;
        const uint32_t* c = code;
        size_t w = words;
        unsigned dirs = 0, prods = 0;
        if (view)
        {
            if (PatchViewRayFS(c, w, kStereoSet, (uint32_t)rg.viewRay, (uint32_t)rg.viewRayUv, (uint32_t)rg.cameraBinding, (uint32_t)rg.projection, a, &dirs)) { c = a.data(); w = a.size(); }
            else Log("[rage] FS %08X lighting: its view direction was not found\n", crc);
        }
        if (pos)
        {
            if (PatchScreenRayPositionFS(c, w, kStereoSet, (uint32_t)rg.screenRay, b, &prods)) { c = b.data(); w = b.size(); }
            else Log("[rage] FS %08X lighting: its ray * depth product was not found\n", crc);
        }
        if (c != code)
        {
            out.code.assign(c, c + w);
            const unsigned ds = prods ? ChainDither(out.code) : 0;
            out.rule = "lighting"; out.dumpTag = prods ? "screenray" : "viewray"; out.always = true;
            sprintf_s(out.row, "%u view direction(s) taken at the right eye's ray, %u position(s) given the eye offset, %u dither site(s)", dirs, prods, ds);
            return true;
        }
    }
    // Screen passes rebuilding a camera-relative position (the volume shape also matches fog and probes: a list).
    {
        std::vector<uint32_t> fs;
        ReconstructionRules rules;
        rules.skewExact = true;
        rules.fragDepth = true;
        rules.volumeRays = In(rg.occlusionVolume, crc);   // only where the FS is drawn with the occlusion-box VS
        rules.volumeLights = true;   // local lights seen from outside their volume rebuild the position from its ray
        rules.screenSkew = true;
        rules.skewedRays = In(rg.sunMaskFar, crc);        // the sun mask's far half has no ray product of its own
        ReconstructionCounts n;
        if (PatchReconstructionFS(code, words, kStereoSet, rules, fs, &n))
        {
            const unsigned ds = ChainDither(fs);
            out.code.swap(fs);
            out.rule = "position"; out.dumpTag = "fsorigin"; out.dumpOriginal = true; out.eyeOriginStats = true; out.products = n.products;
            sprintf_s(out.row, "%u reconstruction(s) given the eye origin, %u from volume geometry, %u screen re-projection(s), %u dither site(s)",
                      n.products, n.volume, n.screen, ds);
            return true;
        }
    }
    // Material shaders fade by a dither on gl_FragCoord, read at the left's pixel so the stipple keeps its depth.
    {
        unsigned sites = 0;
        if (PatchScreenDitherFS(code, words, kStereoSet, out.code, &sites))
        {
            out.rule = "dither"; out.dumpTag = "dither";
            sprintf_s(out.row, "%u dither lookup(s) at the left eye's pixel", sites);
            return true;
        }
    }
    return false;
}

// Compute rules (clouds, reflections): a twin the right run binds.
static bool RagePatchCompute(const uint32_t* code, size_t words, uint32_t crc, FamilyPatch& out, bool* listed)
{
    const RageProfile& rg = s_rage;
    const bool cloud = In(rg.cloudRays, crc), ssr = In(rg.ssrRays, crc);
    *listed = cloud || ssr;
    // The half-res cloud passes (far layer, volumetric raymarch, the fog marching the cumulus) at the right run's skew.
    if (cloud)
    {
        unsigned n = 0;
        const bool ok = PatchSkyRaysCS(code, words, kStereoSet, out.code, &n);
        out.rule = "cloud"; out.dumpTag = "cloud"; out.always = true;
        sprintf_s(out.row, "%u view ray(s) at the right run's skew", n);
        return ok;
    }
    // The reflection trace and its history resolve project through the camera block's P and previous P.
    if (ssr)
    {
        unsigned n[3] = {};
        const bool ok = rg.cameraBinding >= 0 && rg.projection >= 0 && rg.previousProjection >= 0 &&
                        PatchEyeRaysCS(code, words, kStereoSet, (uint32_t)rg.cameraBinding, (1u << rg.projection) | (1u << rg.previousProjection), out.code, n);
        out.rule = "ssr"; out.dumpTag = "ssr"; out.always = true;
        sprintf_s(out.row, "rays=%u positions=%u projections=%u at the right camera", n[0], n[1], n[2]);
        return ok;
    }
    return false;
}

// A depth target not shaped like the backbuffer (shadow atlases, light maps): one image; a pass keeps the game's view.
static bool RageSharedDepth(const VkImageCreateInfo& ci, const VkExtent2D& bb)
{
    return (ci.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) && OffAspect(ci.extent.width, ci.extent.height, bb);
}

// Image shapes with a fixed answer (exposure grid and depth-of-field tile map mono, reflection array doubled).
static bool RageImageRule(const VkImageCreateInfo& ci, const VkExtent2D& bb, bool* doubled, const char** reason)
{
    const RageProfile& rg = s_rage;
    if (!(ci.usage & VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    // The auto-exposure grid: its pass also writes the shared 1x1 exposure texel, so it must run once.
    if (rg.exposureFormat >= 0 && (int)ci.format == rg.exposureFormat && (int)ci.extent.width == rg.exposureWidth &&
        (int)ci.extent.height == rg.exposureHeight && ci.arrayLayers == 1)
    { *doubled = false; *reason = "exposureGrid"; return true; }
    // The depth-of-field tile map: its pass also appends to the shared tile lists.
    if (rg.dofFormat >= 0 && rg.dofDivisor > 0 && (int)ci.format == rg.dofFormat && ci.arrayLayers == 1 && bb.width && bb.height &&
        ci.extent.width == (bb.width + rg.dofDivisor - 1) / rg.dofDivisor && ci.extent.height == (bb.height + rg.dofDivisor - 1) / rg.dofDivisor)
    { *doubled = false; *reason = "dofTiles"; return true; }
    // The water reflection array, written by its temporal resolve in each eye's run: one copy per eye.
    if (rg.reflFormat >= 0 && (int)ci.format == rg.reflFormat && (int)ci.arrayLayers == rg.reflLayers && rg.reflAspect > 0 &&
        ci.extent.width == (uint32_t)rg.reflAspect * ci.extent.height &&
        !(ci.flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)))
    { *doubled = true; *reason = ""; return true; }
    return false;
}

class RageFamily : public VkFamily
{
public:
    const char* Tag() const override { return "rage"; }

    void SetTraits(FamilyTraits& t) const override
    {
        // Cutscene lenses reach P11 = mm/12 (50 mm 4.17, past the generic 6); far-layer ropes and wires have a zero
        // depth row, so a projection without an inverse still shifts.
        t.limits = { 64.f, 1.f / 64.f, 1.f / 64.f, true };
        t.clipSlotStraddle = t.claimedForward = t.previousProjection = t.unpatchedKeys = t.claimMirror = true;
        t.sharedCollect = t.claimedKeysNeedP = t.claimedKeyRetry = t.unitCameraBlock = t.positionBindings = true;
        t.screenTriangles = t.offAspectPasses = t.widthSets = t.matrixOverScalar = true;
        t.fragmentRules = t.computeRules = true;
    }

    void LoadProfile() override { RageLoadProfile(); }

    void OnPositionShader(uint32_t crc) const override
    {
        if (In(s_rage.worldQuad, crc)) Log("[rage] VS %08X worldquad: its <=6-vertex quads claim their camera range as geometry\n", crc);
    }
    bool WorldQuadVS(uint32_t crc) const override { return In(s_rage.worldQuad, crc); }
    bool RepointVS(uint32_t crc) const override { return In(s_rage.waterSurface, crc); }

    bool PatchFragment(const uint32_t* code, size_t words, uint32_t crc, FamilyPatch& out) const override
    { return RagePatchFragment(code, words, crc, out); }
    bool PatchCompute(const uint32_t* code, size_t words, uint32_t crc, FamilyPatch& out, bool* listed) const override
    { return RagePatchCompute(code, words, crc, out, listed); }
    void LogRule(const char* stage, uint32_t crc, const FamilyPatch& rp, bool patched) const override
    { RageLogRule(stage, crc, rp, patched); }

    bool ImageRule(const VkImageCreateInfo& ci, const VkExtent2D& bb, bool* doubled, const char** reason) const override
    { return RageImageRule(ci, bb, doubled, reason); }
    bool SharedDepth(const VkImageCreateInfo& ci, const VkExtent2D& bb) const override { return RageSharedDepth(ci, bb); }

    void LogFrame(const DeviceData& d) const override
    {
        Log("[frame %u] rage: screen triangles on the game's camera set=%u | left on the shadow (doubled image in the set, or bind unknown)=%u\n",
            d.frames, d.statScreenTriLeft.load(), d.statScreenTriRefused.load());
    }
};

std::unique_ptr<VkFamily> MakeRage(const ProfileEntry* entry)
{
    if (!entry || EntryValue(*entry, "Value") == "0") return nullptr;
    return std::unique_ptr<VkFamily>(new RageFamily());
}

}} // namespace wiz3d::vk
