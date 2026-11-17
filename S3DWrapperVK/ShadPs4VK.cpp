/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The PS4 ports built on shadPS4 (Bloodborne, SotC, Uncharted, TLG, inFAMOUS, GoW3, DriveClub), the <VKShadPS4> family:
// which shaders take each treatment of the shared switches, from its children (one entry per shader, as <VKRage>'s).

#include "Layer.h"
#include "FamilyVK.h"
#include <algorithm>
#include <stdio.h>

namespace wiz3d { namespace vk {

namespace {

struct ShadPs4Lists
{
    // VKFsCameraWindows
    std::vector<uint32_t> windows, decal, screenReadTes, sprite, screenX, lampTile, shaft, shaftBlur;
    std::vector<std::pair<uint32_t, int>> spriteVarying;
    // VKComputeRemap, then VKPushDeltas
    std::vector<uint32_t> storesCamera, tiledLights, rayRows, screenRead, sky, keepsScreenUv, softParticle, flare;
    std::vector<uint32_t> occluder;
};
ShadPs4Lists s_ps4;

bool In(const std::vector<uint32_t>& list, uint32_t crc) { return std::find(list.begin(), list.end(), crc) != list.end(); }

void LoadLists()
{
    ShadPs4Lists& l = s_ps4;
    l = ShadPs4Lists();
    const char* e = "VKShadPS4";
    l.windows = ChildShaders(e, "CameraWindows");       l.decal = ChildShaders(e, "CameraDecal");
    l.screenReadTes = ChildShaders(e, "ScreenReadTes"); l.sprite = ChildShaders(e, "CameraSprite");
    l.screenX = ChildShaders(e, "CameraScreenX");       l.lampTile = ChildShaders(e, "LampTile");
    l.shaft = ChildShaders(e, "CameraShaft");
    l.storesCamera = ChildShaders(e, "StoresCamera");   l.tiledLights = ChildShaders(e, "TiledLights");
    l.rayRows = ChildShaders(e, "RayRows");             l.screenRead = ChildShaders(e, "ScreenRead");
    l.sky = ChildShaders(e, "Sky");                     l.keepsScreenUv = ChildShaders(e, "KeepsScreenUv");
    l.softParticle = ChildShaders(e, "SoftParticle");   l.flare = ChildShaders(e, "Flare");
    l.occluder = ChildShaders(e, "DepthOccluder");
    // Per-shader attributes: the sprite varying the parallax also moves, and the shaft pass that blurs.
    for (const ProfileEntry* c : ChildEntries(e))
    {
        DWORD crc = 0;
        const std::string& tag = EntryValue(*c, "@");
        if (tag == "CameraSprite" && sscanf_s(EntryValue(*c, "VertexShader").c_str(), "0x%x", &crc) == 1)
        {
            int v = -1;
            if (sscanf_s(EntryValue(*c, "Varying").c_str(), "%d", &v) == 1) l.spriteVarying.push_back({ crc, v });
        }
        if (tag == "CameraShaft" && EntryValue(*c, "Blur") == "1" && sscanf_s(EntryValue(*c, "PixelShader").c_str(), "0x%x", &crc) == 1)
            l.shaftBlur.push_back(crc);
    }
    const size_t n = l.windows.size() + l.decal.size() + l.screenReadTes.size() + l.sprite.size() + l.screenX.size() +
                     l.lampTile.size() + l.shaft.size() + l.storesCamera.size() + l.tiledLights.size() + l.rayRows.size() +
                     l.screenRead.size() + l.sky.size() + l.keepsScreenUv.size() + l.softParticle.size() + l.flare.size() +
                     l.occluder.size();
    Log("[shadps4] profile: %zu listed shader(s): cameraWindows %zu decal %zu screenReadTes %zu sprite %zu screenX %zu "
        "lampTile %zu shaft %zu | storesCamera %zu tiledLights %zu rayRows %zu screenRead %zu sky %zu keepsScreenUv %zu "
        "softParticle %zu flare %zu | occluder %zu\n", n, l.windows.size(), l.decal.size(), l.screenReadTes.size(),
        l.sprite.size(), l.screenX.size(), l.lampTile.size(), l.shaft.size(), l.storesCamera.size(), l.tiledLights.size(),
        l.rayRows.size(), l.screenRead.size(), l.sky.size(), l.keepsScreenUv.size(), l.softParticle.size(), l.flare.size(),
        l.occluder.size());
}

class ShadPs4Family : public VkFamily
{
public:
    const char* Tag() const override { return "shadps4"; }
    void LoadProfile() override { LoadLists(); }
};

} // namespace

std::unique_ptr<VkFamily> MakeShadPs4(const ProfileEntry* entry)
{
    if (!entry || EntryValue(*entry, "Value") == "0") return nullptr;
    return std::unique_ptr<VkFamily>(new ShadPs4Family());
}

// <CameraWindows>: the fragment shaders whose camera windows are patched (by pattern, every FS would be too many).
bool CameraWindowsAllowed(uint32_t crc) { return In(s_ps4.windows, crc); }

// <CameraDecal>: decals whose VS twin moves their clip varying, then screen x goes to the left eye's before both
// matrices (the decal projection fails the camera test).
bool CameraDecalFS(uint32_t crc) { return In(s_ps4.decal, crc); }

// <ScreenReadTes>: a TES copying Position into a varying that its FS reads as x / w: its twin shifts the copy.
bool ScreenReadTes(uint32_t crc) { return In(s_ps4.screenReadTes, crc); }

// <CameraSprite>: sprites drawn at w == 1, whose twins move them by their depth's parallax (EmitSpriteShift).
bool CameraSpriteVS(uint32_t crc) { return In(s_ps4.sprite, crc); }

// <CameraSprite Varying=>: the (u, v, depth) varying its FS compares with the depth texture at (u, v), moved with
// the sprite.
int CameraSpriteVarying(uint32_t crc)
{
    for (const auto& v : s_ps4.spriteVarying) if (v.first == crc) return v.second;
    return -1;
}

// <CameraScreenX>: screen x is rewritten to the left eye's before every matrix multiplies it.
bool CameraScreenXFS(uint32_t crc) { return In(s_ps4.screenX, crc); }

// <LampTile>: vertex shaders whose lights take the camera twin and open depth bounds in the right eye.
bool LampTileVS(uint32_t crc) { return In(s_ps4.lampTile, crc); }

// <CameraShaft Blur=>: a moved sprite's blur toward a centre given in the left eye's terms: the centre u moves as the
// sprite did, from the depth its FS stored in set 8 binding 1 [8].
bool CameraShaftFS(uint32_t crc) { return In(s_ps4.shaft, crc); }
bool CameraShaftBlur(uint32_t crc) { return In(s_ps4.shaftBlur, crc); }

// <StoresCamera>: shaders holding the camera inverse store its w row, 1/w = n32*z + n33, in set 8 binding 1 [6], [7]
// for readers whose own w row is not the camera's.
bool ComputeRemapStoresCamera(uint32_t crc) { return In(s_ps4.storesCamera, crc); }

// <TiledLights>: tiled lighting compute, integer pixel -> world by a matrix, depth through shared memory, lights
// culled per tile by the left camera.
bool ComputeRemapTiledLights(uint32_t crc) { return In(s_ps4.tiledLights, crc); }

// <RayRows>: screen passes (shadow masks, box decals, full-screen) rebuilding from depth by the per-view inverse.
bool ComputeRemapRayRows(uint32_t crc) { return In(s_ps4.rayRows, crc); }

// <ScreenRead>: shaders reading depth at a clip copy in a way ReadsScreenAtClipVarying does not follow, and sky passes
// whose twins build the rays from the right eye's x: their VS twins move the copy.
bool ComputeRemapScreenRead(uint32_t crc) { return In(s_ps4.screenRead, crc); }

// <Sky>: skies pinned at w == 1 with rays from the left's x (in the scene or a reflection): the twin takes the right's.
bool ComputeRemapSkyVS(uint32_t crc) { return In(s_ps4.sky, crc); }

// <KeepsScreenUv>: decals turning their screen coordinate back into a world point with the left camera's matrix,
// where moving the coordinate alone would slide them.
bool ComputeRemapKeepsScreenUv(uint32_t crc) { return In(s_ps4.keepsScreenUv, crc); }

// <SoftParticle>: soft particles reading depth at a varying (k/w)*x + c built before the VS's shift: read at the
// right eye's pixel instead.
bool ComputeRemapSoftParticle(uint32_t crc) { return In(s_ps4.softParticle, crc); }

// <Flare>: full-screen, pinned glows around points from a table the game fills from the left camera.
bool ComputeRemapFlareFS(uint32_t crc) { return In(s_ps4.flare, crc); }

// <DepthOccluder>: a depth-only occluder mesh (empty FS) whose right-eye draw is skipped (VKPushDeltas): no colour
// needs it, and its right-eye depth could fail every draw behind it.
bool SkippedOccluderVS(uint32_t crc) { return In(s_ps4.occluder, crc); }

}} // namespace wiz3d::vk
