/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// GT Sport on shadPS4's ShadGT fork, the <VKShadGT> family: flat UI, buffer trips, its UI targets and skies, and the
// twins its attributes switch (Velocity, Glass, Decal, Sky). The twins' patchers are GtsVelocity.cpp.

#include "Layer.h"
#include "FamilyVK.h"
#include <algorithm>
#include <stdio.h>

namespace wiz3d { namespace vk {

namespace {

std::vector<uint32_t> s_skyVs, s_skyFs;   // <Sky> / <SkyScreenRead> children

bool In(const std::vector<uint32_t>& list, uint32_t crc) { return std::find(list.begin(), list.end(), crc) != list.end(); }

// An attribute of the element read as a switch: absent or 0 is off.
bool On(const ProfileEntry& e, const char* attr)
{
    int v = 0;
    return sscanf_s(EntryValue(e, attr).c_str(), "%d", &v) == 1 && v != 0;
}

class ShadGtFamily : public VkFamily
{
public:
    explicit ShadGtFamily(const ProfileEntry& e)
        : m_velocity(On(e, "Velocity")), m_glass(On(e, "Glass")), m_decal(On(e, "Decal")), m_sky(On(e, "Sky"))
    {
        unsigned long long n = 0;
        if (sscanf_s(EntryValue(e, "UiTargetMaxPixels").c_str(), "%llu", &n) == 1) m_uiMaxPixels = n;
    }

    const char* Tag() const override { return "shadgt"; }

    void SetTraits(FamilyTraits& t) const override
    {
        t.flatUiPasses = t.bufferTrips = t.clockFrameRate = true;
        t.velocityTwins = m_velocity; t.glassTwins = m_glass; t.decalTwins = m_decal; t.skyTwins = m_sky;
    }

    void LoadProfile() override
    {
        s_skyVs = ChildShaders("VKShadGT", "Sky");
        s_skyFs = ChildShaders("VKShadGT", "SkyScreenRead");
        Log("[shadgt] profile: velocity=%d glass=%d decal=%d sky=%d | sky shaders %zu VS, %zu FS | UI targets up to %llu px\n",
            (int)m_velocity, (int)m_glass, (int)m_decal, (int)m_sky, s_skyVs.size(), s_skyFs.size(), m_uiMaxPixels);
    }

    bool ColourTargetRule(const VkImageCreateInfo& ci, const VkExtent2D&, bool* doubled, const char** reason) const override
    {
        const uint32_t w = ci.extent.width, h = ci.extent.height;
        // Track maps and icons, small RGBA8 targets the UI composites, whose mask pass has no colour: mono, so both
        // eyes show one map (pinning the colour draws alone could split it).
        if (m_uiMaxPixels && ci.format == VK_FORMAT_R8G8B8A8_UNORM && (unsigned long long)w * h <= m_uiMaxPixels)
        { *doubled = false; *reason = "uiTarget"; return true; }
        return false;
    }

private:
    const bool m_velocity, m_glass, m_decal, m_sky;
    unsigned long long m_uiMaxPixels = 0;   // UiTargetMaxPixels: RGBA8 colour targets up to this many pixels stay mono
};

} // namespace

std::unique_ptr<VkFamily> MakeShadGt(const ProfileEntry* entry)
{
    if (!entry || EntryValue(*entry, "Value") == "0") return nullptr;
    return std::unique_ptr<VkFamily>(new ShadGtFamily(*entry));
}

// <Sky> / <SkyScreenRead>: w == 1 sky quads unprojecting x through the left camera (as ComputeRemapSkyVS), which
// take the sky twins; the scene's and the mirror's.
bool GtsSkyVS(uint32_t crc) { return In(s_skyVs, crc); }
bool GtsSkyFS(uint32_t crc) { return In(s_skyFs, crc); }

}} // namespace wiz3d::vk
