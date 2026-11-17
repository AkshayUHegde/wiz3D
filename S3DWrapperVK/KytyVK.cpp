/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// KytyPS5 (Demon's Souls) on Vulkan, the <VKKyty> family: buffer trips, the fused MAD, the tile lists (KytyTiles.cpp),
// DLSS-G (KytyFrameGen.cpp), its material masks and the compute shaders the remap leaves alone, from <VKKyty> children.

#include "Layer.h"
#include "FamilyVK.h"
#include <algorithm>

namespace wiz3d { namespace vk {

namespace {

std::vector<uint32_t> s_remapSkipped;   // <RemapSkipped> children

class KytyFamily : public VkFamily
{
public:
    const char* Tag() const override { return "kyty"; }

    void SetTraits(FamilyTraits& t) const override
    {
        t.bufferTrips = t.fusedMad = t.tileLists = true;
    }

    void LoadProfile() override
    {
        s_remapSkipped = ChildShaders("VKKyty", "RemapSkipped");
        TilesLoadProfile();
    }

    bool ImageUsageRule(const VkImageCreateInfo& ci, const VkExtent2D&, bool* doubled, const char** reason) const override
    {
        const uint32_t w = ci.extent.width, h = ci.extent.height;
        // 256x256 RGBA8 material masks hold nothing per eye: one copy, the left's, as the game's own commands left it
        // (a right copy diverges).
        if (ci.format == VK_FORMAT_R8G8B8A8_UNORM && w == 256 && h == 256 && ci.mipLevels == 1) { *doubled = false; *reason = "squareMask"; return true; }
        return false;
    }
};

} // namespace

std::unique_ptr<VkFamily> MakeKyty(const ProfileEntry* entry)
{
    if (!entry || EntryValue(*entry, "Value") == "0") return nullptr;
    return std::unique_ptr<VkFamily>(new KytyFamily());
}

// <RemapSkipped>: compute shaders the remap leaves alone, e.g. one projecting its position back to the screen for
// its depth reads.
bool ComputeRemapSkipped(uint32_t crc)
{
    return std::find(s_remapSkipped.begin(), s_remapSkipped.end(), crc) != s_remapSkipped.end();
}

}} // namespace wiz3d::vk
