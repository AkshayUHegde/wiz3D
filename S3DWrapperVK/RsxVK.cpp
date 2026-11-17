/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// RPCS3 (RSX) on Vulkan, the <VKRpcs3> family: its traits and image rules. Its position twin is RsxPosition.cpp,
// a pure patcher the offline tools build on its own.

#include "Layer.h"
#include "FamilyVK.h"

namespace wiz3d { namespace vk {

namespace {

class Rpcs3Family : public VkFamily
{
public:
    const char* Tag() const override { return "rpcs3"; }

    void SetTraits(FamilyTraits& t) const override
    {
        t.pinLayer = t.bindTimeDescriptors = t.widthSets = t.rsxPositionTwins = t.flatNoDepthDraws = true;
        t.portalScissor = t.readColorBuffers = t.bufferTrips = t.tripKernels = t.monoPassCopies = true;
        t.staticPipelineState = true;
    }

    bool ImageUsageRule(const VkImageCreateInfo& ci, const VkExtent2D&, bool* doubled, const char**) const override
    {
        const VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        const bool blockCompressed = ci.format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && ci.format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
        // Render targets also reach shaders through temporary copies (TRANSFER_DST|SAMPLED) and blit destinations
        // (usage shared with uploads, so uncompressed and unmipped only): doubled, or the right eye samples the left's.
        if (!blockCompressed && (ci.usage == (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) ||
            (ci.usage == (xfer | VK_IMAGE_USAGE_SAMPLED_BIT) && ci.mipLevels == 1)))
        { *doubled = true; return true; }
        // An MSAA target's linear twin (its resolve surface), at any size: its target is doubled, so both eyes must not
        // resolve into one image and load it back.
        if (ci.usage == (xfer | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT))
        { *doubled = true; return true; }
        return false;
    }
};

} // namespace

std::unique_ptr<VkFamily> MakeRpcs3(const ProfileEntry* entry)
{
    if (!entry || EntryValue(*entry, "Value") == "0") return nullptr;
    return std::unique_ptr<VkFamily>(new Rpcs3Family());
}

}} // namespace wiz3d::vk
