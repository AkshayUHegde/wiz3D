/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The family registry, and the families' answers as the shared layer asks them.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/ShaderProfileData.h"

namespace wiz3d { namespace vk {

FamilyTraits g_familyTraits;

namespace {

typedef std::unique_ptr<VkFamily> (*Factory)(const ProfileEntry*);
// Profile element -> family, asked in this order.
const struct { const char* element; Factory make; } kFamilies[] =
{
    { "VKRage", &MakeRage },
    { "VKRpcs3", &MakeRpcs3 },
    { "VKShadPS4", &MakeShadPs4 },
    { "VKShadGT", &MakeShadGt },
    { "VKKyty", &MakeKyty },
};

std::vector<std::unique_ptr<VkFamily>> g_families;
std::string g_elements;

} // namespace

void LoadFamilies()
{
    g_families.clear();
    g_elements.clear();
    g_familyTraits = FamilyTraits();
    for (const auto& r : kFamilies)
    {
        const auto it = g_ProfileData.FeatureEntries.find(r.element);
        std::unique_ptr<VkFamily> f = r.make(it == g_ProfileData.FeatureEntries.end() ? nullptr : &it->second);
        if (!f) continue;
        f->SetTraits(g_familyTraits);
        g_families.push_back(std::move(f));
        g_elements += std::string(" ") + r.element;
    }
    // VKTileListsLeft's twins read the viewport's half width (member 5) from the per-width right-eye sets.
    if (Vk().VKTileListsLeft) g_familyTraits.widthSets = true;
    if (Vk().VKMonoPassCopies) g_familyTraits.monoPassCopies = true;
}

std::string FamilyElements() { return g_elements; }

bool IsFamilyElement(const std::string& element)
{
    for (const auto& r : kFamilies) if (element == r.element) return true;
    return false;
}

void FamiliesLoadProfile() { for (auto& f : g_families) f->LoadProfile(); }

void FamiliesOnPositionShader(uint32_t crc) { for (const auto& f : g_families) f->OnPositionShader(crc); }

bool FamiliesWorldQuadVS(uint32_t crc)
{
    for (const auto& f : g_families) if (f->WorldQuadVS(crc)) return true;
    return false;
}

bool FamiliesRepointVS(uint32_t crc)
{
    for (const auto& f : g_families) if (f->RepointVS(crc)) return true;
    return false;
}

const VkFamily* FamiliesPatchFragment(const uint32_t* code, size_t words, uint32_t crc, FamilyPatch& out)
{
    for (const auto& f : g_families) if (f->PatchFragment(code, words, crc, out)) return f.get();
    return nullptr;
}

const VkFamily* FamiliesPatchCompute(const uint32_t* code, size_t words, uint32_t crc, FamilyPatch& out, bool* ok)
{
    for (const auto& f : g_families)
    {
        bool listed = false;
        *ok = f->PatchCompute(code, words, crc, out, &listed);
        if (listed || *ok) return f.get();
    }
    *ok = false;
    return nullptr;
}

bool FamiliesImageRule(const VkImageCreateInfo& ci, const VkExtent2D& bb, bool* doubled, const char** reason)
{
    for (const auto& f : g_families) if (f->ImageRule(ci, bb, doubled, reason)) return true;
    return false;
}

bool FamiliesImageUsageRule(const VkImageCreateInfo& ci, const VkExtent2D& bb, bool* doubled, const char** reason)
{
    for (const auto& f : g_families) if (f->ImageUsageRule(ci, bb, doubled, reason)) return true;
    return false;
}

bool FamiliesColourTargetRule(const VkImageCreateInfo& ci, const VkExtent2D& bb, bool* doubled, const char** reason)
{
    for (const auto& f : g_families) if (f->ColourTargetRule(ci, bb, doubled, reason)) return true;
    return false;
}

bool FamiliesSharedDepth(const VkImageCreateInfo& ci, const VkExtent2D& bb)
{
    for (const auto& f : g_families) if (f->SharedDepth(ci, bb)) return true;
    return false;
}

void FamiliesLogFrame(const DeviceData& d) { for (const auto& f : g_families) f->LogFrame(d); }

}} // namespace wiz3d::vk
