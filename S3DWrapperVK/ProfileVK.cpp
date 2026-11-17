/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/
// Switch table over S3DAPI's profile entries (ReadFeatureEntry): DefaultProfile, <Engine>, Profile, UserProfile.
#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include "../S3DAPI/ShaderProfileData.h"
#include "../Shared/version.h"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

ProfileData& g_ProfileData = _g_ProfileData;   // as each wrapper binds S3DAPI's exported profile data

namespace wiz3d { namespace vk {

VkSwitches g_vkSwitches;

namespace {

std::vector<uint32_t> s_screenDepth;

enum class Kind { Bool, Int, Float };
struct Switch { const char* name; Kind kind; size_t offset; };

#define B(n) { #n, Kind::Bool,  offsetof(VkSwitches, n) }
#define I(n) { #n, Kind::Int,   offsetof(VkSwitches, n) }
#define F(n) { #n, Kind::Float, offsetof(VkSwitches, n) }
const Switch kSwitches[] = {
    F(VKSeparationScale), B(VKSkipUnitW), F(VKNearClamp), B(VKDoubleStorageImages), I(VKSiblingBudget), B(VKTileListsLeft),
    B(VKScreenMarch), I(VKUiTargetMaxPixels), B(VKSmallDrawUnshift), F(VKEdgeCrop), B(VKGprPositionChain),
    B(VKServeFallback), B(VKValidateInverse), B(VKRepointLearnedSets), B(VKEqualPassSelfShift),
    B(VKMaskMatrixSpan), B(VKDoubleArrayImages), B(VKMixedFramebufferMono), B(VKMajorityP11), B(VKChainThroughBitcast),
    B(VKDynamicRendering), B(VKBufferTwins), B(VKFsCameraWindows), B(VKScreenReadVaryings), B(VKDoubleTransferImages),
    B(VKShiftPrevClipVaryings), B(VKPushConstantOrder), B(VKShiftNdcVaryings),
    I(VKIndirectFromLeft), I(VKComputeRemap), B(VKDoubleFroxelVolumes), B(VKLightRaysCamera),
    B(VKShadowSetTwins), B(VKDynByPipeline), B(VKPushDeltas),
    B(VKPinLayer),
    B(VKHideDescriptorBuffer), B(VKHideDepthBiasControl), B(VKMonoPassCopies), B(VKMeshDraws),
    B(VKScreenUvVaryings), B(VKFroxelLookups), B(VKFragCoordRays), B(VKOcclusionBothEyes),
    F(VKUnitWEpsilon), B(VKStereoDLSS), B(VKWeaveGpuSync), B(VKFrameGen), B(VKShadowAtlasByShape), I(VKDumpModifiedShaders),
};
#undef B
#undef I
#undef F

char* Field(VkSwitches& s, const Switch& w) { return reinterpret_cast<char*>(&s) + w.offset; }

// The same reads TinyXML's QueryIntAttribute / QueryFloatAttribute make: a value that does not parse keeps the default.
void Apply(const Switch& w, const std::string& value)
{
    char* p = Field(g_vkSwitches, w);
    int i = 0;
    double f = 0.0;
    switch (w.kind)
    {
    case Kind::Bool:  if (sscanf_s(value.c_str(), "%d", &i) == 1) *reinterpret_cast<bool*>(p) = i != 0; break;
    case Kind::Int:   if (sscanf_s(value.c_str(), "%d", &i) == 1) *reinterpret_cast<int*>(p) = i; break;
    case Kind::Float: if (sscanf_s(value.c_str(), "%lf", &f) == 1) *reinterpret_cast<float*>(p) = (float)f; break;
    }
}

bool Known(const std::string& name)
{
    for (const Switch& w : kSwitches) if (name == w.name) return true;
    return name == "VKScreenDepth" || name == "VKViewModel" || IsFamilyElement(name);
}

} // namespace

const std::string& EntryValue(const ProfileEntry& entry, const char* attr)
{
    static const std::string none;
    const ProfileEntry::const_iterator it = entry.find(attr);
    return it == entry.end() ? none : it->second;
}

// As ReadFeatureEntry stores them: "Element/1", "Element/2"...
std::vector<const ProfileEntry*> ChildEntries(const char* element)
{
    std::vector<const ProfileEntry*> out;
    for (int n = 1; ; ++n)
    {
        const auto it = g_ProfileData.FeatureEntries.find(std::string(element) + "/" + std::to_string(n));
        if (it == g_ProfileData.FeatureEntries.end()) break;
        out.push_back(&it->second);
    }
    return out;
}

std::vector<uint32_t> ChildShaders(const char* element, const char* child)
{
    std::vector<uint32_t> out;
    for (const ProfileEntry* e : ChildEntries(element))
    {
        if (EntryValue(*e, "@") != child) continue;
        for (const char* a : { "VertexShader", "DomainShader", "PixelShader", "ComputeShader" })
        {
            DWORD crc = 0;
            if (sscanf_s(EntryValue(*e, a).c_str(), "0x%x", &crc) == 1) out.push_back(crc);
        }
    }
    return out;
}

void LoadVkSwitches()
{
    for (const Switch& w : kSwitches)
    {
        const auto e = g_ProfileData.FeatureEntries.find(w.name);
        if (e == g_ProfileData.FeatureEntries.end()) continue;
        const auto v = e->second.find("Value");
        if (v != e->second.end()) Apply(w, v->second);
    }
    s_screenDepth = ChildShaders("VKScreenDepth", "Draw");
    // <VKViewModel ClipZ=".." Scale=".."/>: the first-person weapon's own sep and conv; Sep / Conv set one apart.
    const auto vm = g_ProfileData.FeatureEntries.find("VKViewModel");
    if (vm != g_ProfileData.FeatureEntries.end())
    {
        float f = 0.f;
        if (sscanf_s(EntryValue(vm->second, "ClipZ").c_str(), "%f", &f) == 1) g_vkSwitches.VKViewModelClipZ = f;
        if (sscanf_s(EntryValue(vm->second, "Scale").c_str(), "%f", &f) == 1) g_vkSwitches.VKViewModelSep = g_vkSwitches.VKViewModelConv = f;
        if (sscanf_s(EntryValue(vm->second, "Sep").c_str(), "%f", &f) == 1) g_vkSwitches.VKViewModelSep = f;
        if (sscanf_s(EntryValue(vm->second, "Conv").c_str(), "%f", &f) == 1) g_vkSwitches.VKViewModelConv = f;
    }
}

const std::vector<uint32_t>& ScreenDepthShaders() { return s_screenDepth; }

void LogVkSwitches()
{
    static const VkSwitches kDefaults;
    VkSwitches& s = g_vkSwitches;
    std::string row;
    char item[96];
    for (const Switch& w : kSwitches)
    {
        const char* a = Field(s, w);
        const char* b = reinterpret_cast<const char*>(&kDefaults) + w.offset;
        switch (w.kind)
        {
        case Kind::Bool:
            if (*reinterpret_cast<const bool*>(a) == *reinterpret_cast<const bool*>(b)) continue;
            sprintf_s(item, " %s=%d", w.name, (int)*reinterpret_cast<const bool*>(a)); break;
        case Kind::Int:
            if (*reinterpret_cast<const int*>(a) == *reinterpret_cast<const int*>(b)) continue;
            sprintf_s(item, " %s=%d", w.name, *reinterpret_cast<const int*>(a)); break;
        case Kind::Float:
            if (*reinterpret_cast<const float*>(a) == *reinterpret_cast<const float*>(b)) continue;
            sprintf_s(item, " %s=%g", w.name, (double)*reinterpret_cast<const float*>(a)); break;
        }
        row += item;
    }
    if (!s_screenDepth.empty())
    {
        sprintf_s(item, " VKScreenDepth=%u shaders", (unsigned)s_screenDepth.size());
        row += item;
    }
    if (s.VKViewModelClipZ > 0.f)
    {
        sprintf_s(item, " VKViewModel=ClipZ %g Sep %g Conv %g", (double)s.VKViewModelClipZ, (double)s.VKViewModelSep, (double)s.VKViewModelConv);
        row += item;
    }
    const std::string families = FamilyElements().empty() ? "" : " families:" + FamilyElements();
    if (gInfo.bMatchedInBase || gInfo.bMatchedInUser)
        Log("wiz3D %s: profile '%S' (%s)%s switches:%s\n", DISPLAYED_VERSION, gInfo.ProfileName,
            gInfo.bMatchedInBase ? (gInfo.bMatchedInUser ? "BaseProfile + UserProfile" : "BaseProfile") : "UserProfile",
            families.c_str(), row.empty() ? " (defaults)" : row.c_str());
    else
        Log("wiz3D %s: drop-in: no profile names this game -> core defaults%s%s%s\n", DISPLAYED_VERSION,
            families.c_str(), row.empty() ? "" : "; switches:", row.c_str());
    // VK elements only: the rest of the profile belongs to other wrappers or to S3DAPI itself.
    for (const auto& e : g_ProfileData.FeatureEntries)
        if (e.first.compare(0, 2, "VK") == 0 && e.first.find('/') == std::string::npos && !Known(e.first))
            Log("  profile: <%s> is no VK switch; ignored\n", e.first.c_str());
}

}} // namespace wiz3d::vk
