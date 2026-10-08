/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <LeftEyeTextures Value="texture ..."/>: light-space maps are the same for both eyes, so shaders reading these names read
// the game's (left) copy in the right eye too (Context11Proxy::BindLeftSRVs).

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <algorithm>
#include <sstream>

namespace wiz3d
{
namespace
{

class LeftEyeTexturesFeature : public ProfileFeature
{
public:
    explicit LeftEyeTexturesFeature(const ProfileEntry& e)
    {
        std::istringstream in(EntryValue(e, "Value"));
        std::string n;
        while (in >> n) m_names.push_back(n);
    }

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        for (const std::string& n : m_names)
        {
            const int slot = rdef.TextureSlot(n);
            if (slot >= 0 && std::find(info.leftSRVSlots.begin(), info.leftSRVSlots.end(), slot) == info.leftSRVSlots.end())
                info.leftSRVSlots.push_back(slot);
        }
    }

private:
    std::vector<std::string> m_names;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeLeftEyeTextures(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new LeftEyeTexturesFeature(*e));
}

} // namespace wiz3d
