/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <UiProjectionParallax Value="1" SceneNear="zn"/>: a clean projection whose focal is over 2x the scene camera's is a UI
// projection, not another FOV (a weapon's stays within); its draw takes the scene camera's parallax (4A's 3D menu items).
// SceneNear: the gameplay camera is the scene camera whose projection has this near plane, so another camera's 3D view
// takes its parallax too (a pause map's telephoto camera).

#include "StdAfx.h"
#include <stdlib.h>
#include "ProfileFeature.h"
#include "..\S3DAPI\ShaderProfileData.h"   // g_ProfileData

namespace wiz3d
{
namespace
{

class UiProjectionParallaxFeature : public ProfileFeature
{
public:
    bool UiProjectionParallax() const override { return true; }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeUiProjectionParallax(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new UiProjectionParallaxFeature());
}

float UiProjectionReferenceNear()
{
    static const float s_near = []
    {
        const auto it = g_ProfileData.FeatureEntries.find("UiProjectionParallax");
        if (it == g_ProfileData.FeatureEntries.end() || EntryValue(it->second, "Value") == "0") return 0.f;
        return (float)atof(EntryValue(it->second, "SceneNear").c_str());
    }();
    return s_near;
}

} // namespace wiz3d
