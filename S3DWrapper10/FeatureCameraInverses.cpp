/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <CameraInverses Value="1"/>: a declared unprojection whose forward is camera-shaped counts as an inverse whatever space
// it maps to, and every inverse is patched with the focal of the camera it inverts, not the last one cached.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class CameraInversesFeature : public ProfileFeature
{
};

} // namespace

bool CameraInversesOn()
{
    static const bool s_on = ProfileSwitchOn("CameraInverses");
    return s_on;
}

std::unique_ptr<ProfileFeature> MakeCameraInverses(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new CameraInversesFeature());
}

} // namespace wiz3d
