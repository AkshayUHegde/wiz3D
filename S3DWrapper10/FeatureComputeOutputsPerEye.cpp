/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <ComputeOutputsPerEye Value="1"/>: a UAV texture gets a right copy like a render target (tile light lists and culling
// grids the right dispatch rewrites); the right eye samples the left one until a right-eye UAV bind writes it.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class ComputeOutputsPerEyeFeature : public ProfileFeature
{
};

} // namespace

bool ComputeOutputsPerEyeOn()
{
    static const bool s_on = ProfileSwitchOn("ComputeOutputsPerEye");
    return s_on;
}

std::unique_ptr<ProfileFeature> MakeComputeOutputsPerEye(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new ComputeOutputsPerEyeFeature());
}

} // namespace wiz3d
