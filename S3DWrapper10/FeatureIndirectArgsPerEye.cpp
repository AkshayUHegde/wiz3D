/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <IndirectArgsPerEye Value="1"/>: a CopyStructureCount destination without a right sibling gets one, seeded from the
// left, so a right indirect dispatch runs over the right eye's count (Hardline's tiled lighting).

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class IndirectArgsPerEyeFeature : public ProfileFeature
{
};

} // namespace

bool IndirectArgsPerEyeOn()
{
    static const bool s_on = ProfileSwitchOn("IndirectArgsPerEye");
    return s_on;
}

std::unique_ptr<ProfileFeature> MakeIndirectArgsPerEye(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new IndirectArgsPerEyeFeature());
}

} // namespace wiz3d
