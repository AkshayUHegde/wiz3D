/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <ConstantBlocks Value="1"/>: every constant block a right draw or dispatch reads, a small buffer whole (under
// PatchConstantsAtDraw) or the range of an 11.1 ring constant buffer, rebuilt for the right eye in one place.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class ConstantBlocksFeature : public ProfileFeature
{
};

} // namespace

bool ConstantBlocksOn()
{
    static const bool s_on = ProfileSwitchOn("ConstantBlocks");
    return s_on;
}

std::unique_ptr<ProfileFeature> MakeConstantBlocks(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new ConstantBlocksFeature());
}

} // namespace wiz3d
