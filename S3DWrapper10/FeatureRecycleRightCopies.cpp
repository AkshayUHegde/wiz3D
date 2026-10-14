/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <RecycleRightCopies Value="1"/>: a freed texture's right copy is kept and serves the next texture created with the same
// desc two Presents later, so a game that remakes its targets every frame (Edith Finch) stops paying for a second one.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class RecycleRightCopiesFeature : public ProfileFeature
{
};

} // namespace

bool RecycleRightCopiesOn()
{
    static const bool s_on = ProfileSwitchOn("RecycleRightCopies");
    return s_on;
}

std::unique_ptr<ProfileFeature> MakeRecycleRightCopies(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new RecycleRightCopiesFeature());
}

} // namespace wiz3d
