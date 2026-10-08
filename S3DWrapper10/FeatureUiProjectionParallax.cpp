/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <UiProjectionParallax Value="1"/>: a clean projection whose focal lies beyond 2x from the scene camera's is a UI
// projection, not another FOV (a weapon's stays within), and its draw takes the scene camera's parallax. 4A's 3D menu
// draws the selected item with its own projection; it moved a seventh as far as the others.

#include "StdAfx.h"
#include "ProfileFeature.h"

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

} // namespace wiz3d
