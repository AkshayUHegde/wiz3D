/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <KeepBareProjections Value="1"/>: a pixel or compute shader's bare projection (view space to clip, no camera rotation)
// projects positions the shader rebuilt from its own eye's depth, so the right eye keeps the game's (Andromeda's reflections).

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class KeepBareProjectionsFeature : public ProfileFeature
{
public:
    bool KeepBareProjections() const override { return true; }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeKeepBareProjections(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new KeepBareProjectionsFeature());
}

} // namespace wiz3d
