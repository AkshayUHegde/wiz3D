/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <ViewMatrices Value="1"/>: a pixel or compute shader without RDEF that reads the camera's view as mul/mad columns gets
// the right camera's: its x translation moves by the eye offset. Andromeda's decals cast their rays from it.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class ViewMatricesFeature : public ProfileFeature
{
public:
    bool ViewMatrices() const override { return true; }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeViewMatrices(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new ViewMatricesFeature());
}

} // namespace wiz3d
