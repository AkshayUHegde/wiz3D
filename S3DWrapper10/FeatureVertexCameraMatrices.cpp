/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <VertexCameraMatrices Value="1"/>: a vertex shader's declared float4x4 outside its position chain is matched against
// the camera, current or last frame's, at the right draw and shifted with it. 4A's per-object motion blur reads
// mblur_previous beside m_VP: with the current clip shifted and the previous not, every moving object smears.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class VertexCameraMatricesFeature : public ProfileFeature
{
public:
    bool VertexCameraMatrices() const override { return true; }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeVertexCameraMatrices(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new VertexCameraMatricesFeature());
}

} // namespace wiz3d
