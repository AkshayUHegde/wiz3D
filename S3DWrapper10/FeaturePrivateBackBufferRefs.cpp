/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <PrivateBackBufferRefs Value="1"/>: our holds on the back buffer (the swap chain's, its views') stay out of the count
// AddRef/Release return, as DXGI's and D3D11's do; UE4 checks the count is 1 before ResizeBuffers (Edith Finch).

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class PrivateBackBufferRefsFeature : public ProfileFeature
{
};

} // namespace

bool PrivateBackBufferRefsOn()
{
    static const bool s_on = ProfileSwitchOn("PrivateBackBufferRefs");
    return s_on;
}

std::unique_ptr<ProfileFeature> MakePrivateBackBufferRefs(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new PrivateBackBufferRefsFeature());
}

} // namespace wiz3d
