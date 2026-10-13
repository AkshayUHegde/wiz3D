/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <Factory2SwapChains Value="1"/>: a swap chain created with CreateSwapChainForHwnd (or ForCoreWindow, ForComposition) on
// a factory from CreateDXGIFactory or CreateDXGIFactory1 is wrapped, as one from CreateDXGIFactory2 is (Mass Effect LE).

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class Factory2SwapChainsFeature : public ProfileFeature
{
};

} // namespace

bool Factory2SwapChainsOn()
{
    static const bool s_on = ProfileSwitchOn("Factory2SwapChains");
    return s_on;
}

std::unique_ptr<ProfileFeature> MakeFactory2SwapChains(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new Factory2SwapChainsFeature());
}

} // namespace wiz3d
