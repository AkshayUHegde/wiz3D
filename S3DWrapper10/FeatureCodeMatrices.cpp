/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <CodeMatrices Value="1"/>: a shader shipped without reflection (RDEF stripped) has the matrices it reads found from its
// code: a vertex shader's from the dp4 that build its outputs, a pixel or compute shader's from dp4 and mul/mad runs.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class CodeMatricesFeature : public ProfileFeature
{
};

} // namespace

bool CodeMatricesOn()
{
    static const bool s_on = ProfileSwitchOn("CodeMatrices");
    return s_on;
}

std::unique_ptr<ProfileFeature> MakeCodeMatrices(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new CodeMatricesFeature());
}

} // namespace wiz3d
