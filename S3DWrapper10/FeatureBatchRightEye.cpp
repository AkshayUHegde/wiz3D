/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The right eye's draws are recorded (RightRecorder11) and replayed on the immediate context once per pass, as DX12's
// right command list, instead of alternating eyes at every draw. On for every game; <BatchRightEye Value="0"/> turns it off.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "..\S3DAPI\ShaderProfileData.h"   // g_ProfileData

namespace wiz3d
{
namespace
{

class BatchRightEyeFeature : public ProfileFeature
{
};

} // namespace

bool BatchRightEyeOn()
{
    static const bool s_on = [] {
        const auto it = g_ProfileData.FeatureEntries.find("BatchRightEye");
        return it == g_ProfileData.FeatureEntries.end() || EntryValue(it->second, "Value") != "0";
    }();
    return s_on;
}

std::unique_ptr<ProfileFeature> MakeBatchRightEye(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new BatchRightEyeFeature());
}

} // namespace wiz3d
