/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <AdditionalMatrixName Value="matrix"/>: a position matrix wherever a shader reads it, even where the analyzer's trace
// misses it (Max Payne 3's particle sprites, gWorldViewProj).

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"              // DDILog
#include "..\S3DAPI\ShaderProfileData.h"   // g_ProfileData

namespace wiz3d
{
namespace
{

class AdditionalMatricesFeature : public ProfileFeature
{
public:
    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        for (const std::string& name : g_ProfileData.AdditionalMatrixName)
            for (const auto& v : rdef.Variables(name, true))
                info.namedMatrices.push_back(std::make_pair(v.first, v.second / 16));
        if (!info.parsed) return;
        for (const auto& m : info.namedMatrices)
        {
            auto& targets = info.projection.matrixData.cb[(DWORD)m.first];
            bool have = false;
            for (const auto& p : targets) if (p.matrixRegister == m.second) have = true;
            if (have) continue;
            ProjectionMatrixData pmd;
            pmd.matrixRegister = m.second;
            targets.push_back(pmd);
            info.projection.matrixData.mask |= 1u << m.first;
            FeatureLog("  AdditionalMatrixName: CRC=0x%08lX matrix cb%d reg %u added from RDEF (trace targets in %u other cb"
                       " slot(s))\n", info.crc32, m.first, m.second, (unsigned)info.projection.matrixData.cb.size() - 1);
        }
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeAdditionalMatrices(const ProfileEntry*)
{
    if (g_ProfileData.AdditionalMatrixName.empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new AdditionalMatricesFeature());
}

} // namespace wiz3d
