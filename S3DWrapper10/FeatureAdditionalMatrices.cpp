/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <AdditionalMatrixName Value="matrix"/>: a position matrix wherever a shader reads it, even where the analyzer's trace
// misses it (Max Payne 3's particle sprites, gWorldViewProj).

#include "StdAfx.h"
#include <algorithm>
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

// <AdditionalMatrixLayout Value="code"/>, after AdditionalMatricesFeature: a named matrix whose first two registers the
// code reads whole in dp4s is patched as rows (a per-object velocity's), otherwise as columns.
class AdditionalMatrixLayoutFeature : public ProfileFeature
{
public:
    void OnShaderCreated(const RdefView&, ShaderAnalysis11Result& info) const override
    {
        if (!info.parsed || info.namedMatrices.empty() || !info.bytecode) return;
        std::vector<std::pair<int, UINT>> dp4;
        CodeDp4ConstantReads(info.bytecode->data(), info.bytecode->size(), dp4);
        auto reads = [&](int slot, UINT reg) { return std::find(dp4.begin(), dp4.end(), std::make_pair(slot, reg)) != dp4.end(); };
        for (const auto& m : info.namedMatrices)
        {
            if (!reads(m.first, m.second) || !reads(m.first, m.second + 1)) continue;
            auto it = info.projection.matrixData.cb.find((DWORD)m.first);
            if (it == info.projection.matrixData.cb.end()) continue;
            for (auto& pmd : it->second)
            {
                if (pmd.matrixRegister != m.second || pmd.matrixIsTransposed) continue;
                pmd.matrixIsTransposed = TRUE;
                FeatureLog("  AdditionalMatrixLayout: CRC=0x%08lX matrix cb%d reg %u read by dp4: patched as rows\n", info.crc32,
                           m.first, m.second);
            }
        }
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeAdditionalMatrices(const ProfileEntry*)
{
    if (g_ProfileData.AdditionalMatrixName.empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new AdditionalMatricesFeature());
}

std::unique_ptr<ProfileFeature> MakeAdditionalMatrixLayout(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") != "code" || g_ProfileData.AdditionalMatrixName.empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new AdditionalMatrixLayoutFeature());
}

} // namespace wiz3d
