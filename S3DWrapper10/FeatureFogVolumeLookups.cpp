/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <FogVolumeLookups Value="1"/>: a volumetric fog grid built once, from the left camera, and looked up at a pixel's own
// screen position: the right eye's pixel shaders run a twin that moves the lookup to the left eye's position.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "FogLookup11.h"
#include "AdapterFunctions.h"   // DDILog
#include <algorithm>
#include <mutex>
#include <set>

namespace wiz3d
{
namespace
{

const char* const kKey = "fogvolumelookups";

std::mutex         s_logLock;
std::set<unsigned> s_logged;   // shader CRCs already logged

class FogVolumeLookupsFeature : public ProfileFeature
{
public:
    // Ndc="0": lookups whose x comes from NDC are left alone; with RightCamera SvPosition their true position,
    // projected by the game's view-projection, already lands on the left eye's cell.
    explicit FogVolumeLookupsFeature(const ProfileEntry& e) : m_ndc(EntryValue(e, "Ndc") != "0") {}

    void OnShaderCreated(const RdefView&, ShaderAnalysis11Result& info) const override
    {
        if (!info.bytecode) return;
        std::vector<FogLookup> lookups;
        if (!FindFogLookups(info.bytecode->data(), info.bytecode->size(), lookups)) return;
        if (!m_ndc)
        {
            lookups.erase(std::remove_if(lookups.begin(), lookups.end(), [](const FogLookup& l) { return l.svposToUv < 0; }),
                          lookups.end());
            if (lookups.empty()) return;
        }
        std::vector<int> v;
        for (const auto& l : lookups)
            v.insert(v.end(), { l.nth, l.reg, l.tslot, l.cu, l.cw, l.view, l.invGrid, l.zParams, l.svposToUv, l.svposComp });
        info.featureOffsets.emplace_back(kKey, v);
        info.keepCode = true;
        std::lock_guard<std::mutex> g(s_logLock);
        if (s_logged.insert(info.crc32).second)
            for (const auto& l : lookups)
                FeatureLog("  FogVolumeLookups: CRC=0x%08lX sample_l #%d r%d -> t%d, slice cb%d[%d]/[%d], x from %s\n", info.crc32,
                           l.nth, l.reg, l.tslot, l.view, l.zParams, l.invGrid, l.svposToUv >= 0 ? "pixels" : "NDC");
    }

    void OnRightDraw(Context11Proxy& ctx) override
    {
        const ShaderAnalysis11Result* ps = ctx.BoundShader(true);
        if (!ps) return;
        const std::vector<int>* a = nullptr;
        for (const auto& f : ps->featureOffsets) if (f.first == kKey) a = &f.second;
        if (!a || a->size() < (size_t)kFogLookupInts) return;
        const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign();
        const float p00 = ctx.CameraP00() > 0.f ? ctx.CameraP00() : stereo::SceneP00();
        if (eyeShift == 0.f) return;
        D3D11_VIEWPORT vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
        UINT n = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
        if (ctx.Recording()) n = ctx.GameViewports(vp);
        else ctx.Real()->RSGetViewports(&n, vp);
        if (!(p00 > 0.f) || !n || !(vp[0].Width > 0.f)) { ++m_noCamera; return; }
        std::vector<FogLookup> lookups;
        for (size_t i = 0; i + kFogLookupInts <= a->size(); i += kFogLookupInts)
        {
            const int* f = a->data() + i;
            lookups.push_back({ f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9] });
        }
        const float b = eyeShift * p00;
        const float k[4] = { 0.5f * b, stereo::Convergence(), 0.5f * b * vp[0].Width, 0.f };
        if (ctx.BindPixelTwin(Context11Proxy::kTwinFogVolume, FogLookupPlan(lookups), k, 4)) ++m_twinDraws;
        else ++m_refused;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] fog volume lookups (this frame): right draws by twin %u, twin refused %u, without a camera %u\n",
                   frame, m_twinDraws, m_refused, m_noCamera);
        m_twinDraws = m_refused = m_noCamera = 0;
    }

private:
    bool     m_ndc = true;
    unsigned m_twinDraws = 0, m_refused = 0, m_noCamera = 0;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeFogVolumeLookups(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new FogVolumeLookupsFeature(*e));
}

} // namespace wiz3d
