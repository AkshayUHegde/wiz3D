/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <DuniaLocalLights Buffer Array Count ViewPosition/>: Dunia local lights given relative to the left camera, moved by
// m = d*R for a moved-origin pass (5 float4s a light; 4.w < 0 spot: translation += M*m; omni: position -= m).

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <mutex>

namespace wiz3d
{
namespace
{

std::mutex s_lightsLock;
int        s_arrayOffset = -1, s_viewPosOffset = -1;

class DuniaLocalLightsFeature : public ProfileFeature
{
public:
    explicit DuniaLocalLightsFeature(const ProfileEntry& e)
        : m_buffer(EntryValue(e, "Buffer")), m_array(EntryValue(e, "Array")), m_viewPos(EntryValue(e, "ViewPosition")),
          m_count(EntryValue(e, "Count").empty() ? 15 : atoi(EntryValue(e, "Count").c_str())) {}

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        const int slot = rdef.CBufferSlot(m_buffer);
        if (slot < 0) return;
        info.NoteRightCB(m_buffer, slot);
        std::lock_guard<std::mutex> g(s_lightsLock);
        if (s_arrayOffset >= 0) return;
        for (const auto& v : rdef.Variables(m_array, false)) if (v.first == slot) s_arrayOffset = (int)v.second;
        if (!m_viewPos.empty())
            for (const auto& v : rdef.Variables(m_viewPos, false)) if (v.first == slot) s_viewPosOffset = (int)v.second;
        if (s_arrayOffset >= 0)
            FeatureLog("  DuniaLocalLights: %s.%s at +%d (%d lights), %s at +%d (cb%d in CRC=0x%08lX)\n", m_buffer.c_str(),
                       m_array.c_str(), s_arrayOffset, m_count / 5, m_viewPos.c_str(), s_viewPosOffset, slot, info.crc32);
    }

    bool EditRightConstants(Context11Proxy& ctx, const std::string& cbuffer, unsigned char* data, size_t size) override
    {
        // Only where the origin moved with the right camera; elsewhere the lights' left-relative data still holds.
        if (cbuffer != m_buffer || s_arrayOffset < 0 || !ctx.MovedOrigin()) return false;
        CameraMove cam;
        if (!ctx.BoundCameraMove(cam)) { ++m_skipped; ctx.RetryEdit(); return false; }
        auto element = [&](int light, int e) -> float*
        {
            const size_t o = (size_t)s_arrayOffset + ((size_t)light * 5 + (size_t)e) * 16;
            return o + 16 <= size ? (float*)(data + o) : nullptr;
        };
        for (int i = 0; i < m_count / 5; ++i)
        {
            float* kind = element(i, 4);
            if (!kind || kind[3] == 0.f) continue;
            if (kind[3] < 0.f)
            {
                float* t = element(i, 3);
                for (int k = 0; k < 3 && t; ++k)
                    if (const float* row = element(i, k)) for (int c = 0; c < 4; ++c) t[c] += cam.world[k] * row[c];
                ++m_spots;
            }
            else if (float* p = element(i, 0))
            {
                for (int k = 0; k < 3; ++k) p[k] -= cam.world[k];
                ++m_omnis;
            }
        }
        if (s_viewPosOffset >= 0 && (size_t)s_viewPosOffset + 12 <= size) ((float*)(data + s_viewPosOffset))[0] -= cam.d;
        ++m_edited;
        return true;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] dunia local lights (this frame): blocks moved %u (spots %u, omnis %u), skipped %u (no"
                   " camera yet)\n", frame, m_edited, m_spots, m_omnis, m_skipped);
        m_edited = m_spots = m_omnis = m_skipped = 0;
    }

private:
    std::string m_buffer, m_array, m_viewPos;
    int         m_count;
    unsigned    m_edited = 0, m_spots = 0, m_omnis = 0, m_skipped = 0;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeDuniaLocalLights(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Buffer").empty() || EntryValue(*e, "Array").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new DuniaLocalLightsFeature(*e));
}

} // namespace wiz3d
