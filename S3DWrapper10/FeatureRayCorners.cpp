/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <RayCorners Buffer="cbuffer" Value="float3 array" Count="4"/>: frustum-corner rays a screen pass rebuilds position from;
// for the right camera every corner moves by -b * (corner[1] - corner[0]) / 2.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Device11Proxy.h"
#include "Buffer11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <mutex>

namespace wiz3d
{
namespace
{

std::mutex s_rayLock;
int        s_rayOffset = -1;   // byte offset of element 0; elements are a register apart

class RayCornersFeature : public ProfileFeature
{
public:
    explicit RayCornersFeature(const ProfileEntry& e)
        : m_buffer(EntryValue(e, "Buffer")), m_name(EntryValue(e, "Value")),
          m_count(EntryValue(e, "Count").empty() ? 4 : atoi(EntryValue(e, "Count").c_str())) {}

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        const int slot = rdef.CBufferSlot(m_buffer);
        if (slot < 0) return;
        info.NoteRightCB(m_buffer, slot);
        std::lock_guard<std::mutex> g(s_rayLock);
        if (s_rayOffset >= 0) return;
        for (const auto& v : rdef.Variables(m_name, false))
            if (v.first == slot)
            {
                s_rayOffset = (int)v.second;
                FeatureLog("  RayCorners: %s.%s at +%d (cb%d in CRC=0x%08lX), %d corners\n", m_buffer.c_str(), m_name.c_str(),
                           s_rayOffset, slot, info.crc32, m_count);
            }
    }

    bool EditRightConstants(Context11Proxy& ctx, const std::string& cbuffer, unsigned char* data, size_t size) override
    {
        if (cbuffer != m_buffer || s_rayOffset < 0 || m_count < 2) return false;
        if ((size_t)s_rayOffset + (size_t)m_count * 16 > size) return false;
        CameraMove cam;
        if (!ctx.BoundCameraMove(cam)) { ++m_skipped; ctx.RetryEdit(); return false; }
        const float bs = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign() * cam.p00;
        const float* c0 = (const float*)(data + s_rayOffset);
        const float* c1 = (const float*)(data + s_rayOffset + 16);
        const float delta[3] = { -bs * (c1[0] - c0[0]) * 0.5f, -bs * (c1[1] - c0[1]) * 0.5f, -bs * (c1[2] - c0[2]) * 0.5f };
        for (int i = 0; i < m_count; ++i)
        {
            float* c = (float*)(data + s_rayOffset + i * 16);
            for (int j = 0; j < 3; ++j) c[j] += delta[j];
        }
        ++(ctx.EditAtDispatch() ? m_atDispatch : m_atDraw);
        return true;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] ray corners (this frame): moved at draws %u, at dispatches %u, skipped %u (no camera yet)\n",
                   frame, m_atDraw, m_atDispatch, m_skipped);
        m_atDraw = m_atDispatch = m_skipped = 0;
    }

private:
    std::string m_buffer, m_name;
    int         m_count;
    unsigned    m_atDraw = 0, m_atDispatch = 0, m_skipped = 0;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeRayCorners(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Buffer").empty() || EntryValue(*e, "Value").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new RayCornersFeature(*e));
}

} // namespace wiz3d
