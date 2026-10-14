/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <StereoTexture Value="texture" EyeOffset="viewInverse" TanX="float4"/>: our per-eye 3D Vision stereo texture, texel (0,0) =
// (sep, conv), in the slot the game's shaders read it from: zeros for the left eye, sep = b for the right.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Device11Proxy.h"
#include "Buffer11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <d3dcompiler.h>
#include <sstream>

namespace wiz3d
{
namespace
{

// The t-slot a shader samples at the 3D Vision stereo texel (0.0625, 0.5), from its disassembly; -1 = none.
int TexelSlot(const std::vector<BYTE>& code)
{
    ID3DBlob* text = nullptr;
    if (code.empty() || FAILED(D3DDisassemble(code.data(), code.size(), 0, nullptr, &text)) || !text) return -1;
    std::istringstream in(std::string((const char*)text->GetBufferPointer(), text->GetBufferSize()));
    text->Release();
    std::string line;
    while (std::getline(in, line))
    {
        if (line.find("l(0.062500, 0.500000") == std::string::npos || line.find("sample") == std::string::npos) continue;
        for (size_t p = line.find(" t"); p != std::string::npos; p = line.find(" t", p + 1))
            if (p + 2 < line.size() && isdigit((unsigned char)line[p + 2])) return atoi(line.c_str() + p + 2);
    }
    return -1;
}

class StereoTextureFeature : public ProfileFeature
{
public:
    // Value: texture names; "CRC:slot"; "3DVisionTexel" = every stripped shader sampling the 3D Vision stereo texel
    // (0.0625, 0.5), as Far Cry 4's GameWorks ShadowLib (soft shadows, PCSS) does.
    explicit StereoTextureFeature(const ProfileEntry& e)
        : m_eyeOffset(EntryValue(e, "EyeOffset")), m_tanX(EntryValue(e, "TanX"))
    {
        std::istringstream in(EntryValue(e, "Value"));
        std::string n;
        while (in >> n)
        {
            if (n == "3DVisionTexel") m_texel = true;
            else if (n.size() > 9 && n[8] == ':') m_crcSlots.emplace_back((DWORD)strtoul(n.substr(0, 8).c_str(), nullptr, 16), atoi(n.c_str() + 9));
            else m_names.push_back(n);
        }
    }
    ~StereoTextureFeature() override { Release(); }

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        for (const std::string& n : m_names) if (info.stereoParmsSlot < 0) info.stereoParmsSlot = rdef.TextureSlot(n);
        for (const auto& cs : m_crcSlots) if (cs.first == info.crc32) info.stereoParmsSlot = cs.second;
        if (m_texel && info.stereoParmsSlot < 0 && rdef.Names().empty() && info.bytecode) info.stereoParmsSlot = TexelSlot(*info.bytecode);
        if (info.stereoParmsSlot >= 0) rdef.Variable(m_eyeOffset, false, info.viewInvCB, info.viewInvOffset);
        rdef.Variable(m_tanX, true, info.projParamsCB, info.projParamsOffset);
        if (info.stereoParmsSlot >= 0)
            FeatureLog("  StereoTexture: CRC=0x%08lX reads the stereo texture at t%d, %s cb%d+%u\n", info.crc32,
                       info.stereoParmsSlot, m_eyeOffset.c_str(), info.viewInvCB, info.viewInvOffset);
    }

    void OnLeftDraw(Context11Proxy& ctx) override  { Bind(ctx, false); }
    void OnRightDraw(Context11Proxy& ctx) override { Bind(ctx, true); }

    // The game's view back in the slots we took, after both eyes drew.
    void OnDrawDone(Context11Proxy& ctx) override
    {
        if (m_bound[0] >= 0) { ID3D11ShaderResourceView* s = ctx.LeftSRV(false, (UINT)m_bound[0]); SetSRV(ctx, false, m_bound[0], s); }
        if (m_bound[1] >= 0) { ID3D11ShaderResourceView* s = ctx.LeftSRV(true,  (UINT)m_bound[1]); SetSRV(ctx, true, m_bound[1], s); }
        m_bound[0] = m_bound[1] = -1;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] stereo texture (this frame): draws %u, right-eye binds %u, texture updates %u, no eye-offset"
                   " matrix %u | right eye: pixel sep %.5f conv %.3f, eye offset (%.4f %.4f %.4f)\n",
                   frame, m_draws, m_rightBinds, m_updates, m_noAxis, m_right[0], m_right[1], m_right[2], m_right[3], m_right[4]);
        m_draws = m_rightBinds = m_updates = m_noAxis = 0;
    }

    float CameraP00() const override { return m_camP00; }

private:
    std::vector<std::string>              m_names;
    std::vector<std::pair<DWORD, int>>    m_crcSlots;
    bool                                  m_texel = false;
    std::string m_eyeOffset, m_tanX;
    // [0] left (zeros), [1] right for vertex shaders (sep 0), [2] right for pixel shaders (sep = b).
    ID3D11Texture2D*          m_tex[3] = {};
    ID3D11ShaderResourceView* m_srv[3] = {};
    float    m_right[8] = {};          // sep, conv, eye offset xyz: what [1]/[2] hold
    float    m_axis[3] = {};           // the last view-inverse row 0 read
    int      m_bound[2] = { -1, -1 };  // VS, PS slot taken this draw
    float    m_camP00 = 0.f;           // 1 / tanX from the last light; the cached projXScale drifts between cameras
    unsigned m_draws = 0, m_rightBinds = 0, m_noAxis = 0, m_updates = 0;

    // MP3's texture is 8x2 RGBA32F; 3D Vision's is 8x1 (ShadowLib samples v = 0.5, row 1 of an 8x2 = the zero eye offset).
    bool Ensure(Context11Proxy& ctx)
    {
        if (m_srv[0]) return true;
        ID3D11Device* dev = ctx.Parent() ? ctx.Parent()->GetReal() : nullptr;
        if (!dev) return false;
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = 8; td.Height = m_texel ? 1 : 2; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        const float zeros[64] = {};
        const D3D11_SUBRESOURCE_DATA srd = { zeros, 8 * 16, 0 };
        for (int i = 0; i < 3; ++i)
            if (FAILED(dev->CreateTexture2D(&td, &srd, &m_tex[i])) || FAILED(dev->CreateShaderResourceView(m_tex[i], nullptr, &m_srv[i])))
            {
                DDILog("  stereo texture: creating our per-eye textures FAILED\n");
                Release();
                return false;
            }
        memset(m_right, 0, sizeof m_right);
        return true;
    }

    void Release()
    {
        for (int i = 0; i < 3; ++i)
        {
            if (m_srv[i]) { m_srv[i]->Release(); m_srv[i] = nullptr; }
            if (m_tex[i]) { m_tex[i]->Release(); m_tex[i] = nullptr; }
        }
    }

    // Ours in the slot the bound VS / PS reads the stereo texture from.
    void Bind(Context11Proxy& ctx, bool right)
    {
        if (!gInfo.DuplicateDraws) return;
        const ShaderAnalysis11Result* vs = ctx.BoundShader(false);
        const ShaderAnalysis11Result* ps = ctx.BoundShader(true);
        const int vsSlot = (vs && vs->stereoParmsSlot >= 0 && vs->stereoParmsSlot < (int)Context11Proxy::kMaxSRVs) ? vs->stereoParmsSlot : -1;
        const int psSlot = (ps && ps->stereoParmsSlot >= 0 && ps->stereoParmsSlot < (int)Context11Proxy::kMaxSRVs) ? ps->stereoParmsSlot : -1;
        if ((vsSlot < 0 && psSlot < 0) || !Ensure(ctx)) return;
        if (!right) ++m_draws;
        else
        {
            ++m_rightBinds;
            // The camera's right axis: the view inverse's row 0 in the constants this draw binds.
            const ShaderAnalysis11Result* src = (ps && ps->viewInvCB >= 0) ? ps : (vs && vs->viewInvCB >= 0) ? vs : nullptr;
            Buffer11Proxy* cb = (src && src->viewInvCB < (int)Context11Proxy::kMaxCBs) ? ctx.BoundCB(src == ps, (UINT)src->viewInvCB) : nullptr;
            if (cb && cb->Shadow().size() >= src->viewInvOffset + 12) memcpy(m_axis, cb->Shadow().data() + src->viewInvOffset, 12);
            else ++m_noAxis;
            // The shader undoes (z - conv) * sep * tanX, so sep = eyeShift / tanX matches the geometry's eyeShift * (z - conv).
            float tanX = 0.f;
            Buffer11Proxy* pcb = (ps && ps->projParamsCB >= 0 && ps->projParamsCB < (int)Context11Proxy::kMaxCBs) ? ctx.BoundCB(true, (UINT)ps->projParamsCB) : nullptr;
            if (pcb && pcb->Shadow().size() >= ps->projParamsOffset + 4) memcpy(&tanX, pcb->Shadow().data() + ps->projParamsOffset, 4);
            const bool tanOk = tanX > 1e-3f && tanX < 100.f;
            if (tanOk) { m_camP00 = 1.f / tanX; stereo::PublishCameraP00(m_camP00); }
            // m41 = -eyeShift*conv*p00 is a camera moved eyeShift*conv along the right axis; b*w is the skew sep undoes.
            const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign(), conv = stereo::Convergence(), d = eyeShift * conv;
            // Without TanX: our last 1/tanX, else the scene camera's p00 (RightCamera, Far Cry 4), else the shear estimate.
            const float p00 = m_camP00 > 0.f ? m_camP00 : ctx.CameraP00();
            const float sep = tanOk ? eyeShift / tanX : (p00 > 0.f ? eyeShift * p00 : stereo::FallbackB(eyeShift));
            const float want[8] = { sep, conv, m_axis[0] * d, m_axis[1] * d, m_axis[2] * d, 0.f, 0.f, 0.f };
            if (memcmp(want, m_right, sizeof want) != 0)
            {
                memcpy(m_right, want, sizeof want);
                float rows[64];
                for (int t = 0; t < 2; ++t)
                {
                    for (int x = 0; x < 8; ++x)
                    {
                        float* r0 = rows + x * 4;
                        float* r1 = rows + 32 + x * 4;
                        r0[0] = t ? want[0] : 0.f; r0[1] = conv; r0[2] = 0.f; r0[3] = 0.f;
                        r1[0] = want[2]; r1[1] = want[3]; r1[2] = want[4]; r1[3] = 0.f;
                    }
                    ctx.Real()->UpdateSubresource(m_tex[1 + t], 0, nullptr, rows, 8 * 16, 0);
                }
                ++m_updates;
            }
        }
        ID3D11ShaderResourceView* vsSrv = m_srv[right ? 1 : 0];
        ID3D11ShaderResourceView* psSrv = m_srv[right ? 2 : 0];
        if (vsSlot >= 0) SetSRV(ctx, false, vsSlot, vsSrv);
        if (psSlot >= 0) SetSRV(ctx, true, psSlot, psSrv);
        m_bound[0] = vsSlot;
        m_bound[1] = psSlot;
    }

    // Around the left draw the right eye's recording (BatchRightEye) takes the same bind as the immediate context.
    static void SetSRV(Context11Proxy& ctx, bool pixel, int slot, ID3D11ShaderResourceView* v)
    {
        ID3D11DeviceContext* c[2] = { ctx.Real(), ctx.Mirror() };
        for (ID3D11DeviceContext* x : c)
        {
            if (!x) continue;
            if (pixel) x->PSSetShaderResources((UINT)slot, 1, &v);
            else       x->VSSetShaderResources((UINT)slot, 1, &v);
        }
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeStereoTexture(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new StereoTextureFeature(*e));
}

} // namespace wiz3d
