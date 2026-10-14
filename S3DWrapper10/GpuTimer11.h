/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// CollectDebugInformation: GPU time of our own work per frame through D3D11 timestamp queries (right-eye replays, right
// dispatches, the output pass), read back three frames later so nothing waits for the GPU.

#pragma once

#include <windows.h>
#include <d3d9types.h>   // D3DCOLORVALUE for the DXGI_RGBA shim below
#ifndef _DXGI_RGBA_DEFINED
#define _DXGI_RGBA_DEFINED
typedef D3DCOLORVALUE DXGI_RGBA;
#endif
#include <d3d11.h>
#include <string>
#include <vector>

namespace wiz3d
{

class GpuTimer11
{
public:
    // kLeft: the game's own work since the last span, recorded by Begin and named after the span that follows it.
    enum Kind { kReplay, kRightDispatch, kOutput, kLeft, kKinds };

    explicit GpuTimer11(ID3D11Device* device);
    ~GpuTimer11();

    // A frame's timestamps run from FrameBegin to FrameEnd; a span is a Begin/End pair inside it.
    void FrameBegin(ID3D11DeviceContext* c);
    void FrameEnd(ID3D11DeviceContext* c);
    UINT Begin(ID3D11DeviceContext* c);
    void End(ID3D11DeviceContext* c, UINT begin, Kind kind, const std::string& label, unsigned draws);
    // The oldest finished frame as one log row (frame = the frame being reported); nothing when it is not ready.
    void Report(ID3D11DeviceContext* c, unsigned frame);

private:
    struct Span { UINT b, e; Kind kind; std::string label; unsigned draws; };
    struct Slot
    {
        ID3D11Query* disjoint = nullptr;
        std::vector<ID3D11Query*> stamps;
        UINT used = 0, last = 0;   // last: the stamp the previous span ended on (0 = the frame's start)
        std::vector<Span> spans;
        bool open = false, done = false;
    };
    UINT Stamp(ID3D11DeviceContext* c);

    static const UINT kSlots = 4, kMaxStamps = 2048;
    ID3D11Device* m_device;
    Slot          m_slot[kSlots];
    UINT          m_cur = 0;
    bool          m_inFrame = false;
};

} // namespace wiz3d
