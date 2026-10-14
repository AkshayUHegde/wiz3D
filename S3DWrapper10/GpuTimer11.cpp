/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// GpuTimer11.h: a disjoint query and a timestamp pool per frame in a ring of four; Report reads the frame three behind
// with DONOTFLUSH and skips it when the GPU has not finished it.

#include "StdAfx.h"
#include "GpuTimer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <algorithm>

namespace wiz3d
{

GpuTimer11::GpuTimer11(ID3D11Device* device)
    : m_device(device)
{
    if (m_device) m_device->AddRef();
}

GpuTimer11::~GpuTimer11()
{
    for (Slot& s : m_slot)
    {
        if (s.disjoint) s.disjoint->Release();
        for (ID3D11Query* q : s.stamps) if (q) q->Release();
    }
    if (m_device) m_device->Release();
}

UINT GpuTimer11::Stamp(ID3D11DeviceContext* c)
{
    Slot& s = m_slot[m_cur];
    if (!m_inFrame || s.used >= kMaxStamps) return ~0u;
    if (s.used == s.stamps.size())
    {
        D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP, 0 };
        ID3D11Query* q = nullptr;
        if (FAILED(m_device->CreateQuery(&qd, &q)) || !q) return ~0u;
        s.stamps.push_back(q);
    }
    c->End(s.stamps[s.used]);
    return s.used++;
}

void GpuTimer11::FrameBegin(ID3D11DeviceContext* c)
{
    m_cur = (m_cur + 1) % kSlots;
    Slot& s = m_slot[m_cur];
    if (!s.disjoint)
    {
        D3D11_QUERY_DESC qd = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
        if (FAILED(m_device->CreateQuery(&qd, &s.disjoint))) s.disjoint = nullptr;
    }
    if (!s.disjoint) return;
    s.used = s.last = 0;
    s.spans.clear();
    s.done = false;
    c->Begin(s.disjoint);
    m_inFrame = true;
    Stamp(c);   // stamp 0: the frame's start
}

void GpuTimer11::FrameEnd(ID3D11DeviceContext* c)
{
    if (!m_inFrame) return;
    Slot& s = m_slot[m_cur];
    Stamp(c);   // the last stamp: the frame's end
    c->End(s.disjoint);
    s.done = true;
    m_inFrame = false;
}

UINT GpuTimer11::Begin(ID3D11DeviceContext* c)
{
    const UINT b = Stamp(c);
    // The game's work since the previous span: the left eye of the pass about to replay.
    if (b != ~0u) m_slot[m_cur].spans.push_back(Span{ m_slot[m_cur].last, b, kLeft, std::string(), 0 });
    return b;
}

void GpuTimer11::End(ID3D11DeviceContext* c, UINT begin, Kind kind, const std::string& label, unsigned draws)
{
    if (begin == ~0u) return;
    const UINT e = Stamp(c);
    if (e == ~0u) return;
    Slot& s = m_slot[m_cur];
    if (!s.spans.empty() && s.spans.back().kind == kLeft && s.spans.back().e == begin) s.spans.back().label = label;
    s.spans.push_back(Span{ begin, e, kind, label, draws });
    s.last = e;
}

void GpuTimer11::Report(ID3D11DeviceContext* c, unsigned frame)
{
    Slot& s = m_slot[(m_cur + 1) % kSlots];   // three frames behind the one just ended
    if (!s.done || !s.disjoint || s.used < 2) return;
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = {};
    if (c->GetData(s.disjoint, &dj, sizeof(dj), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK || dj.Disjoint || !dj.Frequency) return;
    std::vector<UINT64> t(s.used, 0);
    for (UINT i = 0; i < s.used; ++i)
        if (c->GetData(s.stamps[i], &t[i], sizeof(UINT64), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) return;
    const double ms = 1000.0 / (double)dj.Frequency;
    auto span = [&](UINT b, UINT e) { return t[e] > t[b] ? (double)(t[e] - t[b]) * ms : 0.0; };
    const double total = span(0, s.used - 1);
    double sum[kKinds] = {};
    unsigned count[kKinds] = {};
    for (const Span& p : s.spans) { sum[p.kind] += span(p.b, p.e); ++count[p.kind]; }
    // Each replay with the left segment just before it: the same pass's left eye, plus anything since the last span.
    struct Pass { double left, right; const Span* replay; };
    std::vector<Pass> passes;
    for (size_t i = 0; i < s.spans.size(); ++i)
    {
        if (s.spans[i].kind != kReplay) continue;
        const bool hasLeft = i > 0 && s.spans[i - 1].kind == kLeft;
        passes.push_back({ hasLeft ? span(s.spans[i - 1].b, s.spans[i - 1].e) : 0.0, span(s.spans[i].b, s.spans[i].e), &s.spans[i] });
    }
    std::sort(passes.begin(), passes.end(), [](const Pass& a, const Pass& b) { return a.left + a.right > b.left + b.right; });
    std::string top;
    char buf[160];
    for (size_t i = 0; i < passes.size() && i < 8; ++i)
    {
        sprintf_s(buf, " %s %u draws L %.2f / R %.2f ms;", passes[i].replay->label.c_str(), passes[i].replay->draws,
                  passes[i].left, passes[i].right);
        top += buf;
    }
    const double spans = sum[kReplay] + sum[kRightDispatch] + sum[kOutput] + sum[kLeft];
    DDILog("  [frame %u] GPU time (ms, a frame 3 back): frame %.2f | right-eye replays %.2f (%u) | right dispatches %.2f (%u) |"
           " output pass %.2f | left eye before them %.2f, after the last %.2f | costliest passes (left segment / right"
           " replay):%s\n", frame, total, sum[kReplay], count[kReplay], sum[kRightDispatch], count[kRightDispatch],
           sum[kOutput], sum[kLeft], total - spans, top.empty() ? " none" : top.c_str());
}

} // namespace wiz3d
