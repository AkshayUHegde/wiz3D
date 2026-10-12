/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <DeferredContexts Value="1"/>: each deferred context the game creates records a right-eye twin beside its own, as a
// DX12 command list does; ExecuteCommandList runs the left list, then the right half. Off: the real deferred context.

#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vector>

// Declared, not included: d3d11.h needs Device11Proxy.h's DXGI_RGBA shim first.
struct ID3D11DeviceContext;
struct ID3D11CommandList;

namespace wiz3d
{

class Device11Proxy;
class Buffer11Proxy;

// The profile names the element and draws are duplicated per eye.
bool DeferredContextsOn();

// CreateDeferredContext, 1, 2 and 3 (riid names the interface asked for): a recording pair when on, else the real context.
HRESULT CreateDeferredContextFor(Device11Proxy* device, UINT flags, REFIID riid, void** ppContext);

// ExecuteCommandList on a real context: a pair's left list, then its right half when it kept one; any other list as is.
void ExecuteCommandListPair(ID3D11DeviceContext* real, ID3D11CommandList* list, BOOL restore);
// The right-eye replay pass: the right half alone when there is one, else the list itself.
void ExecuteCommandListRight(ID3D11DeviceContext* real, ID3D11CommandList* list, BOOL restore);
// The buffers a pair's recording wrote, whose bytes change when it runs; empty for any other list.
const std::vector<Buffer11Proxy*>& CommandListWrittenBuffers(ID3D11CommandList* list);

// What the deferred contexts did since the last call (pairs and failed twins: totals), for the feature's row.
struct DeferredStats
{
    LONG pairs, rightFailed;
    LONG finished, rightKept, rightDropped, executed, rightMapFailed;
};
DeferredStats TakeDeferredStats();

} // namespace wiz3d
