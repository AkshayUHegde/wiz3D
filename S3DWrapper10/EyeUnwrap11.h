/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Each eye's real object behind a game pointer: the rules the immediate context draws with (Context11Proxy.cpp),
// shared with the deferred contexts' right-eye twin.

#pragma once

// Declared, not included: d3d11.h needs Device11Proxy.h's DXGI_RGBA shim first.
struct ID3D11Buffer;
struct ID3D11Resource;
struct ID3D11ShaderResourceView;
struct ID3D11UnorderedAccessView;
struct ID3D11View;

ID3D11Buffer*             UnwrapBuf(ID3D11Buffer* p);
ID3D11Resource*            UnwrapResourceForEye(ID3D11Resource* p, bool pickRight);
ID3D11ShaderResourceView*  UnwrapSRVForEye(ID3D11ShaderResourceView* p, bool pickRight);
ID3D11UnorderedAccessView* UnwrapUAVForEye(ID3D11UnorderedAccessView* p, bool pickRight);

namespace wiz3d
{
ID3D11View*     UnwrapViewForEye(ID3D11View* p, bool pickRight);
// A buffer's right sibling or a stereo texture's right copy, else the resource itself.
ID3D11Resource* RightEyeResource(ID3D11Resource* p);
}
