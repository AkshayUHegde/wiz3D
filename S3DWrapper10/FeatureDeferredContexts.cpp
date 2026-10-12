/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <DeferredContexts Value="1"/>: the game's deferred contexts record a right-eye twin (DeferredContext11.cpp); this
// feature reports, per frame, what they recorded and which right halves ran.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "DeferredContext11.h"
#include "AdapterFunctions.h"   // DDILog

namespace wiz3d
{
namespace
{

class DeferredContextsFeature : public ProfileFeature
{
public:
    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        const DeferredStats s = TakeDeferredStats();
        if (!report) return;
        DDILog("  [frame %u] deferred contexts (this frame): lists finished %ld, right halves kept %ld, dropped %ld (no"
               " right-eye work), executed %ld, right maps failed %ld | pairs %ld, twins not created %ld\n", frame,
               s.finished, s.rightKept, s.rightDropped, s.executed, s.rightMapFailed, s.pairs, s.rightFailed);
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeDeferredContexts(const ProfileEntry* entry)
{
    if (!entry || EntryValue(*entry, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new DeferredContextsFeature());
}

} // namespace wiz3d
