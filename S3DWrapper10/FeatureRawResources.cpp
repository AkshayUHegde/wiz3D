/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <RawResources Value="1"/>: a texture the game reached past the wrapper (its real left copy) stands for its proxy in
// copies, so the right eye's copy is filled from the right eye's source (Ryse's refraction copy of the scene).

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "AdapterFunctions.h"              // DDILog

namespace wiz3d
{
namespace
{

volatile LONG s_resolved = 0;

class RawResourcesFeature : public ProfileFeature
{
public:
    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        const LONG n = InterlockedExchange(&s_resolved, 0);
        if (report)
            DDILog("  [frame %u] raw resources (this frame): copies whose real texture stood for its proxy %ld\n", frame, n);
    }
};

} // namespace

bool RawResourcesOn()
{
    static const bool s_on = ProfileSwitchOn("RawResources");
    return s_on;
}

void NoteRawResolved() { InterlockedIncrement(&s_resolved); }

std::unique_ptr<ProfileFeature> MakeRawResources(const ProfileEntry* entry)
{
    if (!entry || EntryValue(*entry, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new RawResourcesFeature());
}

} // namespace wiz3d
