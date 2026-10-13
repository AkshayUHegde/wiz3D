/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <ClipSpaceSprites Value="1|2|3"/>: a vertex shader passing CPU-projected NDC (Frostbite's flares) moves for the right
// eye by the eye offset at the depth under it; 2 = at infinity; 3 = as 2, plus world-position sprites at their depth.

#include "StdAfx.h"
#include <stdlib.h>
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "AdapterFunctions.h"   // DDILog

namespace wiz3d
{
namespace
{

class ClipSpaceSpritesFeature : public ProfileFeature
{
public:
    explicit ClipSpaceSpritesFeature(int mode) : m_mode(mode) {}

    void OnRightDraw(Context11Proxy& ctx) override { ctx.BindClipSpaceShift(m_mode); }
    void OnRightDrawDone(Context11Proxy& ctx) override { ctx.UnbindClipSpaceShift(); }

    void OnFrameEnd(Context11Proxy& ctx, unsigned frame, bool report) override
    {
        unsigned shifted = 0, noCamera = 0;
        ctx.ClipSpaceCounts(shifted, noCamera);
        const unsigned twins = ctx.SpriteTwinCount();
        if (report)
            DDILog("  [frame %u] clip-space sprites: right draws moved by their depth %u (by their world depth %u), left"
                   " as the game's for want of a scene camera %u\n", frame, shifted, twins, noCamera);
    }

private:
    int m_mode;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeClipSpaceSprites(const ProfileEntry* e)
{
    if (!e) return nullptr;
    const int mode = atoi(EntryValue(*e, "Value").c_str());
    if (mode <= 0) return nullptr;
    return std::unique_ptr<ProfileFeature>(new ClipSpaceSpritesFeature(mode));
}

} // namespace wiz3d
