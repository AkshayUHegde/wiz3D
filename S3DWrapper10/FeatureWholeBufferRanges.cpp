/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <WholeBufferRanges Value="1"/>: a slot bound whole and rebound through *SetConstantBuffers1 has its size rounded up to
// 16 constants, since 11.1 drops the whole call when a range is not a multiple of 16.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class WholeBufferRangesFeature : public ProfileFeature
{
public:
    bool WholeBufferRanges() const override { return true; }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeWholeBufferRanges(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new WholeBufferRangesFeature());
}

} // namespace wiz3d
