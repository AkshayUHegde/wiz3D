/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <RingKeepsShifts Value="1"/>: a ring range's right version is rewritten in place while draws queued before it still
// read those bytes, so a register whose game bytes are unchanged keeps the shift an earlier write gave it.

#include "StdAfx.h"
#include "ProfileFeature.h"

namespace wiz3d
{
namespace
{

class RingKeepsShiftsFeature : public ProfileFeature
{
public:
    bool RingKeepsShifts() const override { return true; }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeRingKeepsShifts(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new RingKeepsShiftsFeature());
}

} // namespace wiz3d
