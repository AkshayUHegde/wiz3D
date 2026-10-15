/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// FogVolumeLookups, a ShaderPatch11 client: a pixel shader's lookups of a fog grid built from the left camera (slice =
// log2(depth * zParams.x + zParams.y) * zParams.z * invGridSize.z), moved to the left eye's screen position.

#pragma once

#include "ShaderPatch11.h"
#include <string>
#include <vector>

namespace wiz3d
{

// One lookup: the nth sample_l, its coordinate register and grid x / slice components, the cbuffer registers it reads,
// and for x from pixels the register and component scaling them (svposToUv -1 = x from NDC * 0.5 + 0.5).
struct FogLookup { int nth, reg, tslot, cu, cw, view, invGrid, zParams, svposToUv, svposComp; };
static const int kFogLookupInts = 10;

// The lookups in a pixel shader's code, or in its disassembly; false = none.
bool FindFogLookups(const void* bc, size_t len, std::vector<FogLookup>& out);
bool FindFogLookupsInText(const std::string& disassembly, std::vector<FogLookup>& out);

// Before each lookup: a copy of its coordinate with x moved by (C / w - 1) * cb.x (NDC) or * cb.z * svposToUv (pixels),
// w undone from the slice; the sample reads the copy. Constants: { b / 2, C, b * viewport width / 2, 0 }.
ShaderPatchPlan FogLookupPlan(const std::vector<FogLookup>& lookups);

} // namespace wiz3d
