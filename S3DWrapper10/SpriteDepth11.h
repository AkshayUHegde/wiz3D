/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// ClipSpaceSprites 3, a ShaderPatch11 client: a sprite vertex shader's twin moves its NDC position by the eye offset at
// the sprite's own depth, measured from the world position the shader reads (no near plane, no depth buffer).

#pragma once

#include "ShaderPatch11.h"

namespace wiz3d
{

// Before each ret: x += cb[1].x + cb[1].y / max(dot(float4(v[worldInput].xyz, 1), cb[0]), 0.001), i.e. b + m41 / w with
// cb[0] the camera's w row over world space. Constants: { w row (4), b, m41, 0, 0 }.
ShaderPatchPlan SpriteDepthPlan(unsigned worldInput);

} // namespace wiz3d
