#version 450
// The OSD quad: a triangle strip over the rectangle in the push constants (x0, y0, x1, y1 in NDC), uv 0..1 across it.
layout(push_constant) uniform Rect { vec4 r; } pc;
layout(location = 0) out vec2 uv;

void main()
{
    uv = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1));
    gl_Position = vec4(mix(pc.r.xy, pc.r.zw, uv), 0.0, 1.0);
}
