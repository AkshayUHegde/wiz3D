#version 450
// The OSD quad's colour: the slot's text image, blended over the frame by its alpha.
layout(set = 0, binding = 0) uniform sampler2D slot;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main()
{
    color = texture(slot, uv);
}
