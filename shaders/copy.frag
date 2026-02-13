#version 410 core

uniform sampler2D u_tex_source;

in vec2 v_tex_coord;

layout(location = 0) out vec4 o_dest;

void main(void) {
	o_dest = texture(u_tex_source, v_tex_coord);
}
