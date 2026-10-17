/* 
* Project : iZ3D Stereo Driver
* Copyright (C) iZ3D Inc. 2002 - 2010
*/

#pragma once

static char g_szRAWLeftShaderText[] = "					\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
void main() {											\n\
	vec4 cL = texture2D(sL, gl_TexCoord[0].xy);			\n\
	vec3 gamma = vec3(2.2);								\n\
	cL.rgb = pow(cL.rgb, gamma);						\n\
														\n\
	vec4 cRes = cL;										\n\
	cRes.rgb = pow(cRes.rgb, 1/gamma);					\n\
	gl_FragData[0] = cRes;								\n\
}";

static char g_szRAWRightShaderText[] = "				\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
void main() {											\n\
	vec4 cR = texture2D(sR, gl_TexCoord[0].xy);			\n\
	vec3 gamma = vec3(2.2);								\n\
	cR.rgb = pow(cR.rgb, gamma);						\n\
														\n\
	vec4 cRes = cR;										\n\
	cRes.rgb = pow(cRes.rgb, 1/gamma);					\n\
	gl_FragData[0] = cRes;								\n\
}";

static char g_szBackScreenShaderText[] = "				\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
vec4 primaryColor( in vec4 cL, in vec4 cR ) {			\n\
	return (cL + cR) * 0.5f;							\n\
}														\n\
														\n\
void main() {											\n\
	vec4 cL = texture2D(sL, gl_TexCoord[0].xy);			\n\
	vec4 cR = texture2D(sR, gl_TexCoord[0].xy);			\n\
	vec3 gamma = vec3(2.2);								\n\
	cL.rgb = pow(cL.rgb, gamma);						\n\
	cR.rgb = pow(cR.rgb, gamma);						\n\
														\n\
	vec4 cRes = primaryColor(cL, cR);					\n\
	cRes.rgb = pow(cRes.rgb, 1/gamma);					\n\
	gl_FragData[0] = cRes;								\n\
}";

static char g_szFrontScreenShaderText[] = "				\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
vec4 secondaryColor( in vec4 cL, in vec4 cR ) {			\n\
	vec4 Sum = (cL + cR);								\n\
	vec3 rVal = vec3( (cR / Sum));						\n\
	vec4 res;											\n\
	res.r = (Sum.r >= 0.0000001f ? rVal.r : 0.5f);		\n\
	res.g = (Sum.g >= 0.0000001f ? rVal.g : 0.5f);		\n\
	res.b = (Sum.b >= 0.0000001f ? rVal.b : 0.5f);		\n\
	res.a = cR.a;										\n\
	return res;											\n\
}														\n\
														\n\
void main() {											\n\
	vec4 cL = texture2D(sL, gl_TexCoord[0].xy);			\n\
	vec4 cR = texture2D(sR, gl_TexCoord[0].xy);			\n\
	vec3 gamma = vec3(2.2);								\n\
	cL.rgb = pow(cL.rgb, gamma);						\n\
	cR.rgb = pow(cR.rgb, gamma);						\n\
														\n\
	vec4 cRes = secondaryColor( cL, cR);				\n\
	cRes.rgb = pow(cRes.rgb, 1/gamma);					\n\
	gl_FragData[0] = cRes;								\n\
}";

static char g_szFrontScreenCFLShaderText[] = "			\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
vec4 secondaryCFLColor( in vec4 cL, in vec4 cR ) {		\n\
	float rcSum = dot( (cL + cR).xyz , vec3( 1.0f ));	\n\
	float rcR = dot( cR.xyz , vec3( 1.0f ));			\n\
	float rVal = (rcR / rcSum);							\n\
	return vec4( rcSum >= 0.003f ? rVal : 0.5f );		\n\
}														\n\
														\n\
void main() {											\n\
	vec4 cL = texture2D(sL, gl_TexCoord[0].xy);			\n\
	vec4 cR = texture2D(sR, gl_TexCoord[0].xy);			\n\
	vec3 gamma = vec3(2.2);								\n\
	cL.rgb = pow(cL.rgb, gamma);						\n\
	cR.rgb = pow(cR.rgb, gamma);						\n\
														\n\
	vec4 cRes = secondaryCFLColor( cL, cR);				\n\
	cRes.rgb = pow(cRes.rgb, 1/gamma);					\n\
	gl_FragData[0] = cRes;								\n\
}";

static char g_szVertexShaderText[] = "					\n\
void main() {											\n\
	gl_Position = gl_Vertex;							\n\
	gl_TexCoord[0] = gl_MultiTexCoord0;					\n\
}";

// Anaglyph Red/Cyan using the Dubois colour-optimized matrix. Operates in
// linear-light space (gamma-correct), then re-encodes to sRGB. Single anaglyph
// option shipped — earlier grayscale and naive full-colour variants produced
// worse retinal rivalry / ghosting, so they were retired.
static char g_szAnaglyphShaderText[] = "				\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
void main() {											\n\
	vec4 cL = texture2D(sL, gl_TexCoord[0].xy);			\n\
	vec4 cR = texture2D(sR, gl_TexCoord[0].xy);			\n\
	vec3 L = pow(cL.rgb, vec3(2.2));					\n\
	vec3 R = pow(cR.rgb, vec3(2.2));					\n\
	float r = dot(L, vec3( 0.4561, 0.5008, 0.0526))	\n\
			+ dot(R, vec3(-0.0434,-0.0083, 0.0003));	\n\
	float g = dot(L, vec3(-0.0565,-0.0034, 0.0017))	\n\
			+ dot(R, vec3( 0.3784, 0.7334,-0.0162));	\n\
	float b = dot(L, vec3(-0.0103,-0.0087, 0.0009))	\n\
			+ dot(R, vec3(-0.0258,-0.0450, 0.9741));	\n\
	vec3 c = pow(max(vec3(r, g, b), vec3(0.0)), vec3(0.4545)); \n\
	gl_FragData[0] = vec4(c, 1.0);						\n\
}";

// Row Interleaved. Output even rows from left eye, odd rows from right eye.
// Used by passive-3D monitors that polarise alternate scanlines. Uses
// gl_FragCoord.y so the parity is screen-space, independent of the source
// texture sampling.
static char g_szRowInterleavedShaderText[] = "			\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
void main() {											\n\
	vec4 cL = texture2D(sL, gl_TexCoord[0].xy);			\n\
	vec4 cR = texture2D(sR, gl_TexCoord[0].xy);			\n\
	bool isOddRow = mod(floor(gl_FragCoord.y), 2.0) >= 0.5;	\n\
	gl_FragData[0] = isOddRow ? cR : cL;				\n\
}";

// Column Interleaved. Like line-interleaved but on vertical lines instead
// of horizontal. Used by rarer column-polarised displays.
static char g_szColumnInterleavedShaderText[] = "		\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
void main() {											\n\
	vec4 cL = texture2D(sL, gl_TexCoord[0].xy);			\n\
	vec4 cR = texture2D(sR, gl_TexCoord[0].xy);			\n\
	bool isOddCol = mod(floor(gl_FragCoord.x), 2.0) >= 0.5;	\n\
	gl_FragData[0] = isOddCol ? cR : cL;				\n\
}";

// Checkerboard. XOR of x and y pixel parity. Used by DLP-Link 3D projectors
// that expect a checkerboard-packed stereo signal.
static char g_szCheckerboardShaderText[] = "			\n\
uniform sampler2D sL;									\n\
uniform sampler2D sR;									\n\
														\n\
void main() {											\n\
	vec4 cL = texture2D(sL, gl_TexCoord[0].xy);			\n\
	vec4 cR = texture2D(sR, gl_TexCoord[0].xy);			\n\
	float xParity = mod(floor(gl_FragCoord.x), 2.0);	\n\
	float yParity = mod(floor(gl_FragCoord.y), 2.0);	\n\
	bool useRight = mod(xParity + yParity, 2.0) >= 0.5;	\n\
	gl_FragData[0] = useRight ? cR : cL;				\n\
}";
