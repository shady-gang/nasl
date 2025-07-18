#include "nasl/nasl.h"

#include <memory>
#include <cassert>

using namespace nasl;

static_assert(sizeof(vec4) == sizeof(float) * 4, "");
static_assert(sizeof(vec3) == sizeof(float) * 3, "");
static_assert(sizeof(vec2) == sizeof(float) * 2, "");
static_assert(sizeof(ivec4) == sizeof(int) * 4, "");
static_assert(sizeof(ivec3) == sizeof(int) * 3, "");
static_assert(sizeof(ivec2) == sizeof(int) * 2, "");
static_assert(sizeof(uvec4) == sizeof(unsigned) * 4, "");
static_assert(sizeof(uvec3) == sizeof(unsigned) * 3, "");
static_assert(sizeof(uvec2) == sizeof(unsigned) * 2, "");

void check_native_casts(const vec4& v4, const uvec4& u4, const ivec4& i4) {
#ifndef NASL_NO_NATIVE_VEC
    native_vec4 nv4 = v4;
    native_uvec4 nu4 = u4;
    native_ivec4 ni4 = i4;
    vec4 rv4 = nv4;
    native_vec3 nv3;
    nv3 = v4.xyz;
#endif
}

void check_vector_scalar_ctors() {
    vec4 x4 = vec4(0.5f);
    vec4 y4 = { 0.5f };
    vec4 z4(0.5f);
    vec4 w4 = 0.5f;

    vec3 x3 = vec3(0.5f);
    vec3 y3 = { 0.5f };
    vec3 z3(0.5f);
    vec3 w3 = 0.5f;

    vec2 x2 = vec2(0.5f);
    vec2 y2 = { 0.5f };
    vec2 z2(0.5f);
    vec2 w2 = 0.5f;
}

void check_vector_unop(vec3 a, float f) {
    vec3 v1 = -a;
}

void check_vector_scalar_binop(vec3 a, float f) {
    vec3 v1 = a * f;
    vec3 v2 = f * a;
    vec3 v3 = a / f;
}

void check_vector_elementwise_binop(vec3 a, vec3 b) {
    vec3 v1 = a * b;
    vec3 v2 = b * a;
    vec3 v3 = a / b;
    vec3 v4 = b / a;
}

void check_swizzle_const(const vec4& v4, const uvec4& u4, const ivec4& i4) {
    float r1 = v4.x;
    vec2 r2 = v4.xy;
    v4.xyz;
    v4.xyzw;

    v4.xxxx;
    v4.xyww;
}

void check_ctor_weird() {
    vec4(vec2(0.5f), vec2(0.5f));
    vec4(0.5f, vec2(0.5f), 0.5f);
    vec4(0.5f, vec3(0.5f));
    vec4(vec3(0.5f), 0.5f);
}

void check_swizzle_mut(vec4& v) {
    v.x = 0.5f;
    v.xy = vec2(0.5f, 0.9f);
}

#include <cassert>
#include <cstdio>
int main(int argc, char** argv) {
    vec4 v(1.0f, 0.5f, 0.0f, -1.0f);
    float f;
    f = v.x; printf("f = %f;\n", f); assert(f == 1.0f);
    f = v.y; printf("f = %f;\n", f); assert(f == 0.5f);
    f = v.z; printf("f = %f;\n", f); assert(f == 0.0f);
    f = v.w; printf("f = %f;\n", f); assert(f == -1.0f);
    std::unique_ptr<vec4> uptr;
}