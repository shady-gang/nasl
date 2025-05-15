#include "nasl/nasl.h"

#include <assert.h>

static_assert(sizeof(vec4) == sizeof(float) * 4, "");
static_assert(sizeof(vec3) == sizeof(float) * 3, "");
static_assert(sizeof(vec2) == sizeof(float) * 2, "");
static_assert(sizeof(ivec4) == sizeof(int) * 4, "");
static_assert(sizeof(ivec3) == sizeof(int) * 3, "");
static_assert(sizeof(ivec2) == sizeof(int) * 2, "");
static_assert(sizeof(uvec4) == sizeof(unsigned) * 4, "");
static_assert(sizeof(uvec3) == sizeof(unsigned) * 3, "");
static_assert(sizeof(uvec2) == sizeof(unsigned) * 2, "");

int main(int argc, char** argv) {
    return 0;
}