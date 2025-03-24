## NASL 👃 Not A Shading Language

NASL is not a shading language.
It is a simple, C/C++, vector math library for writing shader-like code, in a conventional language.
It supports automatic conversions to and from Clang/GCC's native vector types, swizzling (C++20 only) and can be used together with [Vcc](https://shady-gang.github.io/vcc) to make real shaders.

## Taster

```cpp
#include <nasl.h>

using namespace nasl;

vec3 blue() { return vec3(0.0f, 0.0, 1.0); }
vec3 yellow() { return blue().zzx; }

vec4 shader(vec2 pos) {
    vec3 color;
    if (pos.y > 0.5)
      color = blue();
    else
      color = yellow();
    return vec4(color, pos);
}
```