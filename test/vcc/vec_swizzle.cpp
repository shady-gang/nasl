#include <shady.h>
#include <nasl.h>

using namespace vcc;

extern "C" {

location(0) native_vec3 vertexColor;
location(0) native_vec4 outColor;

fragment_shader void test() {
    nasl::vec4 a;
    a.xyz = vertexColor;
    a.w = 1.0f;
    outColor = a;
}

}