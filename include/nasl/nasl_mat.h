#ifndef NASL_MAT
#define NASL_MAT

#if defined(__cplusplus) & !defined(NASL_CPP_NO_NAMESPACE)
namespace nasl {
#endif

typedef union mat4_ mat4;

NASL_FUNCTION mat4 transpose_mat4(mat4 src);
NASL_FUNCTION mat4 mul_mat4(mat4 l, mat4 r);
NASL_FUNCTION vec4 mul_mat4_vec4f(mat4 l, vec4 r);

union mat4_ {
    struct {
        // we use row-major ordering
        float m00, m01, m02, m03,
            m10, m11, m12, m13,
            m20, m21, m22, m23,
            m30, m31, m32, m33;
    } elems;
    vec4 rows[4];
    float arr[16];

#if defined(__cplusplus)
    NASL_METHOD mat4 operator*(const mat4& other) {
        return mul_mat4(*this, other);
    }

    NASL_METHOD vec4 operator*(const vec4& other) {
        return mul_mat4_vec4f(*this, other);
    }
#endif
};

NASL_CONSTANT mat4 identity_mat4 = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
};

NASL_FUNCTION mat4 transpose_mat4(mat4 src) {
    mat4 m = {
        {
            src.elems.m00, src.elems.m10, src.elems.m20, src.elems.m30,
            src.elems.m01, src.elems.m11, src.elems.m21, src.elems.m31,
            src.elems.m02, src.elems.m12, src.elems.m22, src.elems.m32,
            src.elems.m03, src.elems.m13, src.elems.m23, src.elems.m33,
        }
    };
    return m;
}

NASL_FUNCTION mat4 invert_mat4(mat4 m) {
    float a = m.elems.m00 * m.elems.m11 - m.elems.m01 * m.elems.m10;
    float b = m.elems.m00 * m.elems.m12 - m.elems.m02 * m.elems.m10;
    float c = m.elems.m00 * m.elems.m13 - m.elems.m03 * m.elems.m10;
    float d = m.elems.m01 * m.elems.m12 - m.elems.m02 * m.elems.m11;
    float e = m.elems.m01 * m.elems.m13 - m.elems.m03 * m.elems.m11;
    float f = m.elems.m02 * m.elems.m13 - m.elems.m03 * m.elems.m12;
    float g = m.elems.m20 * m.elems.m31 - m.elems.m21 * m.elems.m30;
    float h = m.elems.m20 * m.elems.m32 - m.elems.m22 * m.elems.m30;
    float i = m.elems.m20 * m.elems.m33 - m.elems.m23 * m.elems.m30;
    float j = m.elems.m21 * m.elems.m32 - m.elems.m22 * m.elems.m31;
    float k = m.elems.m21 * m.elems.m33 - m.elems.m23 * m.elems.m31;
    float l = m.elems.m22 * m.elems.m33 - m.elems.m23 * m.elems.m32;
    float det = a * l - b * k + c * j + d * i - e * h + f * g;
    det = 1.0f / det;
    mat4 r;
    r.elems.m00 = ( m.elems.m11 * l - m.elems.m12 * k + m.elems.m13 * j) * det;
    r.elems.m01 = (-m.elems.m01 * l + m.elems.m02 * k - m.elems.m03 * j) * det;
    r.elems.m02 = ( m.elems.m31 * f - m.elems.m32 * e + m.elems.m33 * d) * det;
    r.elems.m03 = (-m.elems.m21 * f + m.elems.m22 * e - m.elems.m23 * d) * det;
    r.elems.m10 = (-m.elems.m10 * l + m.elems.m12 * i - m.elems.m13 * h) * det;
    r.elems.m11 = ( m.elems.m00 * l - m.elems.m02 * i + m.elems.m03 * h) * det;
    r.elems.m12 = (-m.elems.m30 * f + m.elems.m32 * c - m.elems.m33 * b) * det;
    r.elems.m13 = ( m.elems.m20 * f - m.elems.m22 * c + m.elems.m23 * b) * det;
    r.elems.m20 = ( m.elems.m10 * k - m.elems.m11 * i + m.elems.m13 * g) * det;
    r.elems.m21 = (-m.elems.m00 * k + m.elems.m01 * i - m.elems.m03 * g) * det;
    r.elems.m22 = ( m.elems.m30 * e - m.elems.m31 * c + m.elems.m33 * a) * det;
    r.elems.m23 = (-m.elems.m20 * e + m.elems.m21 * c - m.elems.m23 * a) * det;
    r.elems.m30 = (-m.elems.m10 * j + m.elems.m11 * h - m.elems.m12 * g) * det;
    r.elems.m31 = ( m.elems.m00 * j - m.elems.m01 * h + m.elems.m02 * g) * det;
    r.elems.m32 = (-m.elems.m30 * d + m.elems.m31 * b - m.elems.m32 * a) * det;
    r.elems.m33 = ( m.elems.m20 * d - m.elems.m21 * b + m.elems.m22 * a) * det;
    return r;
}

NASL_FUNCTION mat4 perspective_mat4(float a, float fov, float n, float f) {
    float pi = M_PI;
    float s = 1.0f / tanf(fov * 0.5f * (pi / 180.0f));
    mat4 m = {
        {
            s / a, 0, 0, 0,
            0, s, 0, 0,
            0, 0, -f / (f - n), -1.f,
            0, 0, -(f * n) / (f - n), 0
        }
    };
    return m;
}

NASL_FUNCTION mat4 translate_mat4(vec3 offset) {
    mat4 m = identity_mat4;
    m.elems.m30 = offset.x;
    m.elems.m31 = offset.y;
    m.elems.m32 = offset.z;
    return m;
}

NASL_FUNCTION mat4 rotate_axis_mat4(unsigned int axis, float f) {
    mat4 m = { 0 };
    m.elems.m33 = 1;

    unsigned int t = (axis + 2) % 3;
    unsigned int s = (axis + 1) % 3;

    m.rows[t].arr[t] =  cosf(f);
    m.rows[t].arr[s] = -sinf(f);
    m.rows[s].arr[t] =  sinf(f);
    m.rows[s].arr[s] =  cosf(f);

    // leave that unchanged
    m.rows[axis].arr[axis] = 1;

    return m;
}

NASL_FUNCTION mat4 mul_mat4(mat4 l, mat4 r) {
    mat4 dst = { 0 };
#define a(i, j) elems.m##i##j
#define t(bc, br, i) l.a(i, br) * r.a(bc, i)
#define e(bc, br) dst.a(bc, br) = t(bc, br, 0) + t(bc, br, 1) + t(bc, br, 2) + t(bc, br, 3);
#define row(c) e(c, 0) e(c, 1) e(c, 2) e(c, 3)
#define genmul() row(0) row(1) row(2) row(3)
    genmul()
    return dst;
#undef a
#undef t
#undef e
#undef row
#undef genmul
}

NASL_FUNCTION vec4 mul_mat4_vec4f(mat4 l, vec4 r) {
    float src[4] = { r.x, r.y, r.z, r.w };
    float dst[4];
#define a(i, j) elems.m##i##j
#define t(bc, br, i) l.a(i, br) * src[i]
#define e(bc, br) dst[br] = t(bc, br, 0) + t(bc, br, 1) + t(bc, br, 2) + t(bc, br, 3);
#define row(c) e(c, 0) e(c, 1) e(c, 2) e(c, 3)
#define genmul() row(0)
    genmul()
#undef a
#undef t
#undef e
#undef row
#undef genmul
    vec4 v = { dst[0], dst[1], dst[2], dst[3] };
    return v;
}

typedef union {
    struct {
        // we use row-major ordering
        float m00, m01, m02,
            m10, m11, m12,
            m20, m21, m22;
    } elems;
    //vec4 rows[4];
    float arr[9];
} mat3;

NASL_CONSTANT mat3 identity_mat3 = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
};

NASL_FUNCTION mat3 transpose_mat3(mat3 src) {
    mat3 m = {
        src.elems.m00, src.elems.m10, src.elems.m20,
        src.elems.m01, src.elems.m11, src.elems.m21,
        src.elems.m02, src.elems.m12, src.elems.m22,
    };
    return m;
}

NASL_FUNCTION mat3 mul_mat3(mat3 l, mat3 r) {
    mat3 dst = { 0 };
#define a(i, j) elems.m##i##j
#define t(bc, br, i) l.a(i, br) * r.a(bc, i)
#define e(bc, br) dst.a(bc, br) = t(bc, br, 0) + t(bc, br, 1) + t(bc, br, 2);
#define row(c) e(c, 0) e(c, 1) e(c, 2)
#define genmul() row(0) row(1) row(2)
    genmul()
    return dst;
#undef a
#undef t
#undef e
#undef row
#undef genmul
}

#if defined(__cplusplus) & !defined(NASL_CPP_NO_NAMESPACE)
}
#endif

#endif
