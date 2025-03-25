#ifndef NASL_VEC
#define NASL_VEC

#if defined(__cplusplus) & !defined(NASL_CPP_NO_NAMESPACE)
namespace nasl {
#endif

#ifdef __clang__
typedef float native_vec4     __attribute__((ext_vector_type(4)));
typedef float native_vec3     __attribute__((ext_vector_type(3)));
typedef float native_vec2     __attribute__((ext_vector_type(2)));

typedef int native_ivec4      __attribute__((ext_vector_type(4)));
typedef int native_ivec3      __attribute__((ext_vector_type(3)));
typedef int native_ivec2      __attribute__((ext_vector_type(2)));

typedef unsigned native_uvec4 __attribute__((ext_vector_type(4)));
typedef unsigned native_uvec3 __attribute__((ext_vector_type(3)));
typedef unsigned native_uvec2 __attribute__((ext_vector_type(2)));
#else
// gcc can't cope with this
typedef float native_vec4     __attribute__((vector_size(16)));
typedef float native_vec3     __attribute__((vector_size(16)));
typedef float native_vec2     __attribute__((vector_size(8)));

typedef int native_ivec4      __attribute__((vector_size(16)));
typedef int native_ivec3      __attribute__((vector_size(16)));
typedef int native_ivec2      __attribute__((vector_size(8)));

typedef unsigned native_uvec4 __attribute__((vector_size(16)));
typedef unsigned native_uvec3 __attribute__((vector_size(16)));
typedef unsigned native_uvec2 __attribute__((vector_size(8)));
#endif

#if defined(__cplusplus) & !defined(NASL_CPP_NO_WRAPPER_CLASSES)
#define NASL_ENABLE_WRAPPER_CLASSES
#endif

#ifdef NASL_ENABLE_WRAPPER_CLASSES
static_assert(__cplusplus >= 202002L, "C++20 is required");
template<typename T, unsigned len>
struct vec_native_type {};

template<> struct vec_native_type   <float, 4> { using Native =  native_vec4; };
template<> struct vec_native_type     <int, 4> { using Native = native_ivec4; };
template<> struct vec_native_type<unsigned, 4> { using Native = native_uvec4; };
template<> struct vec_native_type   <float, 3> { using Native =  native_vec3; };
template<> struct vec_native_type     <int, 3> { using Native = native_ivec3; };
template<> struct vec_native_type<unsigned, 3> { using Native = native_uvec3; };
template<> struct vec_native_type   <float, 2> { using Native =  native_vec2; };
template<> struct vec_native_type     <int, 2> { using Native = native_ivec2; };
template<> struct vec_native_type<unsigned, 2> { using Native = native_uvec2; };
template<> struct vec_native_type   <float, 1> { using Native = float; };
template<> struct vec_native_type     <int, 1> { using Native = int; };
template<> struct vec_native_type<unsigned, 1> { using Native = unsigned; };

template<int len>
struct Mapping {
    int data[len];
};

template<int dst_len>
static consteval bool fits(unsigned len, Mapping<dst_len> mapping) {
    for (unsigned i = 0; i < dst_len; i++) {
        if (mapping.data[i] >= len)
            return false;
    }
    return true;
}

template <auto B, auto E, typename F>
constexpr void for_range(F f)
{
    if constexpr (B < E)
    {
        f.template operator()<B>();
        for_range<B + 1, E, F>((f));
    }
}

template<typename T, unsigned len>
struct vec_impl {
    using This = vec_impl<T, len>;
    using Native = typename vec_native_type<T, len>::Native;

    vec_impl() = default;
    vec_impl(T s) {
        for_range<0, len>([&]<auto i>(){
            arr[i] = s;
        });
    }

    vec_impl(T x, T y, T z, T w) requires (len >= 4) {
        this->arr[0] = x;
        this->arr[1] = y;
        this->arr[2] = z;
        this->arr[3] = w;
    }

    vec_impl(vec_impl<T, 2> xy, T z, T w) requires (len >= 4) : vec_impl(xy.x, xy.y, z, w) {}
    vec_impl(T x, vec_impl<T, 2> yz, T w) requires (len >= 4) : vec_impl(x, yz.x, yz.y, w) {}
    vec_impl(T x, T y, vec_impl<T, 2> zw) requires (len >= 4) : vec_impl(x, y, zw.x, zw.y) {}
    vec_impl(vec_impl<T, 2> xy, vec_impl<T, 2> zw) requires (len >= 4) : vec_impl(xy.x, xy.y, zw.x, zw.y) {}

    vec_impl(vec_impl<T, 3> xyz, T w) requires (len >= 4) : vec_impl(xyz.x, xyz.y, xyz.z, w) {}
    vec_impl(T x, vec_impl<T, 3> yzw) requires (len >= 4) : vec_impl(x, yzw.x, yzw.y, yzw.z) {}

    vec_impl(T x, T y, T z) requires (len >= 3) {
        this->arr[0] = x;
        this->arr[1] = y;
        this->arr[2] = z;
    }

    vec_impl(vec_impl<T, 2> xy, T z) requires (len >= 3) : vec_impl(xy.x, xy.y, z) {}
    vec_impl(T x, vec_impl<T, 2> yz) requires (len >= 3) : vec_impl(x, yz.x, yz.y) {}

    vec_impl(T x, T y) {
        this->arr[0] = x;
        this->arr[1] = y;
    }

    vec_impl(Native n) {
        for_range<0, len>([&]<auto i>(){
            arr[i] = n[i];
        });
    }

    operator Native() const {
        Native n;
        for_range<0, len>([&]<auto i>(){
            n[i] = arr[i];
        });
        return n;
    }

    T& operator [](int i) {
        return arr[i];
    }

    T operator [](int i) const {
        return arr[i];
    }

    template <int dst_len, Mapping<dst_len> mapping> // requires(fits<dst_len>(len, mapping))
    struct Swizzler {
        using That = vec_impl<T, dst_len>;
        using ThatNative = typename vec_native_type<T, dst_len>::Native;

        operator That() const requires(dst_len > 1 && fits<dst_len>(len, mapping)) {
            auto src = reinterpret_cast<const This*>(this);
            That dst;
            for_range<0, dst_len>([&]<auto i>(){
                dst.arr[i] = src->arr[mapping.data[i]];
            });
            return dst;
        }

        operator ThatNative() const requires(dst_len > 1 && fits<dst_len>(len, mapping)) {
            That that = *this;
            return that;
        }

        operator T() const requires(dst_len == 1 && fits<dst_len>(len, mapping)) {
            auto src = reinterpret_cast<const This*>(this);
            return src->arr[mapping.data[0]];
        }

        void operator=(const T& t) requires(dst_len == 1 && fits<dst_len>(len, mapping)) {
            auto src = reinterpret_cast<This*>(this);
            src->arr[mapping.data[0]] = t;
        }

        void operator=(const That& src) requires(dst_len > 1 && fits<dst_len>(len, mapping)) {
            auto dst = reinterpret_cast<This*>(this);
            for_range<0, dst_len>([&]<auto i>(){
                dst->arr[mapping.data[i]] = src.arr[i];
            });
        }
    };

    This operator +(This other) {
        This result;
        for_range<0, len>([&]<auto i>(){
            result.arr[i] = this->arr[i] + other.arr[i];
        });
        return result;
    }
    This operator -(This other) {
        This result;
        for_range<0, len>([&]<auto i>(){
            result.arr[i] = this->arr[i] - other.arr[i];
        });
        return result;
    }
    This operator *(This other) {
        This result;
        for_range<0, len>([&]<auto i>(){
            result.arr[i] = this->arr[i] * other.arr[i];
        });
        return result;
    }
    This operator /(This other) {
        This result;
        for_range<0, len>([&]<auto i>(){
            result.arr[i] = this->arr[i] / other.arr[i];
        });
        return result;
    }
    This operator *(T s) {
        This result;
        for_range<0, len>([&]<auto i>(){
            result.arr[i] = this->arr[i] * s;
        });
        return result;
    }
    This operator /(T s) {
        This result;
        for_range<0, len>([&]<auto i>(){
            result.arr[i] = this->arr[i] / s;
        });
        return result;
    }

#define COMPONENT_0 x
#define COMPONENT_1 y
#define COMPONENT_2 z
#define COMPONENT_3 w

#define CONCAT_4_(a, b, c, d) a##b##c##d
#define CONCAT_4(a, b, c, d) CONCAT_4_(a, b, c, d)
#define SWIZZLER_4(a, b, c, d) Swizzler<4, Mapping<4> { a, b, c, d }> CONCAT_4(COMPONENT_##a, COMPONENT_##b, COMPONENT_##c, COMPONENT_##d);
#define GEN_SWIZZLERS_4_D(D, C, B, A) SWIZZLER_4(A, B, C, D)
#define GEN_SWIZZLERS_4_C(C, B, A) GEN_SWIZZLERS_4_D(0, C, B, A) GEN_SWIZZLERS_4_D(1, C, B, A) GEN_SWIZZLERS_4_D(2, C, B, A) GEN_SWIZZLERS_4_D(3, C, B, A)
#define GEN_SWIZZLERS_4_B(B, A) GEN_SWIZZLERS_4_C(0, B, A) GEN_SWIZZLERS_4_C(1, B, A) GEN_SWIZZLERS_4_C(2, B, A) GEN_SWIZZLERS_4_C(3, B, A)
#define GEN_SWIZZLERS_4_A(A) GEN_SWIZZLERS_4_B(0, A) GEN_SWIZZLERS_4_B(1, A) GEN_SWIZZLERS_4_B(2, A) GEN_SWIZZLERS_4_B(3, A)
#define GEN_SWIZZLERS_4() GEN_SWIZZLERS_4_A(0) GEN_SWIZZLERS_4_A(1) GEN_SWIZZLERS_4_A(2) GEN_SWIZZLERS_4_A(3)

#define CONCAT_3_(a, b, c) a##b##c
#define CONCAT_3(a, b, c) CONCAT_3_(a, b, c)
#define SWIZZLER_3(a, b, c) Swizzler<3, Mapping<3> { a, b, c }> CONCAT_3(COMPONENT_##a, COMPONENT_##b, COMPONENT_##c);
#define GEN_SWIZZLERS_3_C(C, B, A) SWIZZLER_3(A, B, C)
#define GEN_SWIZZLERS_3_B(B, A) GEN_SWIZZLERS_3_C(0, B, A) GEN_SWIZZLERS_3_C(1, B, A) GEN_SWIZZLERS_3_C(2, B, A) GEN_SWIZZLERS_3_C(3, B, A)
#define GEN_SWIZZLERS_3_A(A) GEN_SWIZZLERS_3_B(0, A) GEN_SWIZZLERS_3_B(1, A) GEN_SWIZZLERS_3_B(2, A) GEN_SWIZZLERS_3_B(3, A)
#define GEN_SWIZZLERS_3() GEN_SWIZZLERS_3_A(0) GEN_SWIZZLERS_3_A(1) GEN_SWIZZLERS_3_A(2) GEN_SWIZZLERS_3_A(3)

#define CONCAT_2_(a, b) a##b
#define CONCAT_2(a, b) CONCAT_2_(a, b)
#define SWIZZLER_2(a, b) Swizzler<2, Mapping<2> { a, b }> CONCAT_2(COMPONENT_##a, COMPONENT_##b);
#define GEN_SWIZZLERS_2_B(B, A) SWIZZLER_2(A, B)
#define GEN_SWIZZLERS_2_A(A) GEN_SWIZZLERS_2_B(0, A) GEN_SWIZZLERS_2_B(1, A) GEN_SWIZZLERS_2_B(2, A) GEN_SWIZZLERS_2_B(3, A)
#define GEN_SWIZZLERS_2() GEN_SWIZZLERS_2_A(0) GEN_SWIZZLERS_2_A(1) GEN_SWIZZLERS_2_A(2) GEN_SWIZZLERS_2_A(3)

#define SWIZZLER_1(a) Swizzler<1, Mapping<1> { a }> COMPONENT_##a;
#define GEN_SWIZZLERS_1_A(A) SWIZZLER_1(A)
#define GEN_SWIZZLERS_1() GEN_SWIZZLERS_1_A(0) GEN_SWIZZLERS_1_A(1) GEN_SWIZZLERS_1_A(2) GEN_SWIZZLERS_1_A(3)

    union {
        GEN_SWIZZLERS_1()
        GEN_SWIZZLERS_2()
        GEN_SWIZZLERS_3()
        GEN_SWIZZLERS_4()
        T arr[len];
    };

    static_assert(sizeof(T) * len == sizeof(arr));
};
#endif

#ifdef NASL_ENABLE_WRAPPER_CLASSES
typedef vec_impl<float, 4> vec4;
typedef vec_impl<unsigned, 4> uvec4;
typedef vec_impl<int, 4> ivec4;
typedef vec_impl<float, 3> vec3;
typedef vec_impl<unsigned, 3> uvec3;
typedef vec_impl<int, 3> ivec3;
typedef vec_impl<float, 2> vec2;
typedef vec_impl<unsigned, 2> uvec2;
typedef vec_impl<int, 2> ivec2;
#else
typedef union {
    native_vec4 native;
    float arr[4];
    float x, y, z, w;
} vec4;
typedef union {
    // native vec3 is padded to 4N
    float arr[3];
    float x, y, z;
} vec3;
typedef union {
    native_vec2 native;
    float arr[2];
    float x, y;
} vec2;
typedef union {
    native_ivec4 native;
    int arr[4];
    int x, y, z, w;
} ivec4;
typedef union {
    // native ivec3 is padded to 4N
    int arr[3];
    int x, y, z;
} ivec3;
typedef union {
    native_ivec2 native;
    int arr[2];
    int x, y;
} ivec2;
typedef union {
    native_uvec4 native;
    unsigned arr[4];
    unsigned x, y, z, w;
} uvec4;
typedef union {
    // native uvec3 is padded to 4N
    unsigned arr[3];
    unsigned x, y, z;
} uvec3;
typedef union {
    native_uvec2 native;
    unsigned arr[2];
    unsigned x, y;
} uvec2;
#endif

#define MATH_FN_QUALIFIERS static inline __attribute__ ((const))

#define impl_op_ctor f
#define impl_op_scale ((a.arr[i]) * scale)

#define impl_op_binop_helper(op) impl_op_##op(a.arr[i], b.arr[i])
#define impl_op_add(a, b) ((a) + (b))
#define impl_op_sub(a, b) ((a) - (b))
#define impl_op_mul(a, b) ((a) * (b))
#define impl_op_div(a, b) ((a) / (b))

#define impl_op_unop_helper(op) impl_op_##op(a.arr[i])
#define impl_op_neg(a) (-(a))

#define impl_vec_op(size, snake_name, type_name, op_name, op, ...) \
MATH_FN_QUALIFIERS type_name snake_name##_##op_name(__VA_ARGS__) { \
    type_name thing;                                               \
    for (int i = 0; i < size; i++)                                 \
      thing.arr[i] = impl_op_##op;                                 \
    return thing;                                                  \
}

#define impl_vec_bin_op(size, sn, n, opn, op) impl_vec_op(size, sn, n, opn, binop_helper(op), n a, n b)
#define impl_vec_un_op(size, sn, n, opn, op) impl_vec_op(size, sn, n, opn, unop_helper(op), n a)

#define impl_vec_ops(size, snake_name, name, scalar_name) \
impl_vec_bin_op(size, snake_name, name, add, add) \
impl_vec_bin_op(size, snake_name, name, sub, sub) \
impl_vec_bin_op(size, snake_name, name, mul, mul) \
impl_vec_bin_op(size, snake_name, name, div, div) \
impl_vec_un_op(size, snake_name, name, neg, neg) \
impl_vec_op(size, snake_name, name, scale, scale, name a, scalar_name scale) \
impl_vec_op(size, snake_name, name, ctor, ctor, scalar_name f)

impl_vec_ops(2, vec2, vec2, float)
impl_vec_ops(3, vec3, vec3, float)
impl_vec_ops(4, vec4, vec4, float)
impl_vec_ops(2, ivec2, ivec2, int)
impl_vec_ops(3, ivec3, ivec3, int)
impl_vec_ops(4, ivec4, ivec4, int)
impl_vec_ops(2, uvec2, uvec2, unsigned)
impl_vec_ops(3, uvec3, uvec3, unsigned)
impl_vec_ops(4, uvec4, uvec4, unsigned)

#ifdef NASL_ENABLE_WRAPPER_CLASSES
template<unsigned len>
float lengthSquared(vec_impl<float, len> vec) {
    float acc = 0.0f;
    for_range<0, len>([&]<auto i>(){
        acc += vec.arr[i] * vec.arr[i];
    });
    return acc;
}

template<unsigned len>
float length(vec_impl<float, len> vec) {
    return sqrtf(lengthSquared(vec));
}

template<unsigned len>
vec_impl<float, len> normalize(vec_impl<float, len> vec) {
    return vec / length(vec);
}
#endif

#if defined(__cplusplus) & !defined(NASL_CPP_NO_NAMESPACE)
}
#endif

#endif
