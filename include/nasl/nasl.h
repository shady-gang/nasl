#ifndef NASL_H
#define NASL_H

#ifdef NASL_FUNCTION
#elifdef __CUDACC__
#define NASL_FUNCTION __device__ static inline
#define NASL_METHOD __device__ inline constexpr
#define NASL_CONSTANT __device__ __constant__ constexpr
#elif __cplusplus
#define NASL_FUNCTION static inline
#define NASL_METHOD constexpr inline
#define NASL_CONSTANT constexpr
#else
#define NASL_FUNCTION static inline
#define NASL_METHOD inline
#define NASL_CONSTANT const static
#endif

#include "nasl_vec.h"
//#include "nasl_mat.h"

#endif
