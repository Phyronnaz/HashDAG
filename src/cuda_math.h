#pragma once

#include "typedefs.h"
#include <cmath>

#ifndef __CUDACC__
template<typename T>
inline T max(T a, T b) { return (((a) > (b)) ? (a) : (b)); }
template<typename T>
inline T min(T a, T b) { return (((a) < (b)) ? (a) : (b)); }
#endif
template<typename T>
HOST_DEVICE T constexpr constexpr_max(T a, T b) { return (((a) > (b)) ? (a) : (b)); }
template<typename T>
HOST_DEVICE T constexpr constexpr_min(T a, T b) { return (((a) < (b)) ? (a) : (b)); }

template<typename T> constexpr HOST_DEVICE T make_vector3(const decltype(T::x) x, const decltype(T::x) y, const decltype(T::x) z)
{
  T t{}; t.x = x; t.y = y; t.z = z; return t;
}

template<typename T, typename U> HOST_DEVICE constexpr auto lerp(T a, T b, U f) { return a * (1 - f) + b * f; }
template<typename T> HOST_DEVICE constexpr T squared(T f) { return f * f; }
template<typename T> HOST_DEVICE constexpr T clamp(T f, T a, T b) { return f < a ? a : f > b ? b : f; }

#define BIN_VECTOR_OP(T, OP) \
HOST_DEVICE constexpr T operator OP (const decltype(T::x) a, const T& b) { return make_vector3<T>(a OP b.x, a OP b.y, a OP b.z); } \
HOST_DEVICE constexpr T operator OP (const T& a, const decltype(T::x) b) { return make_vector3<T>(a.x OP b, a.y OP b, a.z OP b); } \
HOST_DEVICE constexpr T operator OP (const T& a, const T& b) { return make_vector3<T>(a.x OP b.x, a.y OP b.y, a.z OP b.z); }

#define VECTOR_OP(T) \
BIN_VECTOR_OP(T, +) \
BIN_VECTOR_OP(T, -) \
BIN_VECTOR_OP(T, *) \
BIN_VECTOR_OP(T, /) \
HOST_DEVICE constexpr bool operator < (const T& a, const T& b) { return a.x < b.x && a.y < b.y && a.z < b.z; } \
HOST_DEVICE constexpr bool operator > (const T& a, const T& b) { return a.x > b.x && a.y > b.y && a.z > b.z; } \
HOST_DEVICE constexpr bool operator <= (const T& a, const T& b) { return a.x <= b.x && a.y <= b.y && a.z <= b.z; } \
HOST_DEVICE constexpr bool operator >= (const T& a, const T& b) { return a.x >= b.x && a.y >= b.y && a.z >= b.z; } \
HOST_DEVICE constexpr decltype(T::x) dot(const T& a, const T& b) { return a.x * b.x + a.y * b.y + a.z * b.z; } \
HOST_DEVICE constexpr decltype(T::x) length_squared(const T& a) { return dot(a, a); } \
HOST_DEVICE constexpr decltype(T::x) max(const T& v) { return constexpr_max(v.x, constexpr_max(v.y, v.z)); } \
HOST_DEVICE constexpr decltype(T::x) min(const T& v) { return constexpr_min(v.x, constexpr_min(v.y, v.z)); } \
HOST_DEVICE constexpr T max(const T& a, const T& b) { return make_vector3<T>(constexpr_max(a.x, b.x), constexpr_max(a.y, b.y), constexpr_max(a.z, b.z)); } \
HOST_DEVICE constexpr T min(const T& a, const T& b) { return make_vector3<T>(constexpr_min(a.x, b.x), constexpr_min(a.y, b.y), constexpr_min(a.z, b.z)); } \
HOST_DEVICE constexpr T clamp_vector(const T& x, const T& a, const T& b) { return make_vector3<T>(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z)); } \
HOST_DEVICE constexpr T clamp_vector(const T& x, decltype(T::x) a, decltype(T::x) b) { return make_vector3<T>(clamp(x.x, a, b), clamp(x.y, a, b), clamp(x.z, a, b)); }

#define VECTOR_FLOAT_OP(T) \
HOST_DEVICE constexpr T operator - (T a) { return make_vector3<T>(-a.x, -a.y, -a.z); } \
HOST_DEVICE auto length(T a) { return (decltype(T::x))sqrt(dot(a, a)); } \
HOST_DEVICE T normalize(T a) { return (1 / length(a)) * a; } \
HOST_DEVICE constexpr T abs(const T& v) { return make_vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z)); }; \
HOST_DEVICE T ceil(const T& v) { return make_vector3<T>(std::ceil(v.x), std::ceil(v.y), std::ceil(v.z)); }; \
HOST_DEVICE T round(const T& v) { return make_vector3<T>(std::round(v.x), std::round(v.y), std::round(v.z)); }; \
HOST_DEVICE uint3 round_to_uint(const T& v) { return make_uint3(uint32(std::round(v.x)), uint32(std::round(v.y)), uint32(std::round(v.z))); };

VECTOR_OP(float3)
VECTOR_OP(double3)
VECTOR_OP(uint3)

VECTOR_FLOAT_OP(float3)
VECTOR_FLOAT_OP(double3)

HOST_DEVICE constexpr float3 make_float3(float a) { return make_vector3<float3>(a, a, a); }
HOST_DEVICE constexpr float3 make_float3(double3 d) { return make_vector3<float3>(float(d.x), float(d.y), float(d.z)); }
HOST_DEVICE constexpr float3 make_float3(const uint3 &a) { return make_vector3<float3>(float(a.x), float(a.y), float(a.z)); };

HOST_DEVICE constexpr double3 make_double3(double a) { return make_vector3<double3>(a, a, a); }
HOST_DEVICE constexpr double3 make_double3(const uint3 &a) { return make_vector3<double3>(double(a.x), double(a.y), double(a.z)); }
HOST_DEVICE constexpr double3 make_double3(const float3& f) { return make_vector3<double3>(double(f.x), double(f.y), double(f.z)); }

HOST double length(uint3 a) { return std::sqrt(dot(a, a)); }
HOST_DEVICE constexpr uint3 operator << (const uint3 &v, const uint32 shift) { return make_vector3<uint3>(v.x << shift, v.y << shift, v.z << shift); }
HOST_DEVICE constexpr bool operator == (const uint3 & a, const uint3 & b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
HOST_DEVICE constexpr bool operator != (const uint3 & a, const uint3 & b) { return !(a == b); }
HOST_DEVICE constexpr uint3 make_uint3(const uint32& v) { return make_vector3<uint3>(v, v, v); }
HOST_DEVICE constexpr uint3 make_uint3(const uint4& v) { return make_vector3<uint3>(v.x, v.y, v.z); }

HOST_DEVICE constexpr uint3 truncate(const float3& f) { return make_vector3<uint3>(uint32(f.x), uint32(f.y), uint32(f.z)); }

namespace std 
{
	template <>
	struct hash<uint3>
	{
		inline constexpr std::size_t operator()(const uint3& k) const noexcept
		{
			return size_t(k.x) + 81799 * size_t(k.y) + 38351 * size_t(k.z);
		}
	};
}