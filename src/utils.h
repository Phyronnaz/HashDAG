#pragma once

#include "typedefs.h"

#include <type_traits>
#include <chrono>

namespace Utils
{
	HOST_DEVICE uint32 popc(uint32 a)
	{
#if defined(__CUDA_ARCH__)
		return __popc(a);
#else
#if USE_POPC_INTRINSICS
		return __builtin_popcount(a);
#else // !USE_POPC_INTRINSICS
		// Source: http://graphics.stanford.edu/~seander/bithacks.html
		a = a - ((a >> 1) & 0x55555555);
		a = (a & 0x33333333) + ((a >> 2) & 0x33333333);
		return ((a + (a >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif // ~ USE_POPC_INTRINSICS
#endif
	}
	HOST_DEVICE uint32 popcll(uint64 a)
	{
#if defined(__CUDA_ARCH__)
		return __popcll(a);
#else
#if USE_POPC_INTRINSICS
		return uint32(__builtin_popcountl(a));
#else // !USE_POPC_INTRINSICS
		return popc(uint32(a >> 32)) + popc(uint32(a & 0xFFFFFFFF));
#endif // ~ USE_POPC_INTRINSICS
#endif
	}
	template<uint32 bit = 31>
	HOST_DEVICE bool has_flag(uint32 index)
	{
		return index & (1u << bit);
	}
	template<uint32 bit = 31>
	HOST_DEVICE uint32 set_flag(uint32 index)
	{
		return index | (1u << bit);
	}
	template<uint32 bit = 31>
	HOST_DEVICE uint32 clear_flag(uint32 index)
	{
		return index & ~(1u << bit);
	}
	HOST_DEVICE uint32 level_max_size(uint32 level)
	{
		return 1u << (3 * (12 - level));
	}
	HOST_DEVICE uint8 child_mask(uint32 node)
	{
		return node & 0xff;
	}
	HOST_DEVICE uint32 total_size(uint32 node)
	{
		return Utils::popc(Utils::child_mask(node)) + 1;;
	}
	HOST_DEVICE uint32 child_offset(uint8 childMask, uint8 child)
	{
		return popc(childMask & ((1u << child) - 1u)) + 1;
	}
	HOST_DEVICE uint32 murmurhash32(uint32 h)
	{
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}
	HOST_DEVICE uint64 murmurhash64(uint64 h)
	{
		h ^= h >> 33;
		h *= 0xff51afd7ed558ccd;
		h ^= h >> 33;
		h *= 0xc4ceb9fe1a85ec53;
		h ^= h >> 33;
		return h;
	}

#	if USE_ALTERNATE_HASH
	HOST_DEVICE uint32 murmurhash32xN(uint32 const* ph, std::size_t n, uint32 seed = 0)
	{
		uint32 h = seed;
		for( std::size_t i = 0; i < n; ++i )
		{
			uint32 k = ph[i];
			k *= 0xcc9e2d51;
			k = (k << 15) | (k >> 17);
			k *= 0x1b873593;
			h ^= k;
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}

		h ^= uint32(n);
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}
#	endif // ~ USE_ALTERNATE_HASH

#	if USE_BLOOM_FILTER
	// See https://gist.github.com/badboy/6267743
	HOST_DEVICE
	uint64 hash64shift( uint64 key )
	{
	  key = (~key) + (key << 21);
	  key = key ^ (key >> 24);
	  //key = (key + (key << 3)) + (key << 8); // key * 265
	  key *= 265;
	  key = key ^ (key >> 14);
	  //key = (key + (key << 2)) + (key << 4); // key * 21
	  key *= 21;
	  key = key ^ (key >> 28);
	  key = key + (key << 31);
	  return key;
	}

	// See https://github.com/ZilongTan/fast-hash/blob/master/fasthash.c
	HOST_DEVICE 
	uint64 fasthash64_mix( uint64 aValue )
	{
		aValue ^= aValue >> 23;
		aValue *= 0x2127599bf4325c37ull;
		aValue ^= aValue >> 47;
		return aValue;
	}
	
	HOST_DEVICE
	uint64 fasthash64(uint32 const *buf, size_t len, uint64 seed = 0)
	{
		uint64 const    m = 0x880355f21e6d1965ULL;
		uint32 const *end = buf + (len & ~size_t(1));

		uint64 h = seed ^ (len*sizeof(uint32) * m);

		while( buf != end )
		{
			uint64 v;
			std::memcpy( &v, buf, sizeof(uint64) );
			buf += 2;
			
			h ^= fasthash64_mix(v);
			h *= m;
		}

		if( len & 1 )
		{
			uint64 v = 0;
			v ^= *buf;
			h ^= fasthash64_mix(v);
			h *= m;
		}

		return fasthash64_mix(h);
	} 

	// See http://blog.michaelschmatz.com/2016/04/11/how-to-write-a-bloom-filter-cpp/
	HOST_DEVICE
	uint32 nth_hash( uint32 aN, uint32 aHashA, uint32 aHashB, uint32 aSize )
	{
		return (aHashA + aN * aHashB) % aSize;
	}
#	endif // ~  USE_BLOOM_FILTER

	template<typename T>
	HOST_DEVICE double to_MB(T bytes)
	{
		return double(bytes) / double(1 << 20);
	}

	HOST_DEVICE uint32 subtract_mod(uint32 value, uint32 max /* exclusive */)
	{
		if (value == 0)
		{
			return max - 1;
		}
		else
		{
			return value - 1;
		}
	}

	template<typename T>
	HOST_DEVICE constexpr T divide_ceil(T Dividend, T Divisor)
	{
		return 1 + (Dividend - 1) / Divisor;
	}

	// Prefetching
	using PfEphemeral = std::integral_constant<int, 0>;
	using PfAllLevels = std::integral_constant<int, 3>;
	using PfLowLocality = std::integral_constant<int, 1>;
	using PfMedLocality = std::integral_constant<int, 2>;
	
	template< class tLocality = PfAllLevels > HOST
	void prefetch_ro( void const* aAddr, tLocality = tLocality{} )
	{
#		if defined(__GNUC__)
		__builtin_prefetch( aAddr, 0, tLocality::value );
#else
        checkAlways(false);
#		endif // ~ GCC
	}

    HOST double seconds()
    {
        static auto start = std::chrono::high_resolution_clock::now();
        return double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count()) / 1.e9;
    }
}
