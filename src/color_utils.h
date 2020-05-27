#pragma once

#include "typedefs.h"
#include "cuda_math.h"
#include "array.h"

namespace ColorUtils
{
	HOST_DEVICE constexpr uint16 float3_to_rgb565(float3 rgb)
	{
		const float r = clamp(rgb.x, 0.f, 1.f);
		const float g = clamp(rgb.y, 0.f, 1.f);
		const float b = clamp(rgb.z, 0.f, 1.f);
		return uint16(
			(uint16(r * 31.0f) << 0) |
			(uint16(g * 63.0f) << 5) |
			(uint16(b * 31.0f) << 11));
	}

	HOST_DEVICE constexpr float3 rgb565_to_float3(uint16 rgb)
	{
		return make_vector3<float3>(
			float((rgb >> 0) & 0x1F) / 31.0f,
			float((rgb >> 5) & 0x3F) / 63.0f,
			float((rgb >> 11) & 0x1F) / 31.0f);
	}

	HOST_DEVICE constexpr uint32 float3_to_rgb888(float3 rgb)
	{
		const float r = clamp(rgb.x, 0.f, 1.f);
		const float g = clamp(rgb.y, 0.f, 1.f);
		const float b = clamp(rgb.z, 0.f, 1.f);
		return
			(uint32(r * 255.0f) << 0) |
			(uint32(g * 255.0f) << 8) |
			(uint32(b * 255.0f) << 16) | 0xff000000;
	}
	HOST_DEVICE constexpr float3 rgb888_to_float3(uint32 rgb)
	{
		return make_vector3<float3>(
			float((rgb >> 0) & 0xFF) / 255.0f,
			float((rgb >> 8) & 0xFF) / 255.0f,
			float((rgb >> 16) & 0xFF) / 255.0f);
	}

	HOST_DEVICE constexpr float3 rgb101210_to_float3(uint32 rgb) {
		return make_vector3<float3>(
			float((rgb >> 0) & 0x3FF) / 1023.0f,
			float((rgb >> 10) & 0xFFF) / 4095.0f,
			float((rgb >> 22) & 0x3FF) / 1023.0f);
	}
	HOST_DEVICE constexpr uint32 float3_to_rgb101210(float3 rgb) {
		const float r = clamp(rgb.x, 0.f, 1.f);
		const float g = clamp(rgb.y, 0.f, 1.f);
		const float b = clamp(rgb.z, 0.f, 1.f);
		return // NOTE: was round here, but changed to a simple int cast as round is not constexpr
			(uint32(r * 1023.0f) << 0) |
			(uint32(g * 4095.0f) << 10) |
			(uint32(b * 1023.0f) << 22);
	}

	HOST_DEVICE constexpr uint32 rgb565_to_rgb888(uint16 rgb)
	{
		return float3_to_rgb888(rgb565_to_float3(rgb));
	}
	HOST_DEVICE constexpr uint16 rgb888_to_rgb565(uint32 rgb)
	{
		return float3_to_rgb565(rgb888_to_float3(rgb));
	}
	HOST_DEVICE float color_error(float3 a, float3 b)
	{
		return length(a - b);
	}
	HOST_DEVICE constexpr float get_decimal_weight(uint8 weight, uint8 bitsPerWeight)
	{
		return float(weight) / float((1 << bitsPerWeight) - 1);
	}

	HOST_DEVICE constexpr uint32 make_color_bits(float3 minColor, float3 maxColor)
    {
        return float3_to_rgb565(minColor) | (uint32(float3_to_rgb565(maxColor)) << 16);
    }
	
#if CFG_COLOR_SWAP_BYTE_ORDER
	HOST_DEVICE uint32 swap_byte_order( uint32 aValue )
	{
		return
			  ((aValue & 0x000000ff) << 24)
			| ((aValue & 0x0000ff00) <<  8)
			| ((aValue & 0x00ff0000) >>  8)
			| ((aValue & 0xff000000) >> 24)
		;
	}
	HOST void swap_byte_order(uint32* aArray, std::size_t aCount)
	{
		for (std::size_t i = 0; i < aCount; ++i)
			aArray[i] = swap_byte_order(aArray[i]);
	}
	HOST void swap_byte_order(StaticArray<uint32>& array)
	{
		for(uint32& element : array)
		{
			element = swap_byte_order(element);
		}
	}
#else
	HOST_DEVICE uint32 swap_byte_order(uint32 aValue) { return aValue; }
	HOST void swap_byte_order(uint32*, std::size_t) {}
	HOST void swap_byte_order(StaticArray<uint32>&) {}
#endif // ~ CFG_COLOR_SWAP_BYTE_ORDER

	HOST_DEVICE uint8 extract_bits(const uint32 bits, const StaticArray<uint32> array, const uint64 bitPtr)
	{
		if (bits == 0) return uint8(0);

#if !CFG_COLOR_SWAP_BYTE_ORDER
		const uint32 ptrWord = cast<uint32>(bitPtr / 32);
		const uint32 ptrBit = cast<uint32>(bitPtr % 32);
		const uint32 bitsLeft = 32 - ptrBit;
		// Need to be careful not to try to shift >= 32 steps (undefined)
		const uint32 upperMask = (bitsLeft == 32) ? 0xFFFFFFFF : (~(0xFFFFFFFFu << bitsLeft));
		if (bitsLeft >= bits)
		{
			uint32 val = upperMask & array[ptrWord];
			val >>= (bitsLeft - bits);
			check(val < uint32(1 << bits));
			return uint8(val);
		}
		else
		{
			uint32 val = (upperMask & array[ptrWord]) << (bits - bitsLeft);
			val |= array[ptrWord + 1] >> (32 - (bits - bitsLeft));
			check(val < uint32(1 << bits));
			return uint8(val);
		}
#else
		auto const ptrWord = bitPtr / 8;

		uint16 dst;
		std::memcpy(&dst, reinterpret_cast<uint8 const*>(array.data()) + ptrWord, sizeof(uint16));
		dst = uint16((dst << 8) | (dst >> 8));

		auto const ptrBit = bitPtr % 8;
		auto const shift = 16 - bits - ptrBit;
		auto const mask = (1u << bits) - 1;
		return uint8((dst >> shift) & mask);
#endif
	}
}