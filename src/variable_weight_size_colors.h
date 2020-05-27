#pragma once

#include "typedefs.h"
#include "utils.h"
#include "color_utils.h"
#include "memory.h"

constexpr uint32_t CColorsPerMacroBlock = 16 * 1024;

namespace VariableColorsUtils
{
	HOST_DEVICE uint16 get_block_header_color_index(uint32 header)
	{
		return cast_bits<14>(header & (CColorsPerMacroBlock - 1));
	};
	HOST_DEVICE uint16 get_block_header_weight_offset(uint32 header)
	{
		return cast<uint16>(header >> 16);
	}
	HOST_DEVICE uint8 get_block_header_bits_per_weight(uint32 header)
	{
		if (get_block_header_weight_offset(header) == 0xFFFF)
		{
			return 0;
		}
		else
		{
			return cast<uint8>(((header >> 14) & 0x3) + 1);
		}
	}
	HOST_DEVICE uint32 make_block_header(uint32 weightOffset, uint32 bitsPerWeight, uint32 index)
	{
		check(weightOffset != 0xFFFF);
		
		if (bitsPerWeight == 0)
		{
			weightOffset = 0xFFFF;
		}
		else
		{
			bitsPerWeight--;
		}
		
		check(cast_bits<16>(weightOffset) == weightOffset);
		check(cast_bits<2>(bitsPerWeight) == bitsPerWeight);
		check(cast_bits<14>(index) == index);

		return
			(weightOffset << 16) |
			(bitsPerWeight << 14) |
			index;
	}
}

struct UncompressedColor
{
	uint32 color = 0;

	UncompressedColor() = default;
	HOST_DEVICE constexpr UncompressedColor(uint32 color) : color(color) {}
	HOST_DEVICE constexpr UncompressedColor(float3 color) : color(ColorUtils::float3_to_rgb888(color)) {}
	HOST_DEVICE constexpr UncompressedColor(float color) : color(ColorUtils::float3_to_rgb888(make_float3(color))) {}

	HOST_DEVICE uint32 get_debug_hash() const
	{
		return 0;
	}
	HOST_DEVICE float3 get_min_color() const
	{
		return {};
	}
	HOST_DEVICE float3 get_max_color() const
	{
		return {};
	}
	HOST_DEVICE float get_weight() const
	{
		return 0;
	}
	HOST_DEVICE float3 get_color() const
	{
		return ColorUtils::rgb888_to_float3(color);
	}
};

struct CompressedColor
{
	uint32 colorBits = 0;
	uint8 weight = 0;
	uint8 bitsPerWeight = 0;
	
#if COLOR_DEBUG
    uint32 debugHash = 0;
#endif

    HOST_DEVICE uint32 get_debug_hash() const
    {
#if COLOR_DEBUG
        //return uint32(Utils::murmurhash64((uint64(debugHash) << 32) | colorBits));
		return debugHash;
#else
        return 0;
#endif
    }

	HOST_DEVICE constexpr float3 get_min_color() const
	{
		return ColorUtils::rgb565_to_float3(colorBits & 0xFFFF);
	}
	HOST_DEVICE constexpr float3 get_max_color() const
	{
		return ColorUtils::rgb565_to_float3(uint16((colorBits >> 16) & 0xFFFF));
	}
	HOST_DEVICE constexpr float get_weight() const
	{
		if (bitsPerWeight == 0)
		{
			return 0;
		}
		return ColorUtils::get_decimal_weight(weight, bitsPerWeight);
	}
	HOST_DEVICE constexpr void set_min_color(float3 color)
	{
		colorBits &= 0xFFFF0000;
		colorBits |= ColorUtils::float3_to_rgb565(color);
	}
	HOST_DEVICE constexpr void set_max_color(float3 color)
	{
		colorBits &= 0x0000FFFF;
		colorBits |= ColorUtils::float3_to_rgb565(color) << 16;
	}
	HOST_DEVICE constexpr void set_weight(float value)
	{
		//weight = (uint8)clamp<int32>((int32)floor(value * float((1 << bitsPerWeight) - 1)), 0, (1 << bitsPerWeight) - 1); has floor function call = bad
		weight = uint8(clamp<int32>(int32(value * float((1 << bitsPerWeight) - 1)), 0, (1 << bitsPerWeight) - 1));
	}
	
	HOST_DEVICE constexpr void set_single_color(float3 color)
    {
		bitsPerWeight = 0;
		colorBits = ColorUtils::float3_to_rgb101210(color);
    }
	
	HOST_DEVICE constexpr float3 get_color() const
	{
		if (bitsPerWeight == 0)
		{
			return ColorUtils::rgb101210_to_float3(colorBits);
		}
		else
		{
			return lerp(get_min_color(), get_max_color(), get_weight());
		}
	}
};

struct CompressedColorLeaf
{
private:
	uint64 offset = 0; // offset into the global shared leaf. -1 if unique
	static constexpr uint64 uniqueOffset = uint64(-1);

	StaticArray<uint32> weights_GPU;
	StaticArray<uint64> blocks_GPU;
	StaticArray<uint64> macroBlocks_GPU;

public:
	// Array of bits
	StaticArray<uint32> weights_CPU;

	/////////////////////////////////////////////////////////////////////////
	// FIRST 32 BITS:
	// Array of: | weight offset | bits per color - 1  | color index  |  
	//           | 31    -    16 | 15       -       14 | 13    -    0 |
	// if weight index = 0xFFFF, bits per color = 0
	/////////////////////////////////////////////////////////////////////////
	// LAST 32 BITS:
	// Array of: | max color  | min color  |  
	//           | B | G  | R | B | G  | R |
	//           | 5 | 10 | 5 | 5 | 10 | 5 |
	// OR:       |          color          |
	//           |   B    |   G    |   R   |
	// 			 |   10   |   12   |   10  |
	// if bits per color = 0
	StaticArray<uint64> blocks_CPU;

	// Array [64 bits first block, 64 bits weight offset]
	// Block weight index is relative to those weight offsets
	// To get the first macro block of a color index: macroBlocks[2 * (colorIndex / CColorsPerMacroBlock) + 0]
	// To get the weight offset     of a color index: macroBlocks[2 * (colorIndex / CColorsPerMacroBlock) + 1]
	StaticArray<uint64> macroBlocks_CPU;

    HOST double size_in_MB() const
    {
        return weights_CPU.size_in_MB() + blocks_CPU.size_in_MB() + macroBlocks_CPU.size_in_MB();
    }

    HOST void upload_to_gpu()
    {
		PROFILE_FUNCTION();
    	
        check(blocks_CPU.is_valid());
        check(macroBlocks_CPU.is_valid());
        check(!blocks_GPU.is_valid());
        check(!weights_GPU.is_valid());
        check(!macroBlocks_GPU.is_valid());

        blocks_GPU = blocks_CPU.create_gpu();
        if (weights_CPU.size() > 0)
        {
            weights_GPU = weights_CPU.create_gpu();
        }
        macroBlocks_GPU = macroBlocks_CPU.create_gpu();
    }

#ifdef __CUDA_ARCH__
#define weights weights_GPU
#define blocks blocks_GPU
#define macroBlocks macroBlocks_GPU
#else
#define weights weights_CPU
#define blocks blocks_CPU
#define macroBlocks macroBlocks_CPU
#endif

public:
	HOST_DEVICE bool is_shared() const
	{
		return offset != uniqueOffset;
	}
	HOST_DEVICE void set_as_unique()
	{
		offset = uniqueOffset;
	}
	HOST_DEVICE void set_as_shared(uint64 inOffset)
	{
		check(inOffset != uniqueOffset);
		offset = inOffset;
	}
	HOST_DEVICE uint64 get_offset() const
	{
		check(is_shared());
		return offset;
	}

public:
	HOST_DEVICE bool is_valid() const
	{
        check(weights_CPU.size() == weights_GPU.size());
        check(blocks_CPU.size() == blocks_GPU.size());
        check(macroBlocks_CPU.size() == macroBlocks_GPU.size());
		check(blocks.is_valid() == macroBlocks.is_valid()); // Weights can be empty
		return blocks.is_valid();
	}
	HOST_DEVICE bool is_valid_index(uint64 index) const
    {
        if (is_shared())
        {
            index += offset;
        }
        return macroBlocks.is_valid_index(2 * index / CColorsPerMacroBlock);
    }
	HOST void free()
    {
		PROFILE_FUNCTION();
		
        if (!is_shared())
        {
            weights_CPU.free();
            weights_GPU.free();
            blocks_CPU.free();
            blocks_GPU.free();
            macroBlocks_CPU.free();
            macroBlocks_GPU.free();
        }
        else
        {
            reset();
        }
    }
    HOST void reset()
    {
        weights_CPU.reset();
        weights_GPU.reset();
        blocks_CPU.reset();
        blocks_GPU.reset();
        macroBlocks_CPU.reset();
        macroBlocks_GPU.reset();
    }
    HOST void print_stats() const
	{
		printf(
			"Color stats:\n"
			"\t%" PRIu64 " blocks\n"
			"\t%fMB used by blocks\n"
			"\t%fMB used by weights\n"
			"\t%fMB used by macro blocks\n"
			"\tTotal: %fMB\n",
			blocks.size(),
			blocks.size_in_MB(),
			weights.size_in_MB(),
			macroBlocks.size_in_MB(),
			blocks.size_in_MB() + weights.size_in_MB() + macroBlocks.size_in_MB());
	}
	HOST_DEVICE bool operator==(const CompressedColorLeaf& other) const
	{
        check(weights_CPU.size() == weights_GPU.size());
        check(blocks_CPU.size() == blocks_GPU.size());
        check(macroBlocks_CPU.size() == macroBlocks_GPU.size());
		return
			offset == other.offset &&
			weights == other.weights &&
			blocks == other.blocks &&
			macroBlocks == other.macroBlocks;
	}

public:
	HOST_DEVICE uint64 get_macro_block_first_block_index(uint64 macroBlockIndex) const
	{
		return macroBlocks[2 * macroBlockIndex + 0];
	}
	HOST_DEVICE uint64 get_macro_block_weight_offset(uint64 macroBlockIndex) const
	{
		return macroBlocks[2 * macroBlockIndex + 1];
	}
	
	HOST_DEVICE uint32 get_block_header(uint32 blockIndex) const
	{
		return uint32(blocks[blockIndex]);
	}
	HOST_DEVICE uint64 get_macro_block_last_block_index(uint64 macroIndex) const
	{
		if (2 * (macroIndex + 1) < macroBlocks.size())
		{
			return get_macro_block_first_block_index(macroIndex + 1) - 1;
		}
		else
		{
			return blocks.size() - 1;
		}
	}

	HOST_DEVICE uint32 binary_search_blocks(uint64 colorIndex) const
	{
		checkf(colorIndex < CColorsPerMacroBlock * macroBlocks.size() / 2, "Color index: %" PRIu64 ", number of macros: %" PRIu64, colorIndex, macroBlocks.size());
		
		const uint16 localColorIndex = cast<uint16>(colorIndex % CColorsPerMacroBlock);
		const uint32 macroBlockIndex = cast<uint32>(colorIndex / CColorsPerMacroBlock);

		uint32 lowerBound = cast<uint32>(get_macro_block_first_block_index(macroBlockIndex));
		uint32 upperBound = cast<uint32>(get_macro_block_last_block_index(macroBlockIndex));
		checkf(lowerBound <= upperBound, "%u !<= %u", lowerBound, upperBound);
		
		uint32 position = (lowerBound + upperBound) / 2;
		uint32 blockHeader = get_block_header(position);
		while (VariableColorsUtils::get_block_header_color_index(blockHeader) != localColorIndex && lowerBound <= upperBound)
		{
			if (VariableColorsUtils::get_block_header_color_index(blockHeader) > localColorIndex)
			{
				upperBound = position - 1;
			}
			else
			{
				lowerBound = position + 1;
			}

			position = (lowerBound + upperBound) / 2;
			blockHeader = get_block_header(position);
		}
		return position;
	}

	HOST_DEVICE CompressedColor get_color_for_block(const uint32 blockIndex, const uint64 colorIndex) const
	{
		const uint16 localColorIndex = cast<uint16>(colorIndex % CColorsPerMacroBlock);
		const uint32 macroBlockIndex = cast<uint32>(colorIndex / CColorsPerMacroBlock);

		const uint64 block = blocks[blockIndex];
		const uint32 blockHeader = uint32(block);
		const uint32 blockColor = uint32(block >> 32);

		const uint8 bitsPerWeight = VariableColorsUtils::get_block_header_bits_per_weight(blockHeader);

		CompressedColor color;
		color.bitsPerWeight = bitsPerWeight;
		color.colorBits = blockColor;
		if (bitsPerWeight != 0)
		{
            const uint64 macroWeightOffset = get_macro_block_weight_offset(macroBlockIndex);
			const uint32 blockWeightOffset = VariableColorsUtils::get_block_header_weight_offset(blockHeader);
			const uint32 localWeightOffset = cast<uint32>(localColorIndex - VariableColorsUtils::get_block_header_color_index(blockHeader)) * bitsPerWeight;
            const uint64 weightIndex = macroWeightOffset + blockWeightOffset + localWeightOffset;
            color.weight = ColorUtils::extract_bits(bitsPerWeight, weights, weightIndex);
		}
#if COLOR_DEBUG
		color.debugHash = uint32(Utils::murmurhash64(reinterpret_cast<uint64>(&blocks[blockIndex])));
		//color.debugHash = blockIndex;
		//color.debugHash = macroBlockIndex;
		//color.debugHash = uint32(Utils::murmurhash64(bitsPerWeight));
		//color.debugHash = uint32(Utils::murmurhash64(reinterpret_cast<uint64>(&macroBlocks[2 * macroBlockIndex])));
#endif
		return color;
	}

	HOST_DEVICE CompressedColor get_color(uint64 colorIndex) const
    {
        if (is_shared())
        {
            colorIndex += offset;
        }
		const uint32 position = binary_search_blocks(colorIndex);
		return get_color_for_block(position, colorIndex);
	}

	template<typename T>
	HOST void copy_colors(T& leafBuilder, const uint64 start, const uint64 count) const
	{
		PROFILE_FUNCTION_SLOW();
		
#if 0//!NEW_COPY_COLORS
		const auto get_macro_index = [](uint64 colorIndex)
		{
			return cast<uint32>(colorIndex / CColorsPerMacroBlock);
		};
		const auto get_macro_local_index = [](uint64 colorIndex)
		{
			return cast<uint32>(colorIndex % CColorsPerMacroBlock);
		};
		const auto get_color_index = [&](uint32 blockIndex)
		{
			return VariableColorsUtils::get_block_header_color_index(get_block_header(blockIndex));
		};

		uint32 macroIndex = uint32(-1);
		uint64 macroEnd = 0; // exclusive
		uint32 blockIndex = uint32(-1);
		// Iterate the colors
		for (uint64 colorIndex = start; colorIndex < start + count; colorIndex++)
		{
			// On the end of the current macro, get the next macro
			if (get_macro_index(colorIndex) != macroIndex)
			{
				macroIndex = get_macro_index(colorIndex);
				macroEnd = get_macro_block_last_block_index(macroIndex) + 1; // +!: cuz exclusive
				blockIndex = binary_search_blocks(colorIndex);
			}

			const uint32 localColorIndex = get_macro_local_index(colorIndex);

			// Increment blockIndex once we've iterated over all its elements
			check(get_color_index(blockIndex) <= localColorIndex);
			if (blockIndex + 1 < macroEnd && get_color_index(blockIndex + 1) <= localColorIndex)
			{
				blockIndex++;
			}
			checkf(blockIndex + 1 == macroEnd || get_color_index(blockIndex + 1) > localColorIndex, "%u !> %u", get_color_index(blockIndex + 1), localColorIndex);

			leafBuilder.add(get_color_for_block(blockIndex, colorIndex));
		}
#else
		const auto get_macro_index = [](uint64 colorIndex)
		{
			return cast<uint32>(colorIndex / CColorsPerMacroBlock);
		};
		const auto get_macro_local_index = [](uint64 colorIndex)
		{
			return cast<uint32>(colorIndex % CColorsPerMacroBlock);
		};
		const auto get_block_color_index = [&](uint32 blockIndex)
		{
			return VariableColorsUtils::get_block_header_color_index(get_block_header(blockIndex));
		};
		
		uint32 macroBlockIndex = get_macro_index(start);
        uint32 blockIndex = binary_search_blocks(start);

        uint64 macroLastBlockIndex;
        uint64 macroWeightOffset;

        uint32 blockHeader;
        uint32 blockColor;
        uint8 bitsPerWeight;
        // These 2 are only set if bitsPerWeight > 0
        uint32 blockWeightOffset;
        uint32 blockColorIndex;

        const auto update_block = [&]()
        {
            const uint64 block = blocks[blockIndex];
            blockHeader = uint32(block);
            blockColor = uint32(block >> 32);
            bitsPerWeight = VariableColorsUtils::get_block_header_bits_per_weight(blockHeader);
            if (bitsPerWeight > 0)
            {
                blockWeightOffset = VariableColorsUtils::get_block_header_weight_offset(blockHeader);
                blockColorIndex = VariableColorsUtils::get_block_header_color_index(blockHeader);
            }
        };
        const auto update_macro_block = [&]()
        {
            macroLastBlockIndex = get_macro_block_last_block_index(macroBlockIndex);
            macroWeightOffset = get_macro_block_weight_offset(macroBlockIndex);
        };

        update_block();
        update_macro_block();

		// Iterate the colors
		for (uint64 colorIndex = start; colorIndex < start + count; colorIndex++)
		{
			// On the end of the current macro, get the next macro
			if (get_macro_index(colorIndex) != macroBlockIndex)
			{
				macroBlockIndex++;
				blockIndex++;
				checkEqual(macroBlockIndex, get_macro_index(colorIndex));
                checkEqual(blockIndex, binary_search_blocks(colorIndex));

                update_block();
                update_macro_block();
			}

			const uint32 localColorIndex = get_macro_local_index(colorIndex);

			checkInfEqual(get_block_color_index(blockIndex), localColorIndex);
			checkInfEqual(blockIndex, macroLastBlockIndex);
			// Increment blockIndex once we've iterated over all its elements
			if (blockIndex != macroLastBlockIndex && get_block_color_index(blockIndex + 1) <= localColorIndex)
			{
				blockIndex++;
                update_block();
			}
            if (blockIndex != macroLastBlockIndex)
            {
                // If we're not at the end, then we must be inferior to the next block
			    checkInf(localColorIndex, get_block_color_index(blockIndex + 1));
            }

            CompressedColor color;
            color.bitsPerWeight = bitsPerWeight;
            color.colorBits = blockColor;
            if (bitsPerWeight != 0)
            {
                const uint32 localWeightOffset = cast<uint32>(localColorIndex - blockColorIndex) * bitsPerWeight;
                const uint64 weightIndex = macroWeightOffset + blockWeightOffset + localWeightOffset;
                color.weight = ColorUtils::extract_bits(bitsPerWeight, weights, weightIndex);
            }
			leafBuilder.add(color);
		}
#endif
	}
#undef weights
#undef blocks
#undef macroBlocks
};

struct ColorLeafBuilder
{
public:
	HOST void add_weight(uint8 weight, uint8 bitsPerWeight)
	{
		check(bitsPerWeight != 0);
		checkEqual((weight & ((1 << bitsPerWeight) - 1)), weight);

		auto const target = (targetBitPos + 32 - bitsPerWeight) % 64;
		auto const ww = uint64(weight) << target;

        if (target > uint32(32 - bitsPerWeight))
        {
            auto& buffer = weights.back();
            auto const whi = uint32(ww >> 32);

            buffer |= whi;
            targetBitPos -= bitsPerWeight;
        }

        if (target < 32)
        {
            // auto& buffer = weights.emplace_back(0); // Since C++17
            weights.emplace_back(0u);
            auto& buffer = weights.back();
            auto const wlo = uint32(ww & ~uint32(0));

            buffer |= wlo;
            targetBitPos = target; // == 32 - (32- target);
        }

		numWeights += bitsPerWeight;
	}
	HOST void add(const CompressedColor& color)
	{
		if (colorIndex % CColorsPerMacroBlock == 0)
		{
            lastMacroStartColorIndex = colorIndex;
            lastMacroStartWeightIndex = numWeights;
			macroBlocks.push_back(MacroBlockStruct{ numWeights, cast<uint32>(blocks.size()) });
		}

		// Need to create a new block if on macro boundaries, or if the color is different from the previous one
		if (colorIndex % CColorsPerMacroBlock == 0 || lastBlockColorBits != color.colorBits || lastBlockBitsPerWeight != color.bitsPerWeight)
		{
            lastBlockColorBits = color.colorBits;
            lastBlockBitsPerWeight = color.bitsPerWeight;
            blocks.push_back(BlockStruct
                                     {
                                             color.bitsPerWeight,
                                             color.colorBits,
                                             numWeights - lastMacroStartWeightIndex,
                                             colorIndex - lastMacroStartColorIndex
                                     });
		}
		
		if (color.bitsPerWeight > 0) 
		{
			add_weight(color.weight, color.bitsPerWeight);
		}

        colorIndex++;
    }
    HOST void add_large_single_color(float3 singleColor, const uint32 numVoxels)
    {
		PROFILE_FUNCTION_SLOW();
		
        CompressedColor color;
        color.set_single_color(singleColor);
#if 0
        for (uint32 i = 0; i < numVoxels; i++)
        {
            add(color);
        }
#else
#if ENABLE_CHECKS
        const uint32 startColorIndex = colorIndex;
#endif
        uint32 numVoxelsLeft = numVoxels;
        while (numVoxelsLeft > 0)
        {
            const uint32 numColorsLeftInMacro = CColorsPerMacroBlock - (colorIndex % CColorsPerMacroBlock);
            const uint32 numVoxelsToAddToMacro = min(numColorsLeftInMacro, numVoxelsLeft);
            numVoxelsLeft -= numVoxelsToAddToMacro;
            add(color);
            colorIndex += numVoxelsToAddToMacro - 1; // -1: we just added one
            check(numVoxelsLeft == 0 || (colorIndex % CColorsPerMacroBlock) == 0);
        }
        checkEqual(colorIndex - startColorIndex, numVoxels);
#endif
    }

	HOST void build(CompressedColorLeaf& leaf, bool canFree) const
	{
		PROFILE_FUNCTION();
		
        if (!leaf.is_shared() && canFree)
        {
            leaf.free();
        }
        else
        {
            leaf.reset();
        }

		leaf.set_as_unique();

		leaf.blocks_CPU = StaticArray<uint64>::allocate("leaf blocks", blocks.size(), EMemoryType::CPU);
		for (uint64 blockIndex = 0; blockIndex < blocks.size(); blockIndex++)
		{
			auto& block = blocks[blockIndex];
			leaf.blocks_CPU[blockIndex] = (uint64(block.colorBits) << 32) | VariableColorsUtils::make_block_header(block.weightOffset, block.bitsPerWeight, block.index);
		}

		leaf.weights_CPU = nullptr;
		if (!weights.empty()) 
		{
			leaf.weights_CPU = StaticArray<uint32>::allocate("leaf weights", weights.size(), EMemoryType::CPU);
			for (uint64 weightIndex = 0; weightIndex < weights.size(); weightIndex++)
			{
				leaf.weights_CPU[weightIndex] = ColorUtils::swap_byte_order(weights[weightIndex]);
			}
		}

		leaf.macroBlocks_CPU = StaticArray<uint64>::allocate("leaf macro blocks", 2 * macroBlocks.size(), EMemoryType::CPU);
		for (uint64 macroBlockIndex = 0; macroBlockIndex < macroBlocks.size(); macroBlockIndex++)
		{
			leaf.macroBlocks_CPU[2 * macroBlockIndex + 0] = macroBlocks[macroBlockIndex].startBlockIndex;
			leaf.macroBlocks_CPU[2 * macroBlockIndex + 1] = macroBlocks[macroBlockIndex].startWeightIndex;
		}

		leaf.upload_to_gpu();
    }

	inline uint32 get_color_index() const
	{
		return colorIndex;
	}

private:
	struct BlockStruct
	{
		uint32 bitsPerWeight;
		uint32 colorBits;
		uint32 weightOffset;
		uint32 index;
	};
	struct MacroBlockStruct
	{
		uint32 startWeightIndex;
		uint32 startBlockIndex;
	};

	std::vector<BlockStruct> blocks;
	std::vector<uint32> weights;
	std::vector<MacroBlockStruct> macroBlocks; // uint32: not big enough to need uint64
	uint32 numWeights = 0;
	uint32 targetBitPos = 0;

    uint32 colorIndex = 0; // number of added colors
    uint32 lastBlockBitsPerWeight = 0;
    uint32 lastBlockColorBits = 0;
    uint32 lastMacroStartColorIndex = 0;
    uint32 lastMacroStartWeightIndex = 0;
};
