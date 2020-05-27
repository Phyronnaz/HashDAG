#include "basic_dag.h"
#include "cuda_error_check.h"
#include "color_utils.h"
#include "dag_info.h"
#include "serializer.h"
#include "dags/hash_dag/hash_dag_globals.h"

void BasicDAGFactory::save_uncompressed_colors_to_file(const BasicDAGUncompressedColors& colors, const std::string& path)
{
	PROFILE_FUNCTION();
	
	checkAlways(colors.is_valid());

	FileWriter writer(path);

	writer.write(colors.topLevels);
	writer.write(colors.enclosedLeaves);
	writer.write(colors.leaf.colors);
}

void BasicDAGFactory::load_uncompressed_colors_from_file(BasicDAGUncompressedColors& outColors, const std::string& path)
{
	PROFILE_FUNCTION();
	
	checkAlways(!outColors.is_valid());

	FileReader reader(path);

	reader.read(outColors.topLevels);
	reader.read(outColors.enclosedLeaves, "basic dag enclosed leaves", EMemoryType::GPU_Managed);
	reader.read(outColors.leaf.colors, "basic dag uncompressed colors", EMemoryType::GPU_Managed);
}

void BasicDAGFactory::save_compressed_colors_to_file(const BasicDAGCompressedColors& colors, const std::string& path)
{
	PROFILE_FUNCTION();
	
	checkAlways(colors.is_valid());

	// Swap again before saving, keeping the files on disk in the default order
    ColorUtils::swap_byte_order(const_cast<BasicDAGCompressedColors&>(colors).leaf.weights_CPU);
	
	FileWriter writer(path);

	writer.write(colors.topLevels);
	writer.write(colors.enclosedLeaves);
	writer.write(colors.leaf.weights_CPU);
	writer.write(colors.leaf.blocks_CPU);
	writer.write(colors.leaf.macroBlocks_CPU);

	// Swap back and pretend nothing happened
	ColorUtils::swap_byte_order(const_cast<BasicDAGCompressedColors&>(colors).leaf.weights_CPU);
}

void BasicDAGFactory::load_compressed_colors_from_file(BasicDAGCompressedColors& outColors, const std::string& path, bool enclosedLeavesCompat32)
{
	PROFILE_FUNCTION();
	
    checkAlways(!outColors.is_valid());

    FileReader reader(path);

    reader.read(outColors.topLevels);
    if (enclosedLeavesCompat32)
    {
        checkAlways((std::is_same<EnclosedLeavesType, uint64>::value));
        StaticArray<uint32> dummyEnclosedLeaves;
        reader.read(dummyEnclosedLeaves, "basic dag enclosed leaves", EMemoryType::CPU);
        dummyEnclosedLeaves.free();

        outColors.enclosedLeaves = StaticArray<EnclosedLeavesType>::allocate("temp", dummyEnclosedLeaves.size(), EMemoryType::CPU);
        for (auto& enclosedLeave : outColors.enclosedLeaves)
            enclosedLeave = 0;
    }
    else
    {
        reader.read(outColors.enclosedLeaves, "basic dag enclosed leaves", EMemoryType::GPU_Managed);
    }
    reader.read(outColors.leaf.weights_CPU, "basic dag weights", EMemoryType::CPU);
    reader.read(outColors.leaf.blocks_CPU, "basic dag blocks", EMemoryType::CPU);
    reader.read(outColors.leaf.macroBlocks_CPU, "basic dag macro blocks", EMemoryType::CPU);
	outColors.leaf.set_as_unique();

    ColorUtils::swap_byte_order(outColors.leaf.weights_CPU);

    checkfAlways(outColors.topLevels <= C_colorTreeLevels, "C_colorTreeLevels is too low: is %u, but should be at least %u", C_colorTreeLevels, outColors.topLevels);

    outColors.leaf.upload_to_gpu();
}

void BasicDAGFactory::save_dag_to_file(const DAGInfo& info, const BasicDAG& dag, const std::string& path)
{
	PROFILE_FUNCTION();
	
	checkAlways(dag.is_valid());

	FileWriter writer(path);

	writer.write(info);
	writer.write(dag.levels);
	writer.write(dag.data);
}

void BasicDAGFactory::load_dag_from_file(DAGInfo& outInfo, BasicDAG& outDag, const std::string& path)
{
	PROFILE_FUNCTION();
	
	checkAlways(!outDag.is_valid());

	FileReader reader(path);

	reader.read(outInfo);
	uint32 levels = 0;
	reader.read(levels);
    checkfAlways(levels == MAX_LEVELS, "MAX_LEVELS is %u, should be %u", MAX_LEVELS, levels);
	reader.read(outDag.data, "basic dag nodes", EMemoryType::GPU_Managed);
}