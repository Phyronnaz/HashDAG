#include "typedefs.h"

#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag_factory.h"
#include "dags/dag_utils.h"
#include "engine.h"

int main(int argc, char** argv)
{
    PROFILE_FUNCTION();
	
	auto& engine = Engine::engine;

	printf("Using " SCENE "\n");
    printf("%d levels (resolution=%d^3)\n", MAX_LEVELS, 1 << MAX_LEVELS);
#if ENABLE_CHECKS
    std::fprintf(stderr, "CHECKS: ENABLED\n");
#else
    printf("CHECKS: DISABLED\n");
#endif
    printf("IMAGE RESOLUTION: %ux%u\n", imageWidth, imageHeight);

    const std::string fileName = std::string(SCENE) + std::to_string(1 << (SCENE_DEPTH - 10)) + "k";

    if (LOAD_UNCOMPRESSED_COLORS)
    {
        BasicDAGFactory::load_uncompressed_colors_from_file(engine.basicDagUncompressedColors, "data/" + fileName + ".basic_dag.uncompressed_colors.bin");
    }
    if (LOAD_COMPRESSED_COLORS)
    {
        BasicDAGFactory::load_compressed_colors_from_file(engine.basicDagCompressedColors, "data/" + fileName + ".basic_dag.compressed_colors.variable.bin");
    }
    BasicDAGFactory::load_dag_from_file(engine.dagInfo, engine.basicDag, "data/" + fileName + ".basic_dag.dag.bin");

#if 0
    DAGUtils::fix_enclosed_leaves(engine.basicDag, engine.basicDagCompressedColors.enclosedLeaves, engine.basicDagCompressedColors.topLevels);
#if 0
	BasicDAGFactory::save_compressed_colors_to_file(engine.basicDagCompressedColors, "data/" FILENAME ".basic_dag.compressed_colors.variable.bin");
    engine.basicDagCompressedColors.free();
    BasicDAGFactory::load_compressed_colors_from_file(engine.basicDagCompressedColors, "data/" FILENAME ".basic_dag.compressed_colors.variable.bin");
#endif
#endif

	if (LOAD_COMPRESSED_COLORS)
    {
        HashDAGFactory::load_from_DAG(engine.hashDag, engine.basicDag, 0x8FFFFFFF / C_pageSize / sizeof(uint32));
        HashDAGFactory::load_colors_from_DAG(engine.hashDagColors, engine.basicDag, engine.basicDagCompressedColors);
    }

	engine.basicDagColorErrors.uncompressedColors = engine.basicDagUncompressedColors;
	engine.basicDagColorErrors.compressedColors = engine.basicDagCompressedColors;

    //engine.basicDag.free();

	engine.init(HEADLESS);
#if USE_NORMAL_DAG
	engine.set_dag(EDag::BasicDagCompressedColors);
#else
	engine.set_dag(EDag::HashDag);
#endif

//#if USE_VIDEO
//	engine.toggle_fullscreen();
//    engine.videoManager.load_video("./videos/" SCENE "_" VIDEO_NAME ".txt");
//	std::this_thread::sleep_for(std::chrono::seconds(5));
//#else
//    engine.replayReader.load_csv("./replays/" SCENE "_" REPLAY_NAME ".csv");
//#endif

	printf("Starting...\n");

#ifdef PROFILING_PATH
    engine.hashDag.data.save_bucket_sizes(true);
#endif

	engine.loop();
	engine.destroy();

	return 0;
}
