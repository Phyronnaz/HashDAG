#pragma once

#include "typedefs.h"

#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "camera_view.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/basic_dag/basic_dag.h"
#include "dag_tracer.h"
#include "replay.h"
#include "video.h"

#include "glfont.h"

enum class EDag
{
    BasicDagUncompressedColors,
    BasicDagCompressedColors,
    BasicDagColorErrors,
    HashDag
};

constexpr uint32 CNumDags = 4;

std::string dag_to_string(EDag dag);
std::string tool_to_string(ETool tool);

class Engine
{
public:
    static Engine engine;

    BasicDAG basicDag;
    HashDAG hashDag;
    BasicDAGCompressedColors basicDagCompressedColors;
    BasicDAGUncompressedColors basicDagUncompressedColors;
    BasicDAGColorErrors basicDagColorErrors;
    HashDAGColors hashDagColors;
    HashDAGUndoRedo undoRedo;
    DAGInfo dagInfo;
    CameraView view;

    CameraView targetView;
    double targetLerpSpeed = 1;
    CameraView initialView;
    bool moveToTarget = false;
    double targetLerpTime = 0;

    inline void init_target_lerp()
    {
        moveToTarget = true;
        initialView = view;
        targetLerpTime = 0;
    }

    ReplayManager replayReader;
    VideoManager videoManager;

    StatsRecorder statsRecorder;

    struct EditConfig
    {
        EDag currentDag = EDag::HashDag;
        ETool tool = ETool::Sphere;
        EDebugColors debugColors = EDebugColors::None;
        uint32 debugColorsIndexLevel = 0;
        float radius = 10;
        uint3 copySourcePath = make_uint3(0, 0, 0);
        uint3 copyDestPath = make_uint3(0, 0, 0);
        uint3 path;
    };
    EditConfig config;

    template<typename T, typename... TArgs>
    void edit(TArgs&&... Args)
    {
        PROFILE_FUNCTION();
    	
        lastEditTimestamp = statsRecorder.get_frame_timestamp();
    	lastEditFrame = frameIndex;

    	BasicStats stats;
    	
        // is disabled inside the function
        hashDag.data.prefetch();
    	
        stats.start_work("creating edit tool");
        auto tool = T(std::forward<TArgs>(Args)...);
        stats.flush(statsRecorder);
    	
        stats.start_work("total edits");
        hashDag.edit_threads(tool, hashDagColors, undoRedo, statsRecorder);
        stats.flush(statsRecorder);

        stats.start_work("upload_to_gpu");
        hashDag.data.upload_to_gpu();
        stats.flush(statsRecorder);

        statsRecorder.report("radius", tool.radius);
    }
    void set_dag(EDag dag);

    void init(bool headLess);
    void loop();
    void destroy();

	void toggle_fullscreen();
    static void framebuffer_size_callback(GLFWwindow* window_, int width_, int height_);

private:
    struct InputState
    {
        std::vector<bool> keys = std::vector<bool>(GLFW_KEY_LAST + 1, false);
        std::vector<bool> mouse = std::vector<bool>(8, false);
        double mousePosX = 0;
        double mousePosY = 0;
    };

    GLFWwindow* window = nullptr;
    GLuint image = 0;

    InputState state;

    GLuint programID = 0;
    GLint textureID = 0;
	GLuint fsvao = 0;

	double dt = 0;
	bool headLess = false;
    bool firstReplay = true;
    bool printMemoryStats = false;
    bool shadows = true;
    float shadowBias = 1;
    float fogDensity = 0;
	bool showUI = true;
	float swirlPeriod = 100;
	bool enableSwirl = true;
	bool fullscreen = false;
	Vector3 transformRotation = { 0, 0, 0 };
	float transformScale = 1;
    double time = 0;
	uint32 frameIndex = 0;
    std::unique_ptr<DAGTracer> tracer;
    ReplayManager replayWriter;

	glf::Context* fontctx = nullptr;
	glf::Buffer* dynamicText = nullptr;
	glf::Buffer* staticText = nullptr;

	struct Timings
	{
		double pathsTime = 0;
		double colorsTime = 0;
		double shadowsTime = 0;
		double totalTime = 0;
	};
	Timings timings;

	uint32 lastEditTimestamp = 0;
	uint32 lastEditFrame = 0;

    bool is_dag_valid(EDag dag) const;
    void next_dag();
    void previous_dag();

    void key_callback_impl(int key, int scancode, int action, int mods);
    void mouse_callback_impl(int button, int action, int mods);
    void scroll_callback_impl(double xoffset, double yoffset);

    static void key_callback(GLFWwindow*, int key, int scancode, int action, int mods)
    {
        Engine::engine.key_callback_impl(key, scancode, action, mods);
    }
    static void mouse_callback(GLFWwindow*, int button, int action, int mods)
    {
        Engine::engine.mouse_callback_impl(button, action, mods);
    }
    static void scroll_callback(GLFWwindow*, double xoffset, double yoffset)
    {
        Engine::engine.scroll_callback_impl(xoffset, yoffset);
    }

    void tick();

    void init_graphics();

    void loop_headless();
    void loop_graphics();
};
