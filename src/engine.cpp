#include "engine.h"
#include "hacky_profiler.hpp"
#include "shader.h"
#include "utils.h"
#include "memory.h"
#include "dags/hash_dag/hash_dag_editors.h"

#include "glfont.h"

Engine Engine::engine;

inline void clear_console()
{
    printf("\033[H\033[J");
}

std::string dag_to_string(EDag dag)
{
	switch (dag)
	{
	case EDag::BasicDagUncompressedColors:
		return "BasicDagUncompressedColors";
	case EDag::BasicDagCompressedColors:
		return "BasicDagCompressedColors";
	case EDag::BasicDagColorErrors:
		return "BasicDagColorErrors";
	case EDag::HashDag:
		return "HashDag";
	default:
		check(false);
		return "";
	}
}

std::string tool_to_string(ETool tool)
{
	switch (tool)
	{
	case ETool::Sphere:
		return "Sphere";
	case ETool::SpherePaint:
		return "SpherePaint";;
	case ETool::SphereNoise:
		return "SphereNoise";
	case ETool::Cube:
		return "Cube";
	case ETool::CubeCopy:
		return "CubeCopy";
	case ETool::CubeFill:
		return "CubeFill";
	default:
		check(false);
		return "";
	}
}

bool Engine::is_dag_valid(EDag dag) const
{
	switch (dag)
	{
	case EDag::BasicDagUncompressedColors:
		return basicDag.is_valid() && basicDagUncompressedColors.is_valid();
	case EDag::BasicDagCompressedColors:
		return basicDag.is_valid() && basicDagCompressedColors.is_valid();
	case EDag::BasicDagColorErrors:
		return basicDag.is_valid() && basicDagColorErrors.is_valid();
	case EDag::HashDag:
		return hashDag.is_valid() && hashDagColors.is_valid();
	default:
		check(false);
		return false;
	}
}

void Engine::next_dag()
{
	do
	{
		config.currentDag = EDag((uint32(config.currentDag) + 1) % CNumDags);
	}
	while (!is_dag_valid(config.currentDag));
}

void Engine::previous_dag()
{
	do
	{
		config.currentDag = EDag(Utils::subtract_mod(uint32(config.currentDag), CNumDags));
	}
	while (!is_dag_valid(config.currentDag));
}

void Engine::set_dag(EDag dag)
{
	config.currentDag = dag;
	if (!is_dag_valid(config.currentDag))
	{
		next_dag();
	}
}

void Engine::key_callback_impl(int key, int scancode, int action, int mods)
{
	if (!((0 <= key) && (key <= GLFW_KEY_LAST))) // Media keys
	{
		return;
	}

	if (action == GLFW_RELEASE)
		state.keys[(uint64)key] = false;
	if (action == GLFW_PRESS)
		state.keys[(uint64)key] = true;

	if (action == GLFW_PRESS || action == GLFW_REPEAT)
	{
		if (key == GLFW_KEY_M)
		{
			printMemoryStats = !printMemoryStats;
		}
#if UNDO_REDO
		if (key == GLFW_KEY_Z)
		{
			if (config.currentDag == EDag::HashDag)
			{
				if (state.keys[GLFW_KEY_LEFT_CONTROL] || state.keys[GLFW_KEY_RIGHT_CONTROL])
				{
					if (state.keys[GLFW_KEY_LEFT_SHIFT])
					{
						undoRedo.redo(hashDag, hashDagColors);
						replayWriter.add_action<ReplayActionRedo>();
					}
					else
					{
						undoRedo.undo(hashDag, hashDagColors);
						replayWriter.add_action<ReplayActionUndo>();
					}
				}
			}
		}
#endif
		if (key == GLFW_KEY_BACKSPACE)
		{
			replayWriter.write_csv();
			replayWriter.clear();
			printf("Replay saved!\n");
		}
		if (key == GLFW_KEY_R)
		{
			if (state.keys[GLFW_KEY_LEFT_SHIFT])
			{
				printf("Replay reader cleared\n");
				printf("Replay writer cleared\n");
				replayReader.clear();
				replayWriter.clear();
			}
			else
			{
				printf("Replay reader reset\n");
				printf("Stats cleared\n");
				statsRecorder.clear();
				replayReader.reset_replay();
			}
		}
		if (key == GLFW_KEY_TAB)
		{
			if (state.keys[GLFW_KEY_LEFT_SHIFT])
			{
				config.tool = ETool(Utils::subtract_mod(uint32(config.tool), CNumTools));
			}
			else
			{
				config.tool = ETool((uint32(config.tool) + 1) % CNumTools);
			}

			const auto str = tool_to_string(config.tool);
			printf("Current tool: %s\n", str.c_str());
		}
		if (key == GLFW_KEY_G)
		{
			if (config.currentDag == EDag::HashDag)
			{
				hashDag.remove_stale_nodes(hashDag.levels - 2);
			}
			undoRedo.free();
		}
		if (key == GLFW_KEY_C)
		{
			if (state.keys[GLFW_KEY_LEFT_SHIFT])
			{
				config.debugColors = EDebugColors(Utils::subtract_mod(uint32(config.debugColors), CNumDebugColors));
			}
			else
			{
				config.debugColors = EDebugColors((uint32(config.debugColors) + 1) % CNumDebugColors);
			}
		}
		if (key == GLFW_KEY_U)
		{
            auto previousGPUUsage = Memory::get_gpu_allocated_memory();
            auto previousCPUUsage = Memory::get_cpu_allocated_memory();
            undoRedo.free();
            printf("Undo redo cleared! Memory saved: GPU: %fMB CPU: %fMB\n",
                   Utils::to_MB(previousGPUUsage - Memory::get_gpu_allocated_memory()),
                   Utils::to_MB(previousCPUUsage - Memory::get_cpu_allocated_memory()));
        }
        if (key == GLFW_KEY_CAPS_LOCK)
		{
			if (state.keys[GLFW_KEY_LEFT_SHIFT])
			{
				previous_dag();
			}
			else
			{
				next_dag();
			}
			const auto str = dag_to_string(config.currentDag);
			printf("Current dag: %s\n", str.c_str());
		}
		if (key == GLFW_KEY_1)
		{
			config.debugColorsIndexLevel++;
			config.debugColorsIndexLevel = std::min(config.debugColorsIndexLevel, basicDag.levels);
		}
		if (key == GLFW_KEY_2)
		{
			config.debugColorsIndexLevel = uint32(std::max(int32(config.debugColorsIndexLevel) - 1, 0));
		}
		if (key == GLFW_KEY_3)
		{
			config.debugColors = EDebugColors::Index;
		}
		if (key == GLFW_KEY_4)
		{
			config.debugColors = EDebugColors::Position;
		}
		if (key == GLFW_KEY_5)
		{
			config.debugColors = EDebugColors::ColorTree;
		}
		if (key == GLFW_KEY_6)
		{
			config.debugColors = EDebugColors::ColorBits;
		}
		if (key == GLFW_KEY_7)
		{
			config.debugColors = EDebugColors::MinColor;
		}
		if (key == GLFW_KEY_8)
		{
			config.debugColors = EDebugColors::MaxColor;
		}
		if (key == GLFW_KEY_9)
		{
			config.debugColors = EDebugColors::Weight;
		}
		if (key == GLFW_KEY_0)
		{
			config.debugColors = EDebugColors::None;
		}
		if (key == GLFW_KEY_X)
		{
			shadows = !shadows;
        }
        if (key == GLFW_KEY_EQUAL)
        {
            shadowBias += 0.1f;
            printf("Shadow bias: %f\n", shadowBias);
        }
        if (key == GLFW_KEY_MINUS)
        {
            shadowBias -= 0.1f;
            printf("Shadow bias: %f\n", shadowBias);
        }
        if (key == GLFW_KEY_O)
        {
            fogDensity += 1;
            printf("Fog density: %f\n", fogDensity);
        }
        if (key == GLFW_KEY_H)
        {
        	showUI = !showUI;
        }
        if (key == GLFW_KEY_F)
        {
        	toggle_fullscreen();
        }

		const double rotationStep = (state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT] ? -10 : 10);
		if (key == GLFW_KEY_F1)
		{
			transformRotation.X += rotationStep;
			if (transformRotation.X > 180) transformRotation.X -= 360;
			if (transformRotation.X < -180) transformRotation.X += 360;
		}
		if (key == GLFW_KEY_F2)
		{
			transformRotation.Y += rotationStep;
			if (transformRotation.Y > 180) transformRotation.Y -= 360;
			if (transformRotation.Y < -180) transformRotation.Y += 360;
		}
		if (key == GLFW_KEY_F3)
		{
			transformRotation.Z += rotationStep;
			if (transformRotation.Z > 180) transformRotation.Z -= 360;
			if (transformRotation.Z < -180) transformRotation.Z += 360;
		}
		if (key == GLFW_KEY_F6)
		{
			transformScale += state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT] ? -.1f : .1f;
		}
		
		if (key == GLFW_KEY_F4)
		{
			enableSwirl = !enableSwirl;
		}
		if (key == GLFW_KEY_F5)
		{
			swirlPeriod += state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT] ? -10 : 10;
		}
		
        if (key == GLFW_KEY_I)
        {
            fogDensity -= 1;
            printf("Fog density: %f\n", fogDensity);
        }
		if (key == GLFW_KEY_P)
		{
			const bool printGlobalStats = state.keys[GLFW_KEY_LEFT_SHIFT];
			if (config.currentDag == EDag::BasicDagUncompressedColors)
			{
				if (printGlobalStats) DAGUtils::print_stats(basicDag);
				basicDag.print_stats();
				basicDagUncompressedColors.print_stats();
			}
			else if (config.currentDag == EDag::BasicDagCompressedColors)
			{
				if (printGlobalStats) DAGUtils::print_stats(basicDag);
				basicDag.print_stats();
				basicDagCompressedColors.print_stats();
			}
			else if (config.currentDag == EDag::BasicDagColorErrors)
			{
				if (printGlobalStats) DAGUtils::print_stats(basicDag);
				basicDag.print_stats();
			}
			else if (config.currentDag == EDag::HashDag)
			{
				if (printGlobalStats) DAGUtils::print_stats(hashDag);
				hashDag.data.print_stats();
				hashDagColors.print_stats();
#if UNDO_REDO
				undoRedo.print_stats();
#endif
			}
		}
		if (key == GLFW_KEY_L)
		{
            hashDag.data.save_bucket_sizes(false);
		}
		if (key == GLFW_KEY_KP_ENTER)
		{
			printf("view.rotation = { %f, %f, %f, %f, %f, %f, %f, %f, %f };\n",
			       view.rotation.D00,
			       view.rotation.D01,
			       view.rotation.D02,
			       view.rotation.D10,
			       view.rotation.D11,
			       view.rotation.D12,
			       view.rotation.D20,
			       view.rotation.D21,
			       view.rotation.D22);
			printf("view.position = { %f, %f, %f };\n",
			       view.position.X,
			       view.position.Y,
			       view.position.Z);
		}
	}
}

void Engine::mouse_callback_impl(int button, int action, int mods)
{
	if (button != GLFW_MOUSE_BUTTON_LEFT && button != GLFW_MOUSE_BUTTON_RIGHT)
		return;

	if (action == GLFW_RELEASE)
	{
		state.mouse[(uint64)button] = false;
	}
	else if (action == GLFW_PRESS)
	{
		state.mouse[(uint64)button] = true;
	}
}

void Engine::scroll_callback_impl(double xoffset, double yoffset)
{
	config.radius += float(yoffset) * (state.keys[GLFW_KEY_LEFT_SHIFT] ? 10.f : 1.f);
	config.radius = std::max(config.radius, 0.f);
}

void Engine::tick()
{
	PROFILE_FUNCTION();

	frameIndex++;
	
	if (printMemoryStats)
	{
		clear_console();
		std::cout << Memory::get_stats_string();
	}

	videoManager.tick(*this);

	// Controls
    if (replayReader.is_empty())
    {
        if (state.keys[GLFW_KEY_KP_0])
        {
            targetView.rotation = { -0.573465, 0.000000, -0.819230, -0.034067, 0.999135, 0.023847, 0.818522, 0.041585, -0.572969 };
            targetView.position = { -13076.174715, -1671.669438, 5849.331627 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_1])
        {
            targetView.rotation = { 0.615306, -0.000000, -0.788288, -0.022851, 0.999580, -0.017837, 0.787957, 0.028989, 0.615048 };
            targetView.position = { -7736.138941, -2552.420373, -5340.566371 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_2])
        {
            targetView.rotation = { -0.236573, -0.000000, -0.971614, 0.025623, 0.999652, -0.006239, 0.971276, -0.026372, -0.236491 };
            targetView.position = { -2954.821641, 191.883613, 4200.793442 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_3])
        {
            targetView.rotation = { 0.590287, -0.000000, -0.807193, 0.150128, 0.982552, 0.109786, 0.793109, -0.185987, 0.579988 };
            targetView.position = { -7036.452685, -3990.109906, 7964.129876 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_4])
        {
            targetView.rotation = { -0.222343, -0.000000, -0.974968, 0.070352, 0.997393, -0.016044, 0.972427, -0.072159, -0.221764 };
            targetView.position = { 762.379376, -935.456405, -358.642203 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_5])
        {
            targetView.rotation = { 0, 0, -1, 0, 1, 0, 1, 0, 0 };
            targetView.position = { -951.243605, 667.199855, -27.706481 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_6])
        {
            targetView.rotation = { 0.095015, -0.000000, -0.995476, 0.130796, 0.991331, 0.012484, 0.986846, -0.131390, 0.094192 };
            targetView.position = { 652.972238, 73.188250, -209.028828 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_7])
        {
            targetView.rotation = { -0.004716, -0.000000, -0.999989, 0.583523, 0.812093, -0.002752, 0.812084, -0.583529, -0.003830 };
            targetView.position = { -1261.247484, 1834.904220, -11.976059 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_8])
        {
            targetView.rotation = { 0.019229, -0.000000, 0.999815, -0.040020, 0.999198, 0.000770, -0.999014, -0.040027, 0.019213 };
            targetView.position = { -8998.476232, -2530.419704, -4905.593975 };
            init_target_lerp();
        }

		if (state.keys[GLFW_KEY_KP_ADD])
		    config.radius++;

        double speed = length(make_double3(dagInfo.boundsAABBMax - dagInfo.boundsAABBMin)) / 100 * dt;
		double rotationSpeed = 2 * dt;

		if (state.keys[GLFW_KEY_LEFT_SHIFT])
			speed *= 10;

		if (state.keys[GLFW_KEY_W])
        {
            view.position += speed * view.forward();
            moveToTarget = false;
        }
		if (state.keys[GLFW_KEY_S])
        {
			view.position -= speed * view.forward();
            moveToTarget = false;
        }
		if (state.keys[GLFW_KEY_D])
        {
			view.position += speed * view.right();
            moveToTarget = false;
        }
		if (state.keys[GLFW_KEY_A])
        {
			view.position -= speed * view.right();
            moveToTarget = false;
        }
		if (state.keys[GLFW_KEY_SPACE])
        {
            view.position += speed * view.up();
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_LEFT_CONTROL])
        {
            view.position -= speed * view.up();
            moveToTarget = false;
        }

        if (state.keys[GLFW_KEY_RIGHT] || state.keys[GLFW_KEY_E])
        {
            view.rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(rotationSpeed, Vector3::Up()));
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_LEFT] || state.keys[GLFW_KEY_Q])
        {
            view.rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(-rotationSpeed, Vector3::Up()));
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_DOWN])
        {
            view.rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(rotationSpeed, view.right()));
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_UP])
        {
            view.rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(-rotationSpeed, view.right()));
            moveToTarget = false;
        }
	}

    if (moveToTarget)
    {
        targetLerpTime = clamp(targetLerpTime + targetLerpSpeed * dt, 0., 1.);
        view.position = lerp(initialView.position, targetView.position, targetLerpTime);
        view.rotation = Matrix3x3::FromQuaternion(Quaternion::Slerp(Matrix3x3::ToQuaternion(initialView.rotation), Matrix3x3::ToQuaternion(targetView.rotation), targetLerpTime));
    }

	if (replayReader.is_empty())
	{
		// Save position/rotation
		replayWriter.add_action<ReplayActionSetLocation>(view.position);
		replayWriter.add_action<ReplayActionSetRotation>(view.rotation);
	}
	else if (!replayReader.at_end())
	{
		replayReader.replay_frame();
		if (replayReader.at_end())
		{
            if (firstReplay && REPLAY_TWICE)
            {
                printf("First replay ended, starting again now that everything is loaded in memory...\n");
                firstReplay = false;
                replayReader.reset_replay();
                statsRecorder.clear();
            }
            else
            {
#if BENCHMARK
                printf("Replay ended, saving stats... ");
                statsRecorder.write_csv();
#endif
#ifdef PROFILING_PATH
                hashDag.data.save_bucket_sizes(false);
#endif
                statsRecorder.clear();
                printf("Saved!\n");
            }
        }
	}

	double pathsTime = 0;
	switch (config.currentDag)
	{
	case EDag::BasicDagUncompressedColors:
	case EDag::BasicDagCompressedColors:
	case EDag::BasicDagColorErrors:
		pathsTime = tracer->resolve_paths(view, dagInfo, basicDag);
		break;
	case EDag::HashDag:
		pathsTime = tracer->resolve_paths(view, dagInfo, hashDag);
		break;
	}
	statsRecorder.report("paths", pathsTime);

	constexpr double xMultiplier = double(imageWidth) / windowWidth;
	constexpr double yMultiplier = double(imageHeight) / windowHeight;

	const uint32 posX = uint32(clamp<int32>(int32(xMultiplier * state.mousePosX), 0, imageWidth - 1));
	const uint32 posY = uint32(clamp<int32>(int32(yMultiplier * state.mousePosY), 0, imageHeight - 1));

    if (replayReader.is_empty())
    {
        config.path = tracer->get_path(posX, posY);
#if RECORD_TOOL_OVERLAY
        replayWriter.add_action<ReplayActionSetToolParameters>(config.path, config.copySourcePath, config.copyDestPath, config.radius, uint32(config.tool));
#endif
    }

    double colorsTime = 0;
	const uint32 debugColorsIndexLevel = basicDag.levels - 2 - config.debugColorsIndexLevel;
	const ToolInfo toolInfo
	{
		config.tool,
		config.path,
		config.radius,
		config.copySourcePath,
		config.copyDestPath
	};
	switch (config.currentDag)
	{
	case EDag::BasicDagUncompressedColors:
		colorsTime = tracer->resolve_colors(basicDag, basicDagUncompressedColors, config.debugColors,
		                                    debugColorsIndexLevel, toolInfo);
		break;
	case EDag::BasicDagCompressedColors:
		colorsTime = tracer->resolve_colors(basicDag, basicDagCompressedColors, config.debugColors, debugColorsIndexLevel,
		                                    toolInfo);
		break;
	case EDag::BasicDagColorErrors:
		colorsTime = tracer->resolve_colors(basicDag, basicDagColorErrors, config.debugColors,
		                                    debugColorsIndexLevel, toolInfo);
		break;
	case EDag::HashDag:
		colorsTime = tracer->resolve_colors(hashDag, hashDagColors, config.debugColors, debugColorsIndexLevel, toolInfo);
		break;
	}
	statsRecorder.report("colors", colorsTime);

	double shadowsTime = 0;
    if (shadows && ENABLE_SHADOWS)
    {
        switch (config.currentDag)
        {
            case EDag::BasicDagUncompressedColors:
            case EDag::BasicDagCompressedColors:
            case EDag::BasicDagColorErrors:
                shadowsTime = tracer->resolve_shadows(view, dagInfo, basicDag, shadowBias, fogDensity);
                break;
            case EDag::HashDag:
                shadowsTime = tracer->resolve_shadows(view, dagInfo, hashDag, shadowBias, fogDensity);
                break;
        }
    }
	statsRecorder.report("shadows", shadowsTime);

    if (config.currentDag == EDag::HashDag && replayReader.is_empty())
    {
        if (state.mouse[GLFW_MOUSE_BUTTON_LEFT] || state.mouse[GLFW_MOUSE_BUTTON_RIGHT])
        {
            if (config.tool == ETool::CubeCopy && state.mouse[GLFW_MOUSE_BUTTON_RIGHT])
            {
                if (state.keys[GLFW_KEY_LEFT_SHIFT])
                {
                    config.copySourcePath = config.path;
                }
                else
                {
                    config.copyDestPath = config.path;
                }
            }

            const bool isAdding = state.mouse[GLFW_MOUSE_BUTTON_RIGHT];
			const float3 position = make_float3(config.path);

			if (config.tool == ETool::Sphere)
			{
                if (isAdding)
                {
                    edit<SphereEditor<true>>(position, config.radius);
                }
                else
                {
                    edit<SphereEditor<false>>(position, config.radius);
                }
				replayWriter.add_action<ReplayActionSphere>(position, config.radius, isAdding);
			}
			else if (config.tool == ETool::SpherePaint)
			{
				edit<SpherePaintEditor>(position, config.radius);
				replayWriter.add_action<ReplayActionPaint>(position, config.radius);
			}
			else if (config.tool == ETool::SphereNoise)
			{
				edit<SphereNoiseEditor>(hashDag, position, config.radius, isAdding);
			}
			else if (config.tool == ETool::Cube)
			{
                if (isAdding)
                {
                    edit<BoxEditor<true>>(position, config.radius);
                }
                else
                {
                    edit<BoxEditor<false>>(position, config.radius);
                }
				replayWriter.add_action<ReplayActionCube>(position, config.radius, isAdding);
			}
			else if (config.tool == ETool::CubeCopy)
			{
				if (!isAdding && config.radius >= 1)
				{
                    const float3 src = make_float3(config.copySourcePath);
                    const float3 dest = make_float3(config.copyDestPath);
                    const Matrix3x3 transform = Matrix3x3::FromQuaternion(Quaternion::FromEuler(transformRotation / 180 * M_PI)) * transformScale;
                    edit<CopyEditor>(hashDag, hashDagColors, src, dest, position, config.radius, transform, statsRecorder, enableSwirl, swirlPeriod);
                    replayWriter.add_action<ReplayActionCopy>(position, src, dest, config.radius, transform, enableSwirl, swirlPeriod);
                }
			}
			else if (config.tool == ETool::CubeFill)
			{
			    const float3 center = position + (isAdding ? -1.f : 1.f) * round(2.f * make_float3(view.forward()));
                edit<FillEditorColors>(hashDag, hashDagColors, center, config.radius);
                replayWriter.add_action<ReplayActionFill>(center, config.radius);
            }
		}
	}

	auto currentTime = Utils::seconds();

	timings.pathsTime = pathsTime;
	timings.colorsTime = colorsTime;
	timings.shadowsTime = shadowsTime;
	timings.totalTime = (currentTime - time) * 1e3;

	dt = currentTime - time;
	time = currentTime;

	if (replayReader.is_empty())
	{
		replayWriter.add_action<ReplayActionEndFrame>();
	}
	else
	{
        if (hashDag.data.is_valid())
        {
            statsRecorder.report("virtual_size", hashDag.data.get_virtual_used_size(false));
            statsRecorder.report("allocated_size", hashDag.data.get_allocated_pages_size());
            statsRecorder.report("color_size", hashDagColors.get_total_used_memory());
            statsRecorder.report("color_size undo_redo", undoRedo.get_total_used_memory());
#if SIMULATE_GC
            hashDag.simulate_remove_stale_nodes(statsRecorder);
#endif
        }
		statsRecorder.next_frame();
	}

	HACK_PROFILE_FRAME_ADVANCE();
}

void Engine::init(bool inheadLess)
{
	PROFILE_FUNCTION();
	
    headLess = inheadLess;

    if (!headLess)
    {
        init_graphics();
    }

	tracer = std::make_unique<DAGTracer>(headLess);
	image = tracer->get_colors_image();
	time = Utils::seconds();
}

void Engine::init_graphics()
{
	PROFILE_FUNCTION();
	
	// Initialize GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		exit(1);
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
	// Open a window and create its OpenGL context
	window = glfwCreateWindow(windowWidth, windowHeight, "DAG Edits", NULL, NULL);
	
	if (window == NULL)
	{
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		exit(1);
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glfwSwapInterval(0);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize GLEW\n");
		exit(1);
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Load fonts
	fontctx = glf::make_context(
		"assets/droid-sans-mono/DroidSansMonoDotted.ttf",
		windowWidth, windowHeight,
		"assets/noto-emoji/NotoEmoji-Regular.ttf"
	);

	dynamicText = glf::make_buffer();
	staticText = glf::make_buffer();

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	glGenVertexArrays(1, &fsvao);
	glBindVertexArray(fsvao);

	// Create and compile our GLSL program from the shaders
	programID = LoadShaders("src/shaders/TransformVertexShader.glsl", "src/shaders/TextureFragmentShader.glsl");

	// Get a handle for our "myTextureSampler" uniform
	textureID = glGetUniformLocation(programID, "myTextureSampler");

	// Our vertices. Tree consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
	// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
	static const GLfloat g_vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
	};

	// Two UV coordinates for each vertex. They were created with Blender.
	static const GLfloat g_uv_buffer_data[] = {
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,
	};

	GLuint vertexBuffer = 0;
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	GLuint uvBuffer = 0;
	glGenBuffers(1, &uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glVertexAttribPointer(
		0, // attribute. No particular reason for 0, but must match the layout in the shader.
		3, // size
		GL_FLOAT, // type
		GL_FALSE, // normalized?
		0, // stride
		(void*)0 // array buffer offset
	);

	// 2nd attribute buffer : UVs
	 
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glVertexAttribPointer(
		1, // attribute. No particular reason for 1, but must match the layout in the shader.
		2, // size : U+V => 2
		GL_FLOAT, // type
		GL_FALSE, // normalized?
		0, // stride
		(void*)0 // array buffer offset
	);

	glBindVertexArray( 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	glDeleteBuffers( 1, &vertexBuffer );
	glDeleteBuffers( 1, &uvBuffer );
}

void Engine::loop()
{
    if (headLess)
    {
        loop_headless();
    }
    else
    {
        loop_graphics();
    }
}

void Engine::loop_headless()
{
	PROFILE_FUNCTION();
	
    while (!replayReader.at_end())
    {
		MARK_FRAME();
        tick();
    }
}
void Engine::loop_graphics()
{
	PROFILE_FUNCTION();
	
	do
	{
		MARK_FRAME();
		
        glfwGetCursorPos(window, &state.mousePosX, &state.mousePosY);

		tick();

        glfwSetWindowTitle(window, "HashDag");

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use our shader
		glUseProgram(programID);

		// Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		//glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		// Bind our texture in Texture Unit 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, image);

		// Set our "myTextureSampler" sampler to use Texture Unit 0
		glUniform1i(textureID, 0);

		// Draw the triangle !
		glBindVertexArray( fsvao );
		glDrawArrays(GL_TRIANGLES, 0, 12 * 3); // 12*3 indices starting at 0 -> 12 triangles
		
		glBindVertexArray( 0 );
		glUseProgram( 0 );

		// 2D stuff
		glDisable( GL_DEPTH_TEST );
		glEnable( GL_BLEND );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

#if 0
		// <TESTING>
		glf::add_line( fontctx, dynamicText, 100.f, 1080-50.f, "Testing, testing!" );
		glf::add_line( fontctx, dynamicText, glf::EFmt::glow, 100.f, 1080-75.f, "Testing, testing!" );
		glf::add_line( fontctx, dynamicText, 100.f, 1080-100.f, "💩" );
		glf::add_line( fontctx, dynamicText, glf::EFmt::glow, 150.f, 1080-100.f, "💩" );
		glf::add_line( fontctx, dynamicText, glf::EFmt::glow, 100.f, 1080-125.f, "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσ/ςΤτΥυΦφΧχΨψΩω" );
		// </TESTING>
#endif

		if (showUI)
		{
			using glf::EFmt;
			
			auto const time_ = [&] (auto&& T) {
				std::ostringstream oss;
				oss << std::fixed << std::setw(6) << std::setprecision(2) << T << "ms";
				return oss.str();
			};
			auto const mb_ = [&] (auto&& T) {
				std::ostringstream oss;
				oss << std::fixed << std::setw(6)  << std::setprecision(1) << T << "MB";
				return oss.str();
			};
			auto const cmb_ = [&] (auto&& T, auto&& U) {
				std::ostringstream oss;
				oss << std::fixed << std::setw(6)  << std::setprecision(1) << T << "";
				oss << " (" << std::fixed << std::setw(6)  << std::setprecision(1) << U << "MB)";
				return oss.str();
			};
			auto const mbx_ = [&] (auto&& T, auto&& U) {
				std::ostringstream oss;
				oss << std::fixed << std::setw(6)  << std::setprecision(1) << T << "MB";
				oss << " (+" << std::fixed << std::setw(6)  << std::setprecision(1) << U << "MB)";
				return oss.str();
			};
			auto const mb2_ = [&] (auto&& T, auto&& U) {
				std::ostringstream oss;
				oss << std::fixed << std::setw(6)  << std::setprecision(1) << T << "MB";
				oss << " / " << std::fixed << std::setw(6)  << std::setprecision(1) << U << "MB";
				return oss.str();
			};
			auto const vector3_ = [&] (const Vector3& V) {
				std::ostringstream oss;
				oss << std::fixed << std::setprecision(1) << V.X << ", " << V.Y << ", " << V.Z;
				return oss.str();
			};

			auto const count_ = [&] (auto&& T) {
				std::ostringstream oss;
				oss << std::scientific << std::setw(4) << std::setprecision(3) << T;
				return oss.str();
			};

			auto const draw = [&] (auto&& F) {
				float y = windowHeight - 42.f;
				float const hx = 75.f;
				float const sx = 120.f;
				float const dx = 305.f;

#				define STRINGIFY0_(x) #x
#				define STRINGIFY_(x) STRINGIFY0_(x)
				static constexpr char scene[] = "Scene " STRINGIFY_(SCENE) " (2^" STRINGIFY_(SCENE_DEPTH) ") using ";
				F( hx, y, EFmt::glow, scene, dx+sizeof(scene)*6, dag_to_string(config.currentDag) ); y -= 24.f; 
#				undef STRINGIFY_
#				undef STRINGIFY0_
				F( hx, y, EFmt::glow, "Active tool:", dx, tool_to_string(config.tool) ); y -= 24.f; 
				if (config.tool == ETool::CubeCopy)
				{
#if COPY_APPLY_TRANSFORM
					F( hx, y, EFmt::glow, "Rotation:", dx, vector3_(transformRotation) ); y -= 24.f;
					F( hx, y, EFmt::glow, "Scale:", dx, std::to_string(transformScale) ); y -= 24.f;
#endif
#if COPY_CAN_APPLY_SWIRL 
					F( hx, y, EFmt::glow, "Swirl:", dx, enableSwirl ? "ON" : "OFF" ); y -= 24.f;
					F( hx, y, EFmt::glow, "Swirl period:", dx, std::to_string(swirlPeriod) ); y -= 24.f;
#endif
				}
				y-= 32.f;

				const double editingAndUploading =
					lastEditFrame == frameIndex ?
					statsRecorder.get_value_in_frame(lastEditTimestamp, "total edits") +
					statsRecorder.get_value_in_frame(lastEditTimestamp, "upload_to_gpu") +
					statsRecorder.get_value_in_frame(lastEditTimestamp, "creating edit tool") :
					0;

				F( hx, y, EFmt::large, "Timings", -1.f, nullptr ); y -= 32.f;
				F( sx, y, EFmt::glow, "Trace primary", dx, time_(timings.pathsTime) ); y -= 24.f; 
				F( sx, y, EFmt::glow, "Trace shadow", dx, time_(timings.shadowsTime) ); y -= 24.f; 
				F( sx, y, EFmt::glow, "Resolve colors", dx, time_(timings.colorsTime) ); y -= 24.f; 
				F( sx, y, EFmt::glow, "Edit & Upload", dx, time_(editingAndUploading) ); y -= 24.f; 
				F( sx, y, EFmt::glow, "Total", dx, time_(timings.totalTime) ); y -= 24.f; 
				y-= 24.f;

				F( hx, y, EFmt::large, "Memory", -1.f, nullptr ); y -= 32.f;
				F( sx, y, EFmt::glow, "Page pool", dx, cmb_(hashDag.data.get_total_pages(), hashDag.data.get_pool_size()) ); y -= 24.f; 
				F( sx, y, EFmt::glow, "  used", dx, cmb_(hashDag.data.get_allocated_pages(), hashDag.data.get_allocated_pages_size()) ); y -= 24.f; 
				F( sx, y, EFmt::glow, "Page table", dx, mb_(hashDag.data.get_page_table_size()) ); y -= 24.f; 

#				if USE_VIDEO
				y -= 32.f-24.f;
				F( hx, y, EFmt::glow, "  Total (GPU/CPU)", dx, mb2_(Utils::to_MB(Memory::get_gpu_allocated_memory()),Utils::to_MB(Memory::get_cpu_allocated_memory()+Memory::get_cxx_cpu_allocated_memory())) ); y -= 24.f; 
#				else // !USE_VIDEO
				F( sx, y, EFmt::glow, "Total GPU", dx, mb_(Utils::to_MB(Memory::get_gpu_allocated_memory())) ); y -= 24.f; 
				F( sx, y, EFmt::glow, "Total CPU", dx, mbx_(Utils::to_MB(Memory::get_cpu_allocated_memory()),Utils::to_MB(Memory::get_cxx_cpu_allocated_memory())) ); y -= 24.f; 
#				endif // ~ USE_VIDEO
				y-= 24.f;

				F( hx, y, EFmt::large, "Edits", -1.f, nullptr ); y -= 32.f;
				F( sx, y, EFmt::glow, "Num Voxels", dx, count_(statsRecorder.get_value_in_frame(lastEditTimestamp, "num voxels")) ); y -= 24.f;
				F( sx, y, EFmt::glow, "Num Nodes", dx, count_(statsRecorder.get_value_in_frame(lastEditTimestamp, "num nodes")) ); y -= 24.f;
				y-= 24.f;
			};

			/*static bool const initStatic = [&] {
				draw( [&] (float aX, float aY, EFmt aFmt, auto&& aTxt, float, auto&&)
					{
						if( aX > 0.0f )
							glf::add_line( fontctx, staticText, aFmt, aX, aY, aTxt );
					}
				);
				return true;
			}();*/

			draw( [&] (float aX1, float aY, EFmt aFmt, auto&& aTxt1, float aX2, auto&& aTxt2)
				{
					if( aX1 > 0.0f )
						glf::add_line( fontctx, dynamicText, aFmt, aX1, aY, aTxt1 );
					if( aX2 > 0.0f )
						glf::add_line( fontctx, dynamicText, aFmt, aX2, aY, aTxt2 );
				}
			);
        }
		
		
		glf::draw_buffer( fontctx, staticText );
		glf::draw_buffer( fontctx, dynamicText );

		glf::clear_buffer( dynamicText );

		glDisable( GL_BLEND );
		glEnable( GL_DEPTH_TEST );

		// Swap buffers
		glfwSwapBuffers(window);
        glfwPollEvents();
    } // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS
           && glfwWindowShouldClose(window) == 0
#ifdef EXIT_AFTER_REPLAY
           && !replayReader.at_end()
#endif
            );
}

void Engine::destroy()
{
	glf::destroy_buffer( staticText );
	glf::destroy_buffer( dynamicText );
	glf::destroy_context( fontctx );

	glDeleteVertexArrays( 1, &fsvao );
	
	tracer.reset();
	basicDag.free();
	basicDagCompressedColors.free();
	basicDagUncompressedColors.free();
	basicDagColorErrors.free();
	hashDag.free();
	hashDagColors.free();
	undoRedo.free();
}

void Engine::toggle_fullscreen()
{
	if (!fullscreen)
	{
		fullscreen = true;
		GLFWmonitor* primary = glfwGetPrimaryMonitor();
		const GLFWvidmode* mode = glfwGetVideoMode(primary);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, mode->width, mode->height, mode->refreshRate);
	}
	else
	{
		fullscreen = false;
		glfwSetWindowMonitor(window, NULL, 0, 0, windowWidth, windowHeight, -1);
	}
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
}
