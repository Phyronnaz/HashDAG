#include "replay.h"
#include "engine.h"
#include "dags/hash_dag/hash_dag_editors.h"

constexpr float scalingFactor = float(1 << SCENE_DEPTH) / float(1 << REPLAY_DEPTH);

void ReplayActionUndo::apply()
{
#if UNDO_REDO
    Engine::engine.undoRedo.undo(Engine::engine.hashDag, Engine::engine.hashDagColors);
#endif
}

void ReplayActionRedo::apply()
{
#if UNDO_REDO
    Engine::engine.undoRedo.redo(Engine::engine.hashDag, Engine::engine.hashDagColors);
#endif
}

void ReplayActionSetLocation::apply()
{
	Engine::engine.view.position = location;
}

void ReplayActionSetRotation::apply()
{
	Engine::engine.view.rotation = rotation;
}

void ReplayActionSetToolParameters::apply()
{
    Engine::engine.config.path = truncate(make_float3(path) * scalingFactor);
    Engine::engine.config.copySourcePath = truncate(make_float3(copySourcePath) * scalingFactor);
    Engine::engine.config.copyDestPath = truncate(make_float3(copyDestPath) * scalingFactor);
    Engine::engine.config.radius = radius * scalingFactor;
    Engine::engine.config.tool = ETool(tool);
}

void ReplayActionSphere::apply()
{
    if (add)
    {
        Engine::engine.edit<SphereEditor<true>>(
                location * scalingFactor,
                radius * scalingFactor);
    }
    else
    {
        Engine::engine.edit<SphereEditor<false>>(
                location * scalingFactor,
                radius * scalingFactor);
    }
}

void ReplayActionCube::apply()
{
    if (add)
    {
        Engine::engine.edit<BoxEditor<true>>(
                location * scalingFactor,
                radius * scalingFactor);
    }
    else
    {
        Engine::engine.edit<BoxEditor<false>>(
                location * scalingFactor,
                radius * scalingFactor);
    }
}

void ReplayActionCopy::apply()
{
	Engine::engine.edit<CopyEditor>(
		Engine::engine.hashDag,
		Engine::engine.hashDagColors,
		src * scalingFactor,
		dest * scalingFactor,
		location * scalingFactor,
		radius * scalingFactor,
		transform,
		Engine::engine.statsRecorder,
		enableSwirl,
		swirlPeriod);
}

void ReplayActionFill::apply()
{
    Engine::engine.edit<FillEditorColors>(
            Engine::engine.hashDag,
            Engine::engine.hashDagColors,
            location * scalingFactor,
            radius * scalingFactor);
}

void ReplayActionPaint::apply()
{
    Engine::engine.edit<SpherePaintEditor>(
            location * scalingFactor,
            radius * scalingFactor);
}