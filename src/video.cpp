#include "video.h"
#include "replay.h"
#include "engine.h"

void VideoManager::load_video(const std::string& path)
{
    printf("Loading video %s\n", path.c_str());

    std::ifstream is(path);
    checkfAlways(is.is_open() && is.good(), "Path: %s", path.c_str());
    checkAlways(replays.empty());

    bool isPath = true;
    Replay replay;

    std::string line;
    while (std::getline(is, line))
    {
        if (line.empty()) continue;
        if (isPath)
        {
            replay.replayPath = "./replays/" SCENE "_video_" + line + ".csv";
        }
        else
        {
            replay.timeAfterReplay = std::stof(line);
            replays.push_back(replay);
        }
        isPath = !isPath;
    }
    checkfAlways(isPath, "Invalid video file, need one time after each replay");

    is.close();
}

void VideoManager::tick(Engine& engine)
{
    if (replayIndex >= replays.size()) return;

    if (isReplaying)
    {
        if (engine.replayReader.at_end())
        {
            isReplaying = false;
            const double timeAfterReplay = replays[replayIndex].timeAfterReplay;
            replayIndex++;
            if(replayIndex < replays.size())
            {
                nextReplayTime = Utils::seconds() + timeAfterReplay;
                ReplayManager manager;
                manager.load_csv(replays[replayIndex].replayPath);
                engine.targetView.position = manager.get_initial_position();
                engine.targetView.rotation = manager.get_initial_rotation();
                engine.init_target_lerp();
                engine.targetLerpSpeed = 1. / timeAfterReplay;
                printf("Changing replay to %s in %f seconds...\n", replays[replayIndex].replayPath.c_str(), timeAfterReplay);
            }
        }
    }
    else
    {
        if (Utils::seconds() > nextReplayTime)
        {
            engine.moveToTarget = false;
            engine.replayReader.load_csv(replays[replayIndex].replayPath);
            isReplaying = true;
        }
    }
}