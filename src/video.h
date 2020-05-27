#pragma once

#include "typedefs.h"

class VideoManager
{
public:
    void load_video(const std::string& path);
    void tick(class Engine& engine);

private:
    struct Replay
    {
        std::string replayPath;
        double timeAfterReplay = 0;
    };
    std::vector<Replay> replays;
    uint32 replayIndex = 0;
    double nextReplayTime = 0;
    bool isReplaying = false;
};