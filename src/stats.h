#pragma once

#include "typedefs.h"
#include <string>
#include <chrono>
#include <utility>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <atomic>

class StatsRecorder
{
public:
	StatsRecorder() = default;

	inline void next_frame()
	{
		frame++;
	}
	template<typename T>
	inline void report(std::string name, T time)
	{
		elements.push_back(Element{frame, std::move(name), double(time)});
	}
	inline void clear()
	{
		frame = 0;
		elements.resize(0);
	}

	inline void write_csv()
	{
		std::stringstream path;
#ifdef PROFILING_PATH
        path << PROFILING_PATH << "/";
        path << STATS_FILES_PREFIX;
#else
		path << "./profiling/";
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
        path << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
#endif
		path << ".stats.csv";

		std::ofstream os(path.str());
		checkAlways(os.is_open());

		write_csv(os);

		checkAlways(os.good());
		os.close();
	}
	inline void write_csv(std::ostream& stream)
	{
		for (auto& element : elements)
		{
			stream << element.frame;
			stream << ",";
			stream << element.name;
			stream << ",";
			stream << element.time;
			stream << "\n";
		}
	}
	inline uint32 get_frame_timestamp() const
	{
		return cast<uint32>(elements.size());
	}
	inline double get_value_in_frame(uint32 frameTimestamp, const std::string& name) const
	{
		if (elements.size() <= frameTimestamp) return 0;

		const int32 startFrame = elements[frameTimestamp].frame;
		uint32 index = frameTimestamp;
		while (index < elements.size() && elements[index].frame == startFrame)
		{
			if (elements[index].name == name)
			{
				return elements[index].time;
			}
			index++;
		}
		return 0;
	}

private:
	struct Element
	{
		int32 frame = -1;
		std::string name;
		double time = -1;
	};
	int32 frame = 0;
	std::vector<Element> elements;
};

enum class EStatNames
{
    EarlyExitChecks,
    EntirelyFull_AddColors,
    SkipEdit_CopyColors,
    LeafEdit,
    FindOrAddLeaf,
    InteriorEdit,
    FindOrAddInterior,
    Max
};

inline std::string get_stat_name(EStatNames name)
{
    switch (name)
    {
#define C(a, b) case a: return b;
        C(EStatNames::EarlyExitChecks, "early exit checks");
        C(EStatNames::EntirelyFull_AddColors,"entirely full - add colors");
        C(EStatNames::SkipEdit_CopyColors,"skip edit - copy colors");
        C(EStatNames::LeafEdit,"leaf edit");
        C(EStatNames::FindOrAddLeaf,"find or add leaf");
        C(EStatNames::InteriorEdit,"interior edit");
        C(EStatNames::FindOrAddInterior,"find or add interior");
#undef C
		case EStatNames::Max:
        default: check(false); return "";
    }
}

class LocalStatsRecorder
{
public:
    explicit LocalStatsRecorder(StatsRecorder& statsRecorder)
            : statsRecorder(&statsRecorder)
    {
    }
    LocalStatsRecorder(LocalStatsRecorder& localStatsRecorder)
            : localStatsRecorder(&localStatsRecorder)
    {
    }

    ~LocalStatsRecorder()
    {
        check((statsRecorder && !localStatsRecorder) || (!statsRecorder && localStatsRecorder));
        if (statsRecorder)
        {
            for (uint32 statName = 0; statName < uint32(EStatNames::Max); statName++)
            {
                if (cumulativeElements[statName].num == 0) continue;
                const auto name = get_stat_name(EStatNames(statName));
                statsRecorder->report(name, cumulativeElements[statName].time);
                statsRecorder->report(name + " average", cumulativeElements[statName].time / cumulativeElements[statName].num);
            }
        }
        else
        {
            for (uint32 statName = 0; statName < uint32(EStatNames::Max); statName++)
            {
                auto& element = localStatsRecorder->cumulativeElements[statName];
                element.time += cumulativeElements[statName].time;
                element.num += cumulativeElements[statName].num;
            }
        }
    }

	inline void report(EStatNames name, double time)
	{
		auto& element = cumulativeElements[uint32(name)];
		element.time += time;
		element.num++;
	}

private:
    StatsRecorder* statsRecorder = nullptr;
    LocalStatsRecorder* localStatsRecorder = nullptr;
    struct LocalStat
    {
        double time = 0;
        int32 num = 0;
    };
	LocalStat cumulativeElements[uint32(EStatNames::Max)];
};

inline std::string make_prefix(size_t num)
{
	std::ostringstream s;
	for (size_t index = 0; index < num; index++)
	{
		s << "\t";
	}
	return s.str();
}

class BasicStats
{
public:
	~BasicStats()
	{
	    check(name.empty());
	}

	inline void start_work(const std::string& workName)
	{
        check(name.empty());
        name = workName;
		time = std::chrono::high_resolution_clock::now();
	}
	template<typename T>
	inline double flush(T& recorder)
	{
		const auto currentTime = std::chrono::high_resolution_clock::now();
		const double timeInMs = double((currentTime - time).count()) / 1.e6;
        // ensure(timeInMs > 0.00001);
		if (ensure(!name.empty()))
		{
			recorder.report(name, timeInMs);
		}
		name.clear();
		return timeInMs;
	}

private:
	std::string name;
	std::chrono::time_point<std::chrono::high_resolution_clock> time;
};

class Stats
{
public:
	Stats()
		: prefix(get_prefix())
	{
		stack.push_back(this);
	}
	~Stats()
	{
		flush();
		check(stack.back() == this);
		stack.pop_back();
	}

	FORCEINLINE void start_work(const char* workName)
	{
		flush();
		name = workName;
		time = std::chrono::high_resolution_clock::now();
	}
	FORCEINLINE void start_level_work(uint32 level, const std::string& workName)
	{
		std::ostringstream s;
		s << "level " << level << ": " << workName;
		start_work(s.str().c_str());
	}
	FORCEINLINE double flush(bool print = true)
	{
		const auto currentTime = std::chrono::high_resolution_clock::now();
		const double timeInMs = double((currentTime - time).count()) / 1.e6;
		if (!name.empty() && print)
		{
			std::cout << prefix << name << " took " << timeInMs << "ms" << std::endl;
		}
		name.clear();
		return timeInMs;
	}

	inline static std::string get_prefix()
	{
		return make_prefix(stack.size());
	}

private:
	const std::string prefix;
	std::string name;
	std::chrono::time_point<std::chrono::high_resolution_clock> time;

	static std::vector<Stats*> stack;
};

#define PASTE_HELPER(a,b) a ## b
#define PASTE(a,b) PASTE_HELPER(a,b)
#define SCOPED_STATS(name) printf(name "...\n"); Stats PASTE(stats,__LINE__); PASTE(stats,__LINE__).start_work(name);

struct SimpleScopeStat
{
    SimpleScopeStat() : startTime(std::chrono::high_resolution_clock::now())
    {
    }

    inline double get_time() const
    {
		const auto currentTime = std::chrono::high_resolution_clock::now();
		const double timeInMs = double((currentTime - startTime).count()) / 1.e6;
		return timeInMs;
    }

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

struct EditScopeStat
{
#if EDITS_PROFILING
    EditScopeStat(LocalStatsRecorder& recorder, EStatNames name)
            : recorder(recorder)
            , name(name)
    {
        startTime = std::chrono::high_resolution_clock::now();
    }
    ~EditScopeStat()
    {
        if (!paused)
        {
            const auto time = get_time();
            recorder.report(name, time);
        }
    }

    inline void pause()
    {
        const auto time = get_time();

        check(!paused);
        paused = true;
        recorder.report(name, time);
    }

    inline void resume()
    {
        check(paused);
        paused = false;
        startTime = std::chrono::high_resolution_clock::now();
    }

private:
    LocalStatsRecorder& recorder;
    const EStatNames name;
    bool paused = false;
	std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    inline double get_time() const
    {
		const auto currentTime = std::chrono::high_resolution_clock::now();
		const double timeInMs = double((currentTime - startTime).count()) / 1.e6;
		return timeInMs;
    }
#else
    EditScopeStat(LocalStatsRecorder& recorder, EStatNames name)
    {
    }
    inline void pause()
    {
    }

    inline void resume()
    {
    }
#endif
};