#pragma once

#include "typedefs.h"
#include <string>
#include <vector>
#include <fstream>
#include "gmath/Vector3.h"
#include "gmath/Matrix3x3.h"
#include <sstream>
#include <iomanip>

enum class EReplayActionType
{
	SetLocation,
	SetRotation,
	SetToolParameters,
	EditSphere,
	EditCube,
	EditCopy,
	EditFill,
	EditPaint,
	Undo,
	Redo,
	EndFrame,
};

inline std::string replay_action_to_string(EReplayActionType action)
{
	switch(action)
	{
	case EReplayActionType::SetLocation: return "SetLocation";
	case EReplayActionType::SetRotation: return "SetRotation";
	case EReplayActionType::SetToolParameters: return "SetToolParameters";
	case EReplayActionType::EditSphere: return "EditSphere";
	case EReplayActionType::EditCube: return "EditCube";
	case EReplayActionType::EditCopy: return "EditCopy";
	case EReplayActionType::EditFill: return "EditFill";
	case EReplayActionType::EditPaint: return "EditPaint";
	case EReplayActionType::Undo: return "Undo";
	case EReplayActionType::Redo: return "Redo";
	case EReplayActionType::EndFrame: return "EndFrame";
	default: checkAlways(false); return "";
	}
}

inline EReplayActionType string_to_replay_action(const std::string& string)
{
	if(string == "SetLocation") return EReplayActionType::SetLocation;
	if(string == "SetRotation") return EReplayActionType::SetRotation;
	if(string == "SetToolParameters") return EReplayActionType::SetToolParameters;
	if(string == "EditSphere") return EReplayActionType::EditSphere;
	if(string == "EditCube") return EReplayActionType::EditCube;
	if(string == "EditCopy") return EReplayActionType::EditCopy;
	if(string == "EditFill") return EReplayActionType::EditFill;
	if(string == "EditPaint") return EReplayActionType::EditPaint;
	if(string == "Undo") return EReplayActionType::Undo;
	if(string == "Redo") return EReplayActionType::Redo;
	if(string == "EndFrame") return EReplayActionType::EndFrame;
	checkAlways(false);
	return EReplayActionType::SetLocation;
}

class IReplayAction
{
public:
	const EReplayActionType type;

	IReplayAction(EReplayActionType type) : type(type) {}
	virtual ~IReplayAction() = default;

	virtual void load(const std::vector<std::string>& data) = 0;
	virtual void write(std::vector<std::string>& data) = 0;
	virtual void apply() = 0;
};

template<EReplayActionType Type>
class TReplayAction : public IReplayAction
{
public:
	TReplayAction()
		: IReplayAction(Type)
	{
	}
};

class ReplayActionEndFrame : public TReplayAction<EReplayActionType::EndFrame>
{
public:
	ReplayActionEndFrame() = default;

	virtual void load(const std::vector<std::string>& data) override {}
	virtual void write(std::vector<std::string>& data) override {}
	virtual void apply() override {}
};

class ReplayActionUndo : public TReplayAction<EReplayActionType::Undo>
{
public:
	ReplayActionUndo() = default;

	virtual void load(const std::vector<std::string>& data) override {}
	virtual void write(std::vector<std::string>& data) override {}
	virtual void apply() override;
};

class ReplayActionRedo : public TReplayAction<EReplayActionType::Redo>
{
public:
	ReplayActionRedo() = default;

	virtual void load(const std::vector<std::string>& data) override {}
	virtual void write(std::vector<std::string>& data) override {}
	virtual void apply() override;
};

class ReplayActionSetLocation : public TReplayAction<EReplayActionType::SetLocation>
{
public:
	ReplayActionSetLocation() = default;
	ReplayActionSetLocation(Vector3 location)
		: location(location)
	{
	}

	Vector3 location;

	virtual void load(const std::vector<std::string>& data) override
	{
		checkAlways(data.size() == 3);
		location.X = std::stof(data[0]);
		location.Y = std::stof(data[1]);
		location.Z = std::stof(data[2]);
	}
	virtual void write(std::vector<std::string>& data) override
	{
		data.push_back(std::to_string(location.X));
		data.push_back(std::to_string(location.Y));
		data.push_back(std::to_string(location.Z));
	}

	virtual void apply() override;
};

class ReplayActionSetRotation : public TReplayAction<EReplayActionType::SetRotation>
{
public:
	ReplayActionSetRotation() = default;
	ReplayActionSetRotation(Matrix3x3 rotation)
		: rotation(rotation)
	{
	}

	Matrix3x3 rotation;

	virtual void load(const std::vector<std::string>& data) override
	{
		checkAlways(data.size() == 9);
		rotation.D00 = std::stof(data[0]);
		rotation.D01 = std::stof(data[1]);
		rotation.D02 = std::stof(data[2]);
		rotation.D10 = std::stof(data[3]);
		rotation.D11 = std::stof(data[4]);
		rotation.D12 = std::stof(data[5]);
		rotation.D20 = std::stof(data[6]);
		rotation.D21 = std::stof(data[7]);
		rotation.D22 = std::stof(data[8]);
	}
	virtual void write(std::vector<std::string>& data) override
	{
		data.push_back(std::to_string(rotation.D00));
		data.push_back(std::to_string(rotation.D01));
		data.push_back(std::to_string(rotation.D02));
		data.push_back(std::to_string(rotation.D10));
		data.push_back(std::to_string(rotation.D11));
		data.push_back(std::to_string(rotation.D12));
		data.push_back(std::to_string(rotation.D20));
		data.push_back(std::to_string(rotation.D21));
		data.push_back(std::to_string(rotation.D22));
	}

	virtual void apply() override;
};

class ReplayActionSetToolParameters : public TReplayAction<EReplayActionType::SetToolParameters>
{
public:
	ReplayActionSetToolParameters() = default;
	ReplayActionSetToolParameters(uint3 path, uint3 copySourcePath, uint3 copyDestPath, float radius, uint32 tool)
		: path(path)
		, copySourcePath(copySourcePath)
		, copyDestPath(copyDestPath)
		, radius(radius)
		, tool(tool)
	{
	}

    uint3 path{};
    uint3 copySourcePath;
    uint3 copyDestPath;
    float radius{};
	uint32 tool{};

	virtual void load(const std::vector<std::string>& data) override
	{
		checkAlways(data.size() == 11);
		path.x = (uint32)std::stoi(data[0]);
		path.y = (uint32)std::stoi(data[1]);
		path.z = (uint32)std::stoi(data[2]);
		copySourcePath.x = (uint32)std::stoi(data[3]);
		copySourcePath.y = (uint32)std::stoi(data[4]);
		copySourcePath.z = (uint32)std::stoi(data[5]);
		copyDestPath.x = (uint32)std::stoi(data[6]);
		copyDestPath.y = (uint32)std::stoi(data[7]);
		copyDestPath.z = (uint32)std::stoi(data[8]);
		radius = std::stof(data[9]);
		tool = (uint32)std::stoi(data[10]);
	}
	virtual void write(std::vector<std::string>& data) override
	{
		data.push_back(std::to_string(path.x));
		data.push_back(std::to_string(path.y));
		data.push_back(std::to_string(path.z));
		data.push_back(std::to_string(copySourcePath.x));
		data.push_back(std::to_string(copySourcePath.y));
		data.push_back(std::to_string(copySourcePath.z));
		data.push_back(std::to_string(copyDestPath.x));
		data.push_back(std::to_string(copyDestPath.y));
		data.push_back(std::to_string(copyDestPath.z));
		data.push_back(std::to_string(radius));
		data.push_back(std::to_string(tool));
	}

	virtual void apply() override;
};

class ReplayActionSphere : public TReplayAction<EReplayActionType::EditSphere>
{
public:
	ReplayActionSphere() = default;
	ReplayActionSphere(float3 location, float radius, bool add)
		: location(location)
		, radius(radius)
		, add(add)
	{
	}

	float3 location{};
	float radius = 0;
	bool add = false;

	virtual void load(const std::vector<std::string>& data) override
	{
		checkAlways(data.size() == 5);
		location.x = std::stof(data[0]);
		location.y = std::stof(data[1]);
		location.z = std::stof(data[2]);
		radius = std::stof(data[3]);
		checkAlways(data[4] == "true" || data[4] == "false" || data[4] == "TRUE" || data[4] == "FALSE");
		add = data[4] == "true" || data[4] == "TRUE";
	}
	virtual void write(std::vector<std::string>& data) override
	{
		data.push_back(std::to_string(location.x));
		data.push_back(std::to_string(location.y));
		data.push_back(std::to_string(location.z));
		data.push_back(std::to_string(radius));
		data.push_back(add ? "true" : "false");
	}

	virtual void apply() override;
};

class ReplayActionCube : public TReplayAction<EReplayActionType::EditCube>
{
public:
	ReplayActionCube() = default;
	ReplayActionCube(float3 location, float radius, bool add)
		: location(location)
		, radius(radius)
		, add(add)
	{
	}

	float3 location{};
	float radius = 0;
	bool add = false;

	virtual void load(const std::vector<std::string>& data) override
	{
		checkAlways(data.size() == 5);
		location.x = std::stof(data[0]);
		location.y = std::stof(data[1]);
		location.z = std::stof(data[2]);
		radius = std::stof(data[3]);
		checkAlways(data[4] == "true" || data[4] == "false" || data[4] == "TRUE" || data[4] == "FALSE");
		add = data[4] == "true" || data[4] == "TRUE";
	}
	virtual void write(std::vector<std::string>& data) override
	{
		data.push_back(std::to_string(location.x));
		data.push_back(std::to_string(location.y));
		data.push_back(std::to_string(location.z));
		data.push_back(std::to_string(radius));
		data.push_back(add ? "true" : "false");
	}

	virtual void apply() override;
};

class ReplayActionCopy : public TReplayAction<EReplayActionType::EditCopy>
{
public:
	ReplayActionCopy() = default;
	ReplayActionCopy(float3 location, float3 src, float3 dest, float radius, const Matrix3x3& transform, bool enableSwirl, float swirlPeriod)
		: location(location)
		, src(src)
		, dest(dest)
		, radius(radius)
		, transform(transform)
		, enableSwirl(enableSwirl)
		, swirlPeriod(swirlPeriod)
	{
	}

	float3 location{};
	float3 src{};
	float3 dest{};
	float radius = 0;
	Matrix3x3 transform;
	bool enableSwirl = false;
	float swirlPeriod = 1;
	
	virtual void load(const std::vector<std::string>& data) override
	{
		checkAlways(data.size() == 10 || data.size() == 21);
		location.x = std::stof(data[0]);
		location.y = std::stof(data[1]);
		location.z = std::stof(data[2]);
		src.x = std::stof(data[3]);
		src.y = std::stof(data[4]);
		src.z = std::stof(data[5]);
		dest.x = std::stof(data[6]);
		dest.y = std::stof(data[7]);
		dest.z = std::stof(data[8]);
		radius = std::stof(data[9]);
		if (data.size() == 21)
		{
			transform.D00 = std::stof(data[10]);
			transform.D01 = std::stof(data[11]);
			transform.D02 = std::stof(data[12]);
			transform.D10 = std::stof(data[13]);
			transform.D11 = std::stof(data[14]);
			transform.D12 = std::stof(data[15]);
			transform.D20 = std::stof(data[16]);
			transform.D21 = std::stof(data[17]);
			transform.D22 = std::stof(data[18]);
			checkAlways(data[19] == "true" || data[19] == "false" || data[19] == "TRUE" || data[19] == "FALSE");
			enableSwirl = data[19] == "true" || data[19] == "TRUE";
			swirlPeriod = std::stof(data[20]);
		}
	}
	virtual void write(std::vector<std::string>& data) override
	{
		data.push_back(std::to_string(location.x));
		data.push_back(std::to_string(location.y));
		data.push_back(std::to_string(location.z));
		data.push_back(std::to_string(src.x));
		data.push_back(std::to_string(src.y));
		data.push_back(std::to_string(src.z));
		data.push_back(std::to_string(dest.x));
		data.push_back(std::to_string(dest.y));
		data.push_back(std::to_string(dest.z));
		data.push_back(std::to_string(radius));
		data.push_back(std::to_string(transform.D00));
		data.push_back(std::to_string(transform.D01));
		data.push_back(std::to_string(transform.D02));
		data.push_back(std::to_string(transform.D10));
		data.push_back(std::to_string(transform.D11));
		data.push_back(std::to_string(transform.D12));
		data.push_back(std::to_string(transform.D20));
		data.push_back(std::to_string(transform.D21));
		data.push_back(std::to_string(transform.D22));
		data.push_back(enableSwirl ? "true" : "false");
		data.push_back(std::to_string(swirlPeriod));
	}

	virtual void apply() override;
};

class ReplayActionFill : public TReplayAction<EReplayActionType::EditFill>
{
public:
	ReplayActionFill() = default;
	ReplayActionFill(float3 location, float radius)
		: location(location)
		, radius(radius)
	{
	}

	float3 location{};
	float radius = 0;

	virtual void load(const std::vector<std::string>& data) override
	{
		checkAlways(data.size() == 4);
		location.x = std::stof(data[0]);
		location.y = std::stof(data[1]);
		location.z = std::stof(data[2]);
		radius = std::stof(data[3]);
	}
	virtual void write(std::vector<std::string>& data) override
	{
		data.push_back(std::to_string(location.x));
		data.push_back(std::to_string(location.y));
		data.push_back(std::to_string(location.z));
		data.push_back(std::to_string(radius));
	}

	virtual void apply() override;
};

class ReplayActionPaint : public TReplayAction<EReplayActionType::EditPaint>
{
public:
	ReplayActionPaint() = default;
	ReplayActionPaint(float3 location, float radius)
		: location(location)
		, radius(radius)
	{
	}

	float3 location{};
	float radius = 0;

	virtual void load(const std::vector<std::string>& data) override
	{
		checkAlways(data.size() == 4);
		location.x = std::stof(data[0]);
		location.y = std::stof(data[1]);
		location.z = std::stof(data[2]);
		radius = std::stof(data[3]);
	}
	virtual void write(std::vector<std::string>& data) override
	{
		data.push_back(std::to_string(location.x));
		data.push_back(std::to_string(location.y));
		data.push_back(std::to_string(location.z));
		data.push_back(std::to_string(radius));
	}

	virtual void apply() override;
};

inline std::unique_ptr<IReplayAction> replay_action_factory(EReplayActionType type)
{
	switch (type)
	{
	case EReplayActionType::SetLocation: return std::make_unique<ReplayActionSetLocation>();
	case EReplayActionType::SetRotation: return std::make_unique<ReplayActionSetRotation>();
	case EReplayActionType::SetToolParameters: return std::make_unique<ReplayActionSetToolParameters>();
	case EReplayActionType::EditSphere: return std::make_unique<ReplayActionSphere>();
	case EReplayActionType::EditCube: return std::make_unique<ReplayActionCube>();
	case EReplayActionType::EditCopy: return std::make_unique<ReplayActionCopy>();
	case EReplayActionType::EditFill: return std::make_unique<ReplayActionFill>();
	case EReplayActionType::EditPaint: return std::make_unique<ReplayActionPaint>();
	case EReplayActionType::Undo: return std::make_unique<ReplayActionUndo>();
	case EReplayActionType::Redo: return std::make_unique<ReplayActionRedo>();
	case EReplayActionType::EndFrame: return std::make_unique<ReplayActionEndFrame>();
	default: checkAlways(false);
		return {};
	}
}

class ReplayManager
{
public:
	inline void replay_frame()
	{
		if (replayIndex < actions.size())
		{
            frameIndex++;
            printf("Replaying frame %" PRIu64 "/%" PRIu64 "\n", uint64(frameIndex), uint64(numFrames));
			IReplayAction* action;
			do
			{
				action = actions[replayIndex].get();
				check(action);
				action->apply();
				replayIndex++;
			}
			while (action->type != EReplayActionType::EndFrame && ensure(replayIndex < actions.size()));
		}
	}

	template<typename T, typename ... TArgs>
	inline void add_action(TArgs&&... args)
	{
		actions.push_back(std::make_unique<T>(std::forward<TArgs>(args)...));
	}

	inline void write_csv()
	{
		std::stringstream path;
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		path << "./replays/" << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S.csv");

		std::ofstream os(path.str());
		checkAlways(os.is_open());
		
		write_csv(os);
		
		checkAlways(os.good());
		os.close();
	}
	inline void write_csv(std::ostream& stream)
	{
		for(auto& action : actions)
		{
			stream << replay_action_to_string(action->type);
			std::vector<std::string> data;
			action->write(data);
			for (auto& str : data)
			{
				stream  << "," << str;
			}
			stream << "\n";
		}
	}
	inline void load_csv(const std::string& path)
	{
        printf("Loading replay %s\n", path.c_str());

		std::ifstream is(path);
		checkfAlways(is.is_open() && is.good(), "Path: %s", path.c_str());

		frameIndex = 0;
		numFrames = 0;
		replayIndex = 0;
		actions.resize(0);

		load_csv(is);

		is.close();
	}
	inline void load_csv(std::ifstream& stream)
	{
		std::vector<std::string> data;

		char c;
		while (!(stream.get(c), stream.eof()))
		{
			if (c == '\n')
			{
				const EReplayActionType type = string_to_replay_action(data[0]);
				data.erase(data.begin());

                if (type == EReplayActionType::EndFrame) numFrames++;

				// remove empty cells
				while(!data.empty() && data.back().empty()) data.pop_back();

				auto action = replay_action_factory(type);
				checkAlways(action->type == type);
				action->load(data);
				actions.push_back(std::move(action));
				data.resize(0);
			}
			else if (c == ',')
			{
				data.emplace_back();
			}
			else
			{
				if (data.empty())
				{
					data.emplace_back();
				}
				data.back().push_back(c);
			}
		}

		check(data.empty());
	}

	inline void clear()
	{
		actions.resize(0);
	}
	inline void reset_replay()
	{
	    frameIndex = 0;
		replayIndex = 0;
	}
	inline bool is_empty() const
	{
		return actions.empty();
	}
	inline bool at_end() const
	{
		return replayIndex == actions.size();
	}
    inline std::size_t num_frames() const
    {
        return numFrames;
    }
    inline Vector3 get_initial_position() const
    {
        for (auto& action : actions)
        {
            if (action->type == EReplayActionType::SetLocation)
            {
                return static_cast<const ReplayActionSetLocation&>(*action).location;
            }
        }
        return {};
    }
    inline Matrix3x3 get_initial_rotation() const
    {
        for (auto& action : actions)
        {
            if (action->type == EReplayActionType::SetRotation)
            {
                return static_cast<const ReplayActionSetRotation&>(*action).rotation;
            }
        }
        return {};
    }

private:
    std::size_t frameIndex = 0;
    std::size_t numFrames = 0;
    std::size_t replayIndex = 0;
	std::vector<std::unique_ptr<IReplayAction>> actions;
};
