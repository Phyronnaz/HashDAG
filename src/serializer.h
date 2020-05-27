#pragma once

#include "typedefs.h"
#include "dag_info.h"
#include "array.h"
#include "stats.h"
#include <string>
#include <fstream>
#include <iosfwd>

class FileWriter
{
public:
	explicit FileWriter(const std::string& path)
		: os(path, std::ios::binary)
	{
		printf("%sWriting to %s\n", Stats::get_prefix().c_str(), path.c_str());
		checkfAlways(os.is_open(), "File: %s", path.c_str());
	}
	~FileWriter()
	{
		checkAlways(os.good());
		os.close();
		printf("%sWriting took %fms\n", Stats::get_prefix().c_str(), double((std::chrono::high_resolution_clock::now() - startTime).count()) / 1.e6);
	}
	
	inline void write(const void* data, size_t num)
	{
		os.write(reinterpret_cast<const char*>(data), std::streamsize(num));
		check(os.good());
	}
	inline void write(uint32 data)
	{
		write(&data, sizeof(uint32));
	}
	inline void write(uint64 data)
	{
		write(&data, sizeof(uint64));
	}
	inline void write(double data)
	{
		write(&data, sizeof(double));
	}
	template<typename T, typename Size = uint64>
	inline void write(const StaticArray<T, Size>& array)
	{
		write(array.size());
		write(array.data(), array.size() * sizeof(T));
	}
	inline void write(const DAGInfo& info)
	{
		write(info.boundsAABBMin.X);
		write(info.boundsAABBMin.Y);
		write(info.boundsAABBMin.Z);
		
		write(info.boundsAABBMax.X);
		write(info.boundsAABBMax.Y);
		write(info.boundsAABBMax.Z);
	}

private:
	const std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
	std::ofstream os;
};

class FileReader
{
public:
	explicit FileReader(const std::string& path)
		: is(path, std::ios::binary)
	{
		printf("%sLoading from %s\n", Stats::get_prefix().c_str(), path.c_str());
		checkfAlways(is.is_open(), "File: %s", path.c_str());
	}
	~FileReader()
	{
		checkAlways(is.good());
		is.close();
		printf("%sLoading took %fms\n", Stats::get_prefix().c_str(), double((std::chrono::high_resolution_clock::now() - startTime).count()) / 1.e6);
	}

	inline void read(void* data, size_t num)
	{
		is.read(reinterpret_cast<char*>(data), std::streamsize(num));
		check(is.good());
	}
	inline void read(uint32& data)
	{
		read(&data, sizeof(uint32));
	}
	inline void read(uint64& data)
	{
		read(&data, sizeof(uint64));
	}
	inline void read(double& data)
	{
		read(&data, sizeof(double));
	}
	template<typename T, typename Size = uint64>
	inline void read(StaticArray<T, Size>& array, const char* name, EMemoryType type)
	{
		check(!array.is_valid());
		Size size;
		read(size);
		array = StaticArray<T, Size>::allocate(name, size, type);
		read(array.data(), array.size() * sizeof(T));
	}
	inline void read(DAGInfo& info)
	{
		read(info.boundsAABBMin.X);
		read(info.boundsAABBMin.Y);
		read(info.boundsAABBMin.Z);
		
		read(info.boundsAABBMax.X);
		read(info.boundsAABBMax.Y);
		read(info.boundsAABBMax.Z);
	}

private:
	const std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
	std::ifstream is;
};