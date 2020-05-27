#pragma once

#include "typedefs.h"
#include "utils.h"
#include "memory.h"
#include "cuda_error_check.h"

template<typename T, typename Size = uint64>
struct StaticArray
{
public:
	StaticArray() = default;
	StaticArray(T* __restrict__ arrayData, Size arraySize)
		: arrayData(arrayData)
		, arraySize(arraySize)
	{
	}
	StaticArray(decltype(nullptr))
		: StaticArray(nullptr, 0)
	{
	}
	
	static StaticArray<T, Size> allocate(const char* name, Size arraySize, EMemoryType type)
	{
		PROFILE_FUNCTION();
		auto* ptr = Memory::malloc<T>(name, arraySize * sizeof(T), type);
		return { ptr, arraySize };
	}

	HOST StaticArray<T, Size> create_gpu() const
	{
		PROFILE_FUNCTION();
	    check(is_valid());
        auto result = allocate(Memory::get_alloc_name(data()), size(), EMemoryType::GPU_Malloc);
        copy_to_gpu_strict(result);
        return result;
    }

    HOST void copy_to_gpu_strict(StaticArray<T, Size>& gpuArray) const
    {
		PROFILE_FUNCTION();
        check(gpuArray.size() == size());
        Memory::cuda_memcpy(gpuArray.data(), data(), size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    HOST void free()
    {
		PROFILE_FUNCTION();
		Memory::free(arrayData);
		reset();
	}
	HOST void reset()
	{
		*this = nullptr;
	}

	HOST_DEVICE Size size() const
	{
		return arraySize;
	}
	HOST_DEVICE const T* data() const
	{
		return arrayData;
	}
	HOST_DEVICE T* data()
	{
		return arrayData;
	}
	
	HOST_DEVICE bool is_valid() const
	{
		return arrayData != nullptr;
	}

	HOST_DEVICE bool is_valid_index(Size index) const
	{
		return index < arraySize;
	}
	
	HOST_DEVICE uint64 size_in_bytes() const
	{
		return arraySize * sizeof(T);
	}
	HOST_DEVICE double size_in_MB() const
	{
		return Utils::to_MB(size_in_bytes());
	}

	HOST_DEVICE T& operator[](Size index)
	{
		checkf(index < arraySize, "invalid index: %" PRIu64 " for size %" PRIu64, index, arraySize);
		return arrayData[index];
	}
	HOST_DEVICE const T& operator[](Size index) const
	{
		checkf(index < arraySize, "invalid index: %" PRIu64 " for size %" PRIu64, index, arraySize);
		return arrayData[index];
	}
	HOST_DEVICE bool operator==(StaticArray<T> other) const
	{
		return other.arrayData == arrayData && other.arraySize == arraySize;
	}
	HOST_DEVICE bool operator!=(StaticArray<T> other) const
	{
		return other.arrayData != arrayData || other.arraySize != arraySize;
	}

	HOST_DEVICE const T* begin() const
	{
		return arrayData;
	}
	HOST_DEVICE const T* end() const
	{
		return arrayData + arraySize;
	}
	
	HOST_DEVICE T* begin()
	{
		return arrayData;
	}
	HOST_DEVICE T* end()
	{
		return arrayData + arraySize;
	}

protected:
	T* __restrict__ arrayData = nullptr;
	Size arraySize = 0;
};

template<typename T, typename Size = uint64>
struct DynamicArray : StaticArray<T, Size>
{
public:
	DynamicArray() = default;
	DynamicArray(T* __restrict__ arrayData, Size arraySize)
		: StaticArray<T, Size>(arrayData, arraySize)
		, allocatedSize(arraySize)
	{
	}
	DynamicArray(decltype(nullptr))
		: StaticArray<T, Size>(nullptr, 0)
		, allocatedSize(0)
	{
	}
	DynamicArray(StaticArray<T, Size> array)
		: StaticArray<T, Size>(array)
		, allocatedSize(array.size())
	{
	}

    HOST void copy_to_gpu_flexible(DynamicArray<T, Size>& gpuArray) const
    {
        PROFILE_FUNCTION();
		
        if (!gpuArray.is_valid())
        {
            gpuArray = this->allocate(Memory::get_alloc_name(this->data()), this->allocated_size(), EMemoryType::GPU_Malloc);;
        }
        if (gpuArray.allocated_size() < this->size())
        {
            gpuArray.reserve(this->allocated_size() - gpuArray.allocated_size());
        }
        gpuArray.arraySize = this->size();
        Memory::cuda_memcpy(gpuArray.data(), this->data(), this->size() * sizeof(T), cudaMemcpyHostToDevice);
    }

	HOST_DEVICE Size allocated_size() const
	{
		return allocatedSize;
	}
	
	HOST_DEVICE uint64 allocated_size_in_bytes() const
	{
		return allocatedSize * sizeof(T);
	}
	HOST_DEVICE double allocated_size_in_MB() const
	{
		return Utils::to_MB(allocated_size_in_bytes());
	}

	HOST void hack_set_size(Size newSize)
    {
	    this->arraySize = newSize;
    }

	HOST Size add(T element)
	{
		check(this->arraySize <= allocatedSize);
		if(this->arraySize == allocatedSize)
		{
			reserve(allocatedSize); // Double the storage
		}

		const Size oldSize = this->arraySize;
		this->arraySize++;
		check(this->arraySize <= allocatedSize);

		(*this)[oldSize] = element;

		return oldSize;
	}
	template<typename... TArgs>
	HOST Size emplace(TArgs... args)
	{
		return add(T{ std::forward<TArgs>(args)... });
	}
	
	HOST void reserve(Size amount)
	{
		PROFILE_FUNCTION();
		check(this->arrayData);
		allocatedSize += amount;
		Memory::realloc(this->arrayData, allocatedSize * sizeof(T));
	}
	
	HOST void shrink()
	{
		PROFILE_FUNCTION();
		check(this->arrayData);
		check(this->arraySize != 0);
		allocatedSize = this->arraySize;
		Memory::realloc(this->arrayData, allocatedSize * sizeof(T));
	}
	
protected:
	Size allocatedSize = 0;
};