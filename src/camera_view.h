#pragma once

#include "typedefs.h"
#include "gmath/Vector3.h"
#include "gmath/Matrix3x3.h"
#include "gmath/Quaternion.h"

struct CameraView
{
	static constexpr float fov = 60.f;

	Vector3 position = { 0.f, 0.f, 0.f };
	Matrix3x3 rotation = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

	inline Vector3 right() const
	{
		// No idea why minus, but else it's inverted
		return { -rotation.D00, -rotation.D01, -rotation.D02 };
	}
	inline Vector3 up() const
	{
		return { rotation.D10, rotation.D11, rotation.D12 };
	}
	inline Vector3 forward() const
	{
		return { rotation.D20, rotation.D21, rotation.D22 };
	}
};