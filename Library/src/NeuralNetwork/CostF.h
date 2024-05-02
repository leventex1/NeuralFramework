#pragma once
#include <functional>
#include <vector>
#include <stdexcept>
#include "Core.h"
#include <assert.h>
#include <iostream>
#include "../Math/Operation.h"
#include "../Math/Tensor3D.h"


namespace_start

struct LIBRARY_API CostFunction
{
	std::function<float(const Tensor3D& output)> Cost;
	std::function<Tensor3D(const Tensor3D& output)> DiffCost;
};

struct LIBRARY_API MeanSquareError : public CostFunction
{
	MeanSquareError(const Tensor3D& target);
};

struct LIBRARY_API CrossEntropyLoss : public CostFunction
{
	CrossEntropyLoss(const Tensor3D& target);
};

struct LIBRARY_API BinaryCrossEntropyLoss : public CostFunction
{
	BinaryCrossEntropyLoss(const Tensor3D& target);
};

namespace_end