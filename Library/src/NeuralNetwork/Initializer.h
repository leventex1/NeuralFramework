#pragma once
#include <functional>
#include "Core.h"
#include "Layer.h"
#include <chrono>
#include <random>


namespace_start

struct LIBRARY_API Initializer
{
	std::function<float()> Init;
};

struct LIBRARY_API Xavier : public Initializer
{
	Xavier(size_t inputUnits, size_t outputUnits)
	{
		float stddev = std::sqrt(2.0 / (float)(inputUnits + outputUnits));

		Init = [stddev]() -> float
		{
			std::random_device rd;
			std::mt19937 gen(rd());
			std::normal_distribution<> d(0.0f, stddev);

			return d(gen);
		};
	}
};

struct LIBRARY_API He : public Initializer
{
	He(size_t inputUnits)
	{
		float stddev = std::sqrt(2.0 / (float)(inputUnits));

		Init = [stddev]() -> float
		{
			std::random_device rd;
			std::mt19937 gen(rd());
			std::normal_distribution<> d(0.0f, stddev);

			return d(gen);
		};
	}
};

struct LIBRARY_API Uniform : public Initializer
{
	Uniform(float min, float max)
	{
		Init = [min, max]() -> float {
			std::random_device rd;
			float r = (float)rd() / (float)rd.max();
			return min + r * (max - min);
		};
	}
};

namespace_end
