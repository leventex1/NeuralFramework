#pragma once
#include <functional>
#include <string>
#include <assert.h>

#include "Core.h"
#include "../Math/Tensor.h"


namespace_start

struct LIBRARY_API ActivationFunciton
{
	std::function<float(float v)> Activation;
	std::function<float(float v)> DiffActivation;

	std::function<void(Tensor*)> MapActivation;
	std::function<void(Tensor*)> MapDiffActivation;

	std::string Name;
	std::string Params;
};

struct LIBRARY_API Sigmoid : public ActivationFunciton
{
	Sigmoid();
};

struct LIBRARY_API RelU : public ActivationFunciton
{
	RelU(float alpha = 0.0f);
	RelU(const std::string& param);
	void InitActivation(float alpha);
	void InitDiffActivation(float alpha);
};

static ActivationFunciton GetActivationFunctionByName(const std::string& name, const std::string& params)
{
	if (name == "sigmoid") return Sigmoid();
	if (name == "RelU") return RelU(params);

	assert(false && "Unknown activation function name!");
	return ActivationFunciton();
}

namespace_end