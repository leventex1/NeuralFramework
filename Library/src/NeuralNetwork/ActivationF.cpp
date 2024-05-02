#include "ActivationF.h"

#include <MogiAccelerator.h>


namespace_start

Sigmoid::Sigmoid()
{
	Activation = [](float v) -> float { return 1.0f / (1.0f + exp(-v)); };
	DiffActivation = [](float v) -> float { float sigm = 1.0f / (1.0f + exp(-v)); return sigm * (1.0f - sigm); };

	MapActivation = [](Tensor* tensor) -> void {
		if (tensor->IsOnDevice())
		{
			accelerator::CudaSigmoid(tensor);
		}
		else
		{
			tensor->Map([](float v) -> float { return 1.0f / (1.0f + exp(-v)); });
		}
	};

	MapDiffActivation = [](Tensor* tensor) -> void {
		if (tensor->IsOnDevice())
		{
			accelerator::CudaDiffSigmoid(tensor);
		}
		else
		{
			tensor->Map([](float v) -> float { float sigm = 1.0f / (1.0f + exp(-v)); return sigm * (1.0f - sigm); });
		}
	};

	Name = "sigmoid";
	Params = "";
}


// Change alpha to get leakyRelU
RelU::RelU(float alpha)
{
	InitActivation(alpha);
	InitDiffActivation(alpha);
	Name = "RelU";
	Params = std::to_string(alpha);
}

RelU::RelU(const std::string& param)
{
	float alpha = std::stof(param);
	InitActivation(alpha);
	InitDiffActivation(alpha);
	Name = "RelU";
	Params = std::to_string(alpha);
}

void RelU::InitActivation(float alpha) 
{ 
	Activation = [alpha](float v) -> float { return v > 0 ? v : alpha * v; };
	MapActivation = [alpha](Tensor* tensor) -> void {
		if (tensor->IsOnDevice())
		{
			accelerator::CudaRelU(tensor, alpha);
		}
		else
		{
			tensor->Map([alpha](float v) -> float { return v > 0 ? v : alpha * v; });
		}
	};
}

void RelU::InitDiffActivation(float alpha) 
{ 
	DiffActivation = [alpha](float v) -> float { return v > 0 ? 1.0f : alpha; }; 
	MapDiffActivation = [alpha](Tensor* tensor) -> void {
		if (tensor->IsOnDevice())
		{
			accelerator::CudaDiffRelU(tensor, alpha);
		}
		else
		{
			tensor->Map([alpha](float v) -> float { return v > 0 ? 1.0f : alpha; });
		}
	};
}

namespace_end