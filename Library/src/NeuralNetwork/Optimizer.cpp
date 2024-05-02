#include "Optimizer.h"

#include <MogiAccelerator.h>


namespace_start

void SGDOptimizer::Update(Tensor* params, Tensor* gradient, float learningRate)
{
	if (params->IsOnDevice() && gradient->IsOnDevice())
	{
		gradient->Mult(learningRate);
		params->Sub(*gradient);
	}
	else if (!params->IsOnDevice() && !gradient->IsOnDevice())
	{
		params->ElementWise(*gradient, [learningRate](float p, float g) -> float { return p - learningRate * g; });
	}
	else
	{
		throw std::runtime_error("Params and gradient on different devices.");
	}
}

std::string SGDOptimizer::ToString() const
{
	std::stringstream ss;
	ss << GetName() << " ";
	return ss.str();
}


void AdamOptimizer::ToHost()
{
	m_FirstMoments.ToHost();
	m_SecondMoments.ToHost();
}

void AdamOptimizer::ToDevice()
{
	m_FirstMoments.ToDevice();
	m_SecondMoments.ToDevice();
}

void AdamOptimizer::Update(Tensor* params, Tensor* gradient, float learningRate)
{
	const float b1 = 0.9f;
	const float b2 = 0.999f;
	const float ep = 0.0000001f;
	size_t timeStep = m_TrainingTimeStep;

	if (params->IsOnDevice() && gradient->IsOnDevice())
	{
		accelerator::CudaAdamOptimization(params, gradient, &m_FirstMoments, &m_SecondMoments, b1, b2, ep, timeStep, learningRate);
	}
	else if (!params->IsOnDevice() && !gradient->IsOnDevice())
	{
		m_FirstMoments.ElementWise(*gradient, [b1](float m, float g) -> float { return b1 * m + (1.0f - b1) * g; });
		m_SecondMoments.ElementWise(*gradient, [b2](float v, float g) -> float { return b2 * v + (1.0f - b2) * g * g; });

		Tensor2D correctedFirstMoments = Map(m_FirstMoments, [b1, timeStep](float m) -> float { return m / (1.0f - pow(b1, timeStep)); });
		Tensor2D correctedSecondMoments = Map(m_SecondMoments, [b2, timeStep](float v) -> float { return v / (1.0f - pow(b2, timeStep)); });

		Tensor2D correctedGradient = Map(correctedFirstMoments, [learningRate](float m) -> float { return learningRate * m; });
		correctedGradient.ElementWise(correctedSecondMoments, [ep](float v1, float v2) -> float { return v1 / (sqrt(v2) + ep); });

		params->Sub(correctedGradient);
	}
	else
	{
		throw std::runtime_error("Params and gradient on different devicec.");
	}

	m_TrainingTimeStep++;
}

std::string AdamOptimizer::ToString() const
{
	std::stringstream ss;
	ss << m_TrainingTimeStep << " " << m_FirstMoments.GetSize() << " ";
	for (size_t t = 0; t < m_FirstMoments.GetSize(); t++)
		ss << m_FirstMoments.GetData()[t] << " ";
	for (size_t t = 0; t < m_SecondMoments.GetSize(); t++)
		ss << m_SecondMoments.GetData()[t] << " ";
	return ss.str();
}

void AdamOptimizer::FromString(const std::string& fromString)
{
	std::stringstream ss(fromString);
	size_t size;
	ss >> m_TrainingTimeStep >> size;
	m_FirstMoments = Tensor2D(size, 1);
	m_SecondMoments = Tensor2D(size, 1);
	for (size_t t = 0; t < size; t++)
		ss >> m_FirstMoments.GetData()[t];
	for (size_t t = 0; t < size; t++)
		ss >> m_SecondMoments.GetData()[t];
};

namespace_end