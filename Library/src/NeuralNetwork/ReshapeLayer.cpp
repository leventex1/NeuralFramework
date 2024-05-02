#include "ReshapeLayer.h"
#include <assert.h>
#include <sstream>
#include <iostream>


namespace_start

ReshapeLayer::ReshapeLayer(
	size_t inputHeight, size_t inputWidth, size_t inputDepth, 
	size_t outputHeight, size_t outputWidth, size_t outputDepth
) : m_InputHeight(inputHeight), m_InputWidth(inputWidth), m_InputDepth(inputDepth),
	m_OutputHeight(outputHeight), m_OutputWidth(outputWidth), m_OutputDepth(outputDepth)
{
	assert(inputWidth * inputHeight * inputDepth == outputWidth * outputHeight * outputDepth
		&& "Number of input units has to be the same as the number of output unit!");
}

ReshapeLayer::ReshapeLayer(const std::string& fromString)
{
	FromString(fromString);
}

Tensor3D ReshapeLayer::FeedForward(const Tensor3D& inputs)
{
	assert(m_InputHeight == inputs.GetRows() && 
			m_InputWidth == inputs.GetCols() &&
			m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	return Tensor3D(m_OutputHeight, m_OutputWidth, m_OutputDepth, inputs.GetData(), inputs.IsOnDevice());
}

Tensor3D ReshapeLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFucntion, float learningRate, size_t t)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	Tensor3D costs = NextLayer ? 
		NextLayer->BackPropagation(Tensor3D(m_OutputHeight, m_OutputWidth, m_OutputDepth, inputs.GetData(), inputs.IsOnDevice()),
		costFucntion, learningRate, t) : costFucntion.DiffCost(Tensor3D(m_OutputHeight, m_OutputWidth, m_OutputDepth, inputs.GetData(), inputs.IsOnDevice()));


	return Tensor3D(m_InputHeight, m_InputWidth, m_InputDepth, std::move(costs));
}

LayerShape ReshapeLayer::GetLayerShape() const
{
	return
	{
		m_InputHeight, m_InputWidth, m_InputDepth,
		m_OutputHeight, m_OutputWidth, m_OutputDepth
	};
}

std::string ReshapeLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " <<
		m_InputHeight << " " << m_InputWidth << " " << m_InputDepth << " " <<
		m_OutputHeight << " " << m_OutputWidth << " " << m_OutputDepth
	<< " ]";

	return ss.str();
}

void ReshapeLayer::FromString(const std::string& data)
{
	std::size_t numsStartPos = data.find(']');
	assert(data[0] == '[' && numsStartPos != std::string::npos && "Invalid hyperparameter format.");
	numsStartPos += 1;

	std::string hyperparams = data.substr(1, numsStartPos - 2);

	std::stringstream ss(hyperparams);

	try
	{
		ss >> m_InputHeight;
		ss >> m_InputWidth;
		ss >> m_InputDepth;
		ss >> m_OutputHeight;
		ss >> m_OutputWidth;
		ss >> m_OutputDepth;
	}
	catch (...)
	{
		assert(false && "Invalid shape param!");
	}
}

std::string ReshapeLayer::ToDebugString() const
{
	std::stringstream ss;

	ss << "Input shape (" << m_InputHeight << ", " << m_InputWidth << ", " << m_InputDepth << ")\n";
	ss << "Output shape (" << m_OutputHeight << ", " << m_OutputWidth << ", " << m_OutputDepth << ")\n";

	return ss.str();
}

std::string ReshapeLayer::Summarize() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << ClassName() << ":\t Input: (" <<
		shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << "), Output: (" <<
		shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << "), " <<
		", # learnable parameters: " << 0;

	return ss.str();
}

namespace_end