#include "MaxPoolingLayer.h"
#include <assert.h>
#include <sstream>
#include <iostream>


namespace_start

MaxPoolingLayer::MaxPoolingLayer(
	size_t inputHeight, size_t inputWidth, size_t inputDepth,
	size_t poolingHeight, size_t poolingWidth
) : m_InputHeight(inputHeight), m_InputWidth(inputWidth), m_InputDepth(inputDepth), m_PoolingWidth(poolingWidth), m_PoolingHeight(poolingHeight)
{
}

MaxPoolingLayer::MaxPoolingLayer(const std::string& fromString)
{
	FromString(fromString);
}

Tensor3D MaxPoolingLayer::FeedForward(const Tensor3D& inputs)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	LayerShape layerShape = GetLayerShape();

	Tensor3D output = MaxPool(inputs, m_PoolingHeight, m_PoolingWidth);

	return output;
}

Tensor3D MaxPoolingLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFucntion, float learningRate, size_t t)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	LayerShape layerShape = GetLayerShape();

	Tensor3D output = MaxPool(inputs, m_PoolingHeight, m_PoolingWidth);

	Tensor3D costs = NextLayer ?
								NextLayer->BackPropagation(output, costFucntion, learningRate, t) :
								output;

	assert(layerShape.OutputRows == costs.GetRows() &&
		layerShape.OutputCols == costs.GetCols() &&
		layerShape.OutputDepth == costs.GetDepth()
		&& "Costs shape not match!");

	Tensor3D gradient = Tensor3D(layerShape.InputRows, layerShape.InputCols, layerShape.InputDepth, 0.0f, inputs.IsOnDevice());

	DistributeReverseMaxPool(gradient, inputs, costs, m_PoolingHeight, m_PoolingWidth);

	return gradient;
}

LayerShape MaxPoolingLayer::GetLayerShape() const
{
	return
	{
		m_InputHeight, m_InputWidth, m_InputDepth,
		m_InputHeight / m_PoolingHeight, m_InputWidth / m_PoolingWidth, m_InputDepth
	};
}

std::string MaxPoolingLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " <<
		m_InputHeight << " " << m_InputWidth << " " << m_InputDepth << " " <<
		m_PoolingHeight << " " << m_PoolingWidth
		<< " ]";

	return ss.str();
}

void MaxPoolingLayer::FromString(const std::string& data)
{
	std::size_t numsStartPos = data.find(']');
	assert(data[0] == '[' && numsStartPos != std::string::npos && "Invalid hyperparameter format.");
	numsStartPos += 1;

	std::string hyperparams = data.substr(1, numsStartPos - 2);

	std::stringstream ss(hyperparams);

	ss >> m_InputHeight;
	ss >> m_InputWidth;
	ss >> m_InputDepth;
	ss >> m_PoolingHeight;
	ss >> m_PoolingWidth;
}

std::string MaxPoolingLayer::ToDebugString() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << "Input shape (" << shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << ")\n";
	ss << "Output shape (" << shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << ")\n";

	return ss.str();
}

std::string MaxPoolingLayer::Summarize() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << ClassName() << ":\t Input: (" <<
		shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << "), Output: (" <<
		shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << "), " <<
		"Pooling: (" << m_PoolingHeight << ", " << m_PoolingWidth << ")" <<
		", # learnable parameters: " << 0;

	return ss.str();
}

namespace_end