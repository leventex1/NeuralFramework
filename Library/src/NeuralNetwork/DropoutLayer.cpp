#include "DropoutLayer.h"
#include <assert.h>
#include <sstream>


namespace_start

DropoutLayer::DropoutLayer(
	size_t inputHeight, size_t inputWidth, size_t inputDepth, float dropoutRate
) : m_InputHeight(inputHeight), m_InputWidth(inputWidth), m_InputDepth(inputDepth), m_DropoutRate(dropoutRate)
{
}

DropoutLayer::DropoutLayer(const std::string& fromString)
{
	FromString(fromString);
}

Tensor3D DropoutLayer::FeedForward(const Tensor3D& inputs)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	return inputs;
}

Tensor3D DropoutLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFucntion, float learningRate, size_t t)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	LayerShape layerShape = GetLayerShape();

	Tensor3D dropOutTensor(layerShape.OutputRows, layerShape.OutputCols, layerShape.OutputDepth, 0.0f, inputs.IsOnDevice());
	Tensor3D output = DropOut(inputs, m_DropoutRate, &dropOutTensor);

	Tensor3D costs = NextLayer ?
		NextLayer->BackPropagation(output, costFucntion, learningRate, t) :
		output;

	assert(layerShape.OutputRows == costs.GetRows() &&
		layerShape.OutputCols == costs.GetCols() &&
		layerShape.OutputDepth == costs.GetDepth()
		&& "Costs shape not match!");

	costs.Mult(dropOutTensor);

	return costs;
}

LayerShape DropoutLayer::GetLayerShape() const
{
	return
	{
		m_InputHeight, m_InputWidth, m_InputDepth,
		m_InputHeight, m_InputWidth, m_InputDepth
	};
}

std::string DropoutLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " <<
		m_InputHeight << " " << m_InputWidth << " " << m_InputDepth << " " <<
		m_DropoutRate
		<< " ]";

	return ss.str();
}

void DropoutLayer::FromString(const std::string& data)
{
	std::size_t numsStartPos = data.find(']');
	assert(data[0] == '[' && numsStartPos != std::string::npos && "Invalid hyperparameter format.");
	numsStartPos += 1;

	std::string hyperparams = data.substr(1, numsStartPos - 2);

	std::stringstream ss(hyperparams);

	ss >> m_InputHeight;
	ss >> m_InputWidth;
	ss >> m_InputDepth;
	ss >> m_DropoutRate;
}

std::string DropoutLayer::ToDebugString() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << "Input shape (" << shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << ")\n";
	ss << "Output shape (" << shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << ")\n";

	return ss.str();
}

std::string DropoutLayer::Summarize() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << ClassName() << ":\t Input: (" <<
		shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << "), Output: (" <<
		shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << "), " <<
		"DropoutRate: " + std::to_string(m_DropoutRate) <<
		", # learnable parameters: " << 0;

	return ss.str();
}

namespace_end