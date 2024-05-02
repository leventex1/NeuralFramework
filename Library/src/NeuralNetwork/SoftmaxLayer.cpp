#include "SoftmaxLayer.h"
#include <assert.h>
#include <sstream>


namespace_start

SoftmaxLayer::SoftmaxLayer(size_t inputNodes) : m_InputNodes(inputNodes) { }

SoftmaxLayer::SoftmaxLayer(const std::string& fromString)
{
	FromString(fromString);
}

Tensor3D SoftmaxLayer::FeedForward(const Tensor3D& inputs)
{
	assert(inputs.GetRows() == m_InputNodes && 
			inputs.GetCols() == 1 &&
			inputs.GetDepth() == 1
		&& "Invalid input params!");

	Tensor3D input = inputs;
	input.ToHost();

	float sum = 0.0f;
	for (size_t i = 0; i < m_InputNodes; i++)
	{
		sum += exp(input.GetAt(i, 0, 0));
	}

	Tensor3D output = Map(input, [sum](float v) -> float { return exp(v) / sum; });

	if (inputs.IsOnDevice())
		output.ToDevice();

	return output;
}

Tensor3D SoftmaxLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate, size_t t)
{
	assert(inputs.GetRows() == m_InputNodes &&
		inputs.GetCols() == 1 &&
		inputs.GetDepth() == 1
		&& "Invalid input params!");

	Tensor3D input = inputs;
	input.ToHost();

	float sum = 0.0f;
	for (size_t i = 0; i < m_InputNodes; i++)
	{
		sum += exp(input.GetAt(i, 0, 0));
	}
	Tensor3D output = Map(input, [sum](float v) -> float { return exp(v) / sum; });

	if (inputs.IsOnDevice())
		output.ToDevice();

	Tensor3D costs = NextLayer ?
								NextLayer->BackPropagation(output, costFunction, learningRate, t) :
								costFunction.DiffCost(output);

	assert(costs.GetRows() == m_InputNodes &&
		costs.GetCols() == 1 &&
		costs.GetDepth() == 1
		&& "Invalid input params!");

	costs.ToHost();
	output.ToHost();

	Tensor3D gradCosts = Tensor3D(m_InputNodes, 1, 1);

	for (size_t i = 0; i < m_InputNodes; i++)
	{
		float value = 0.0f;
		for (size_t j = 0; j < m_InputNodes; j++)
		{
			if (i == j)
			{
				value += costs.GetAt(j, 0, 0) * output.GetAt(j, 0, 0) * (1.0f - output.GetAt(j, 0, 0));
			}
			else
			{
				value += costs.GetAt(j, 0, 0) * output.GetAt(j, 0, 0) * output.GetAt(i, 0, 0) * -1.0f;
			}
		}
		gradCosts.SetAt(i, 0, 0, value);
	}

	if (inputs.IsOnDevice())
		gradCosts.ToDevice();

	return gradCosts;
}

LayerShape SoftmaxLayer::GetLayerShape() const
{
	return 
	{
		m_InputNodes, 1, 1,
		m_InputNodes, 1, 1
	};
}

std::string SoftmaxLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " << m_InputNodes << " ]";

	return ss.str();
}

std::string SoftmaxLayer::ToDebugString() const
{
	return "Soft max layer, input nodes: " + std::to_string(m_InputNodes);
}

std::string SoftmaxLayer::Summarize() const
{
	std::stringstream ss;

	ss << ClassName() << ":\t # input nodes: " << m_InputNodes <<
	", # learnable parameters : " << 0;

	return ss.str();
}

void SoftmaxLayer::FromString(const std::string& data)
{
	std::size_t numsStartPos = data.find(']');
	assert(data[0] == '[' && numsStartPos != std::string::npos && "Invalid hyperparameter format.");
	numsStartPos += 1;

	std::string hyperparams = data.substr(1, numsStartPos - 2);
	std::stringstream ss(hyperparams);

	std::string inputNodesStr;
	ss >> inputNodesStr;

	try
	{
		m_InputNodes = std::stoi(inputNodesStr);
	}
	catch (...)
	{
		assert(false && "Invalid number.");
	}
}

namespace_end