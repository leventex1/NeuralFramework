#include "DenseLayer.h"
#include <assert.h>
#include <sstream>
#include <iostream>

#include "../Math/Operation.h"


namespace_start

DenseLayer::DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction)
	: m_ActivationFunction(activationFunction)
{
	m_Weights = Random2D(outputNodes, inputNodes, -1.0f, 1.0f, true);
	m_Bias = Random2D(outputNodes, 1, -1.0f, 1.0f, true);
	m_Weights.ToHost();
	m_Bias.ToHost();
}

DenseLayer::DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction, Initializer initializer)
	: m_ActivationFunction(activationFunction)
{
	m_Weights = Tensor2D(outputNodes, inputNodes, initializer.Init);
	m_Bias = Tensor2D(outputNodes, 1, initializer.Init);
}

DenseLayer::DenseLayer(const std::string& fromString)
{
	FromString(fromString);
}

void DenseLayer::ToHost()
{
	m_Weights.ToHost();
	m_Bias.ToHost();

	if(m_WeightsOptimizer)
		m_WeightsOptimizer->ToHost();
	if(m_BiasOptimizer)
		m_BiasOptimizer->ToHost();
}

void DenseLayer::ToDevice()
{
	m_Weights.ToDevice();
	m_Bias.ToDevice();

	if(m_WeightsOptimizer)
		m_WeightsOptimizer->ToDevice();
	if(m_BiasOptimizer)
		m_BiasOptimizer->ToDevice();
}

void DenseLayer::InitOptimizer(OptimizerFactory optimizerFactory)
{
	m_WeightsOptimizer = optimizerFactory.Get(m_Weights.GetSize());
	m_BiasOptimizer = optimizerFactory.Get(m_Bias.GetSize());
}

Tensor3D DenseLayer::FeedForward(const Tensor3D& inputs)
{
	assert(inputs.GetDepth() == 1 && "Dense layer's input is 1 Tensor2D!");

	Tensor2D input = SliceTensor(inputs, 0);

	Tensor2D sum = MatrixMult(m_Weights, input);
	sum.Add(m_Bias);
	m_ActivationFunction.MapActivation(&sum);

	return Tensor3D(sum.GetRows(), sum.GetCols(), 1, std::move(sum));
}

Tensor3D DenseLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate, size_t t)
{
	if (!m_WeightsOptimizer || !m_BiasOptimizer)
		throw std::exception("No optimizer for training!");

	assert(inputs.GetDepth() == 1 && "Dense layer's input should contain a tensor with a depth of 1!");
	assert(inputs.GetCols() == 1 && "Dense layer's input should contain 1 column!");
	assert(inputs.GetRows() == m_Weights.GetCols() && "Dense layer's input should contain as many rows as the weight's cols!");

	LayerShape layerShape = GetLayerShape();

	Tensor2D input = SliceTensor(inputs, 0);

	Tensor2D sum = MatrixMult(m_Weights, input);
	sum.Add(m_Bias);

	Tensor3D output = Tensor3D(layerShape.OutputRows, layerShape.OutputCols, layerShape.OutputDepth, (const float*)sum.GetData(), sum.IsOnDevice());
	m_ActivationFunction.MapActivation(&output);

	Tensor3D costs = NextLayer ? 
								NextLayer->BackPropagation(output, costFunction, learningRate, t) : 
								costFunction.DiffCost(output);

	assert(inputs.GetDepth() == 1 && "Dense layer's cost should contain a tensor with a depth of 1!");
	assert(costs.GetCols() == 1 && "Dense layer's cost should contain 1 column!");
	assert(costs.GetRows() == m_Weights.GetRows() && "Dense layer's cost should contain as many rows as the weight's rows!");

	Tensor2D cost = CreateWatcher(costs, 0);
	m_ActivationFunction.MapDiffActivation(&sum);
	Tensor2D diffSum = Mult(cost, sum);

	Tensor2D gradWeights = MatrixMultRightTranspose(diffSum, input);
	Tensor2D& gradBiases = diffSum;
	Tensor2D gradCosts = MatrixMultLeftTranspose(m_Weights, diffSum);

	m_WeightsOptimizer->Update(&m_Weights, &gradWeights, learningRate);
	m_BiasOptimizer->Update(&m_Bias, &gradBiases, learningRate);

	return Tensor3D(layerShape.InputRows, layerShape.InputCols, 1, std::move(gradCosts));
}

LayerShape DenseLayer::GetLayerShape() const
{
	return 
	{
		m_Weights.GetCols(), 1, 1,
		m_Weights.GetRows(), 1, 1
	};
}

std::string DenseLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " << 
		m_Weights.GetCols() << " " << m_Weights.GetRows() << " " << 
		m_ActivationFunction.Name << " ( " << m_ActivationFunction.Params << " )" << 
	" ]";

	for (size_t t = 0; t < m_Weights.GetSize(); t++)
	{
		ss << " " << m_Weights.GetData()[t];
	}
	for (size_t t = 0; t < m_Bias.GetSize(); t++)
	{
		ss << " " << m_Bias.GetData()[t];
	}

	ss << " { " << m_WeightsOptimizer->GetName() << " " << m_WeightsOptimizer->ToString() << "}";
	ss << " { " << m_BiasOptimizer->GetName() << " " << m_BiasOptimizer->ToString() << "}";

	return ss.str();
}

void DenseLayer::FromString(const std::string& data)
{
	std::size_t numsStartPos = data.find(']');
	assert(data[0] == '[' && numsStartPos != std::string::npos && "Invalid hyperparameter format.");
	numsStartPos += 1;

	std::string hyperparams = data.substr(1, numsStartPos - 2);
	std::size_t acivationParamsStart = hyperparams.find('(');
	std::size_t acivationParamsEnd = hyperparams.find(')');
	assert(acivationParamsStart != std::string::npos && acivationParamsEnd != std::string::npos && "Invalid activation function params format.");

	std::stringstream ss(hyperparams);

	std::string activationFStr;
	std::string activationFParamsStr = hyperparams.substr(acivationParamsStart + 2, acivationParamsEnd - acivationParamsStart - 3);
	size_t inputNodes, outputNodes;
	ss >> inputNodes;
	ss >> outputNodes;
	ss >> activationFStr;

	m_Weights = Tensor2D(outputNodes, inputNodes);
	m_Bias = Tensor2D(outputNodes, 1);
	m_ActivationFunction = GetActivationFunctionByName(activationFStr, activationFParamsStr);

	std::istringstream iss(data.substr(numsStartPos));

	for (size_t i = 0; i < m_Weights.GetSize(); i++)
	{
		iss >> m_Weights.GetData()[i];
	}
	for (size_t i = 0; i < m_Bias.GetSize(); i++)
	{
		iss >> m_Bias.GetData()[i];
	}

	std::string remaining;
	std::getline(iss, remaining);

	size_t weightOptimizerStart = remaining.find('{');
	size_t weightOptimizerEnd = remaining.find('}');
	size_t biasOptimizerStart = remaining.find('{', weightOptimizerEnd);
	size_t biasOptimizerEnd = remaining.find('}', biasOptimizerStart);
	std::string weightOptimizerStr = remaining.substr(weightOptimizerStart + 1, weightOptimizerEnd - weightOptimizerStart - 2);
	std::string biasOptimizerStr = remaining.substr(biasOptimizerStart + 1, biasOptimizerEnd - biasOptimizerStart - 2);

	OptimizerFactory optimizerFactory;
	m_WeightsOptimizer = optimizerFactory.Get(weightOptimizerStr);
	m_BiasOptimizer = optimizerFactory.Get(biasOptimizerStr);
}

std::string DenseLayer::ToDebugString() const
{
	std::stringstream ss;
	ss << "Weights (" << m_Weights.GetRows() << ", " << m_Weights.GetCols() << ")\n";
	ss << m_Weights.ToString() << "\n";
	ss << "Bias (" << m_Bias.GetRows() << ", " << m_Bias.GetCols() << ")\n";
	ss << m_Bias.ToString() << "\n";
	return ss.str();
}

std::string DenseLayer::Summarize() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << ClassName() << ":\t Input: (" <<
	shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << "), Output: (" <<
	shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << "), " <<
	"Activation: " << m_ActivationFunction.Name << "(" << m_ActivationFunction.Params << "), " <<
	"# learnable parameters: " << m_Weights.GetSize() + m_Bias.GetSize();

	return ss.str();
}


namespace_end