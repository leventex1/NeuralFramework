#include "ConvolutionalLayer.h"
#include <assert.h>
#include <sstream>
#include <iostream>


namespace_start

ConvolutionalLayer::ConvolutionalLayer(
	size_t inputHeight, size_t inputWidth, size_t inputDepth,
	size_t kernelHeight, size_t kernelWidth, size_t numKernels,
	size_t padding, ActivationFunciton activationFunction, Initializer initializer,
	bool isUseBias
) : m_ActivationFunction(activationFunction), m_Padding(padding), m_InputWidth(inputWidth), m_InputHeight(inputHeight), m_InputDepth(inputDepth), m_NumKernels(numKernels), m_IsUseBias(isUseBias)
{
	assert(inputHeight == inputWidth && "Different input size ratio not supported!");
	assert(kernelHeight == kernelWidth && "Different kernel size ratio not supported!");

	size_t kernelDepth = numKernels * inputDepth;
	size_t outputHeight = CalcConvSize(inputHeight, kernelHeight, 1, padding);
	size_t outputWidth = CalcConvSize(inputWidth, kernelWidth, 1, padding);

	if (m_IsUseBias)
	{
		m_Bias = Tensor3D(outputHeight, outputWidth, numKernels, initializer.Init);
	}
	m_Kernels = Tensor3D(kernelHeight, kernelWidth, kernelDepth, initializer.Init);
}

ConvolutionalLayer::ConvolutionalLayer(const std::string& fromString)
{
	FromString(fromString);
}

void ConvolutionalLayer::ToHost()
{
	m_Kernels.ToHost();
	if (m_IsUseBias)
		m_Bias.ToHost();

	if(m_KernelOptimizer)
		m_KernelOptimizer->ToHost();
	if(m_BiasOptimizer)
		m_BiasOptimizer->ToHost();
}

void ConvolutionalLayer::ToDevice()
{
	m_Kernels.ToDevice();
	if (m_IsUseBias)
		m_Bias.ToDevice();

	if(m_KernelOptimizer)
		m_KernelOptimizer->ToDevice();
	if(m_BiasOptimizer)
		m_BiasOptimizer->ToDevice();
}

void ConvolutionalLayer::InitOptimizer(OptimizerFactory optimizerFactory)
{
	m_KernelOptimizer = optimizerFactory.Get(m_Kernels.GetSize());
	if(m_IsUseBias)
		m_BiasOptimizer = optimizerFactory.Get(m_Bias.GetSize());;
}

Tensor3D ConvolutionalLayer::FeedForward(const Tensor3D& inputs)
{
	assert(inputs.GetRows() == m_InputHeight && inputs.GetCols() == m_InputWidth && inputs.GetDepth() == m_InputDepth 
		&& "Invalid input shape!");

	LayerShape layerShape = GetLayerShape();

	Tensor3D output = Tensor3D(layerShape.OutputRows, layerShape.OutputCols, layerShape.OutputDepth, 0.0f, inputs.IsOnDevice());

	for (int d = 0; d < layerShape.OutputDepth; d++)
	{
		Tensor3D kernelBlock = CreateWatcher(m_Kernels, d * layerShape.InputDepth, layerShape.InputDepth);
		Tensor2D outputSlice = CreateWatcher(output, d);
		Convolution(outputSlice, inputs, kernelBlock, 1, m_Padding);
	}

	if (m_IsUseBias)
	{
		output.Add(m_Bias);
	} 
	m_ActivationFunction.MapActivation(&output);
	return output;
}

Tensor3D ConvolutionalLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate, size_t t)
{
	assert(inputs.GetRows() == m_InputHeight && inputs.GetCols() == m_InputWidth && inputs.GetDepth() == m_InputDepth
		&& "Invalid input shape!");

	LayerShape layerShape = GetLayerShape();
	
	Tensor3D filterMap = Tensor3D(layerShape.OutputRows, layerShape.OutputCols, layerShape.OutputDepth, 0.0f, inputs.IsOnDevice());

	for (int d = 0; d < layerShape.OutputDepth; d++)
	{
		Tensor3D kernelBlock = CreateWatcher(m_Kernels, d * layerShape.InputDepth, layerShape.InputDepth);
		Tensor2D outputSlice = CreateWatcher(filterMap, d);
		Convolution(outputSlice, inputs, kernelBlock, 1, m_Padding);
	}
	if (m_IsUseBias)
	{
		filterMap.Add(m_Bias);
	}

	Tensor3D output = filterMap;
	m_ActivationFunction.MapActivation(&output);


	Tensor3D costs = NextLayer ?
								NextLayer->BackPropagation(output, costFunction, learningRate, t) :
								costFunction.DiffCost(output);

	assert(costs.GetRows() == layerShape.OutputRows && costs.GetCols() == layerShape.OutputCols && costs.GetDepth() == layerShape.OutputDepth
		&& "Invalid cost shape!");


	m_ActivationFunction.MapDiffActivation(&filterMap);
	costs.Mult(filterMap);  // costs -> Gradient.

	Tensor3D& gradBias = costs;

	Tensor3D gradKernel = Tensor3D(m_Kernels.GetRows(), m_Kernels.GetCols(), m_Kernels.GetDepth(), 0.0f, m_Kernels.IsOnDevice());

	size_t gradInputPadding = m_Kernels.GetRows() - 1 - m_Padding;
	Tensor3D gradInput = Tensor3D(layerShape.InputRows, layerShape.InputCols, layerShape.InputDepth, 0.0f, inputs.IsOnDevice());

	for (size_t d = 0; d < layerShape.OutputDepth; d++)
	{
		Tensor2D grad = CreateWatcher(costs, d);

		Tensor3D gradKernelBlock = CreateWatcher(gradKernel, d * layerShape.InputDepth, layerShape.InputDepth);
		Convolution(gradKernelBlock, inputs, grad, 1, m_Padding);

		Tensor3D kernelBlock = CreateWatcher(m_Kernels, d * layerShape.InputDepth, layerShape.InputDepth);
		ConvolutionKernelFlip(gradInput, grad, kernelBlock, 1, gradInputPadding);
	}

	m_KernelOptimizer->Update(&m_Kernels, &gradKernel, learningRate);
	if (m_IsUseBias)
	{
		m_BiasOptimizer->Update(&m_Bias, &gradBias, learningRate);
	}

	return gradInput;
}

LayerShape ConvolutionalLayer::GetLayerShape() const
{
	size_t outputHeight = CalcConvSize(m_InputHeight, m_Kernels.GetRows(), 1, m_Padding);
	size_t outputWidth = CalcConvSize(m_InputWidth, m_Kernels.GetCols(), 1, m_Padding);

	return
	{
		m_InputHeight, m_InputWidth, m_InputDepth,
		outputHeight, outputWidth, m_NumKernels
	};
}

std::string ConvolutionalLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " <<
		m_InputWidth << " " << m_InputHeight << " " << m_InputDepth << " " << 
		m_Kernels.GetRows() << " " << m_Kernels.GetCols() << " " << m_NumKernels << " " << m_Padding << " " << m_IsUseBias << " " <<
		m_ActivationFunction.Name << " ( " << m_ActivationFunction.Params << " )" <<
	" ]";

	for (size_t t = 0; t < m_Kernels.GetSize(); t++)
	{
		ss << " " << m_Kernels.GetData()[t];
	}
	if (m_IsUseBias)
	{
		for (size_t t = 0; t < m_Bias.GetSize(); t++)
		{
			ss << " " << m_Bias.GetData()[t];
		}
	}

	ss << " { " << m_KernelOptimizer->GetName() << " " << m_KernelOptimizer->ToString() << "}";
	if(m_IsUseBias)
		ss << " { " << m_BiasOptimizer->GetName() << " " << m_BiasOptimizer->ToString() << "}";

	return ss.str();
}

void ConvolutionalLayer::FromString(const std::string& data)
{
	std::size_t numsStartPos = data.find(']');
	assert(data[0] == '[' && numsStartPos != std::string::npos && "Invalid hyperparameter format.");
	numsStartPos += 1;

	std::string hyperparams = data.substr(1, numsStartPos - 2);
	std::size_t acivationParamsStart = hyperparams.find('(');
	std::size_t acivationParamsEnd = hyperparams.find(')');
	assert(acivationParamsStart != std::string::npos && acivationParamsEnd != std::string::npos && "Invalid activation function params format.");

	std::stringstream ss(hyperparams);
	std::string activationFParamsStr = hyperparams.substr(acivationParamsStart + 2, acivationParamsEnd - acivationParamsStart - 3);

	size_t kernelHeight, kernelWidth;
	std::string activationName;

	ss >> m_InputWidth >> m_InputHeight >> m_InputDepth >> kernelHeight >> kernelWidth >> m_NumKernels >> m_Padding >> m_IsUseBias;
	ss >> activationName;

	size_t kernelDepth = m_NumKernels * m_InputDepth;
	size_t outputHeight = CalcConvSize(m_InputHeight, kernelHeight, 1, m_Padding);
	size_t outputWidth = CalcConvSize(m_InputWidth, kernelWidth, 1, m_Padding);

	m_Bias = Tensor3D(outputHeight, outputWidth, m_NumKernels);
	m_Kernels = Tensor3D(kernelHeight, kernelWidth, kernelDepth);
	m_ActivationFunction = GetActivationFunctionByName(activationName, activationFParamsStr);

	std::istringstream iss(data.substr(numsStartPos));

	for (size_t t = 0; t < m_Kernels.GetSize(); t++)
	{
		iss >> m_Kernels.GetData()[t];
	}
	if (m_IsUseBias)
	{
		for (size_t t = 0; t < m_Bias.GetSize(); t++)
		{
			iss >> m_Bias.GetData()[t];
		}
	}

	std::string remaining;
	std::getline(iss, remaining);

	size_t kernelOptimizerStart = remaining.find('{');
	size_t kernelOptimizerEnd = remaining.find('}');
	size_t biasOptimizerStart = remaining.find('{', kernelOptimizerEnd);
	size_t biasOptimizerEnd = remaining.find('}', biasOptimizerStart);

	OptimizerFactory optimizerFactory;
	std::string kernelOptimizerStr = remaining.substr(kernelOptimizerStart + 1, kernelOptimizerEnd - kernelOptimizerStart - 2);
	m_KernelOptimizer = optimizerFactory.Get(kernelOptimizerStr);
	if (m_IsUseBias)
	{
		std::string biasOptimizerStr = remaining.substr(biasOptimizerStart + 1, biasOptimizerEnd - biasOptimizerStart - 2);
		m_BiasOptimizer = optimizerFactory.Get(biasOptimizerStr);
	}
}

std::string ConvolutionalLayer::ToDebugString() const
{
	std::stringstream ss;
	ss << "Weights (" << m_Kernels.GetRows() << ", " << m_Kernels.GetCols() << ", " << m_Kernels.GetDepth() << ")\n";
	ss << m_Kernels.ToString() << "\n";
	ss << "Bias (" << m_Bias.GetRows() << ", " << m_Bias.GetCols() << ", " << m_Bias.GetDepth() << ")\n";
	ss << m_Bias.ToString() << "\n";
	return ss.str();
}

std::string ConvolutionalLayer::Summarize() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << ClassName() << ":\t Input: (" <<
		shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << "), Output: (" <<
		shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << "), " <<
		"# kernels: " << m_NumKernels << ", Padding: " << m_Padding <<
		", Activation: " << m_ActivationFunction.Name << "(" << m_ActivationFunction.Params << "), " <<
		"# learnable parameters: " << m_Kernels.GetSize() + m_Bias.GetSize();

	return ss.str();
}

namespace_end