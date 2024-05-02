#pragma once
#include "Layer.h"
#include "ActivationF.h"
#include "Initializer.h"


namespace_start

class LIBRARY_API ConvolutionalLayer : public Layer
{
public:
	ConvolutionalLayer(
		size_t inputHeight, size_t inputWidth, size_t inputDepth,
		size_t kernelHeight, size_t kernelWidth, size_t numKernels,
		size_t padding, ActivationFunciton activationFunction, Initializer initializer,
		bool isUseBias = true
	);
	ConvolutionalLayer(const std::string& fromString);

	virtual void ToHost() override;
	virtual void ToDevice() override;

	virtual void InitOptimizer(OptimizerFactory optimizerFactory) override;
	virtual Tensor3D FeedForward(const Tensor3D& inputs) override;
	virtual Tensor3D BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate, size_t t) override;

	virtual LayerShape GetLayerShape() const override;

	virtual std::string GetName() const override { return ClassName(); }
	virtual std::string ToString() const override;
	virtual std::string ToDebugString() const override;
	virtual std::string Summarize() const override;

	virtual ActivationFunciton GetActivationFunction() const override { return m_ActivationFunction; }
	virtual size_t GetLearnableParams() const override { return m_Kernels.GetSize() + (m_IsUseBias ? m_Bias.GetSize() : 0); };
	virtual std::string GetSepcialParams() const override { return "Kernel: (" + std::to_string(m_Kernels.GetRows()) + ", " + std::to_string(m_Kernels.GetCols()) + ", " + std::to_string(m_InputDepth) + " * " + std::to_string(m_NumKernels) + "), Padding: " + std::to_string(m_Padding) + ", Stride: 1, Optimizer: " + m_KernelOptimizer->GetName(); };

	virtual void FromString(const std::string& data) override;

	static std::string ClassName() { return "ConvolutionalLayer"; }
private:
	Tensor3D m_Kernels;  // k x k x (#2Dinputs * n)
	Tensor3D m_Bias;  // Wo x Ho x n
	ActivationFunciton m_ActivationFunction;
	 
	size_t m_InputWidth, m_InputHeight, m_InputDepth;
	size_t m_NumKernels;
	size_t m_Padding;

	bool m_IsUseBias;

	std::unique_ptr<Optimizer> m_KernelOptimizer;
	std::unique_ptr<Optimizer> m_BiasOptimizer;
};

namespace_end
