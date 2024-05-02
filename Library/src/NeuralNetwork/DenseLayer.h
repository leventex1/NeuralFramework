#pragma once
#include "Layer.h"
#include "ActivationF.h"
#include "Initializer.h"


namespace_start

class LIBRARY_API DenseLayer : public Layer
{
public:
	DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction);
	DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction, Initializer initializer);
	DenseLayer(const std::string& fromString);

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
	virtual size_t GetLearnableParams() const override { return m_Weights.GetSize() + m_Bias.GetSize(); };
	virtual std::string GetSepcialParams() const override { return ("Optimizer: " + m_WeightsOptimizer->GetName()); };

	virtual void FromString(const std::string& data) override;

	static std::string ClassName() { return "DenseLayer"; }
private:
	Tensor2D m_Weights;
	Tensor2D m_Bias;
	ActivationFunciton m_ActivationFunction;

	std::unique_ptr<Optimizer> m_WeightsOptimizer;
	std::unique_ptr<Optimizer> m_BiasOptimizer;
};

namespace_end
