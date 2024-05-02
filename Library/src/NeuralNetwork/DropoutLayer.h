#pragma once
#include "Layer.h"


namespace_start

class LIBRARY_API DropoutLayer : public Layer
{
public:
	DropoutLayer(
		size_t inputHeight, size_t inputWidth, size_t inputDepth, float dropoutRate);
	DropoutLayer(const std::string& fromString);

	virtual void ToHost() override { }
	virtual void ToDevice() override { }

	virtual Tensor3D FeedForward(const Tensor3D& inputs);
	virtual Tensor3D BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate, size_t t);

	virtual LayerShape GetLayerShape() const;

	virtual std::string GetName() const { return ClassName(); }
	virtual std::string ToString() const;
	virtual std::string ToDebugString() const;
	virtual std::string Summarize() const;

	virtual ActivationFunciton GetActivationFunction() const { return ActivationFunciton(); }
	virtual size_t GetLearnableParams() const { return 0; };
	virtual std::string GetSepcialParams() const { return "DropoutRate: " + std::to_string(m_DropoutRate); };

	virtual void FromString(const std::string& data);

	static std::string ClassName() { return "DropoutLayer"; }
private:
	size_t m_InputHeight, m_InputWidth, m_InputDepth;
	float m_DropoutRate;
};

namespace_end
