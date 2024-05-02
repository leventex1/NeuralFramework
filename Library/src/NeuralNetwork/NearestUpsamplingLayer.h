#pragma once
#include "Layer.h"
#include "ActivationF.h"
#include "Initializer.h"


namespace_start

class LIBRARY_API NearestUpsamplingLayer : public Layer
{
public:
	NearestUpsamplingLayer(
		size_t inputHeight, size_t inputWidth, size_t inputDepth,
		size_t upsamplingHeight, size_t upsamplingWidth);
	NearestUpsamplingLayer(const std::string& fromString);

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
	virtual std::string GetSepcialParams() const { return "Upsample: (" + std::to_string(m_UpsamplingHeight) + ", " + std::to_string(m_UpsamplingWidth) + "), Stride: 1"; };

	virtual void FromString(const std::string& data);

	static std::string ClassName() { return "NearestUpsamplingLayer"; }
private:
	size_t m_InputHeight, m_InputWidth, m_InputDepth;
	size_t m_UpsamplingHeight, m_UpsamplingWidth;
};

namespace_end
