#pragma once
#include <vector>
#include <memory>
#include <assert.h>
#include <string>
#include "Core.h"
#include "../Math/Tensor3D.h"
#include "CostF.h"
#include "ActivationF.h"

#include "Optimizer.h"


namespace_start

struct LayerShape
{
	size_t InputRows, InputCols, InputDepth;
	size_t OutputRows, OutputCols, OutputDepth;
};

class LIBRARY_API Layer
{
public:
	virtual ~Layer() { }

	virtual void ToHost() = 0;
	virtual void ToDevice() = 0;

	/*
		Implement gradient optimizer algorithms for learnable parameters, Like adam, or SGD optimizer.
		Implement! the save and loading of these optimizers witht the layer itself. 
		With some optimizers like adam, there is a state that needs to be stored between training sessions.
	*/
	virtual void InitOptimizer(OptimizerFactory optimizerFactory) { };

	virtual Tensor3D FeedForward(const Tensor3D& inputs) = 0;

	/*
		Recursive backpropagation learning algorithm.
		First calculates the feedforward of the current layer.
		Than calls the nextLayer's backpropagation with the output as the next input.
		Than calculate the updated parameters respect to the returned costs.

		Returns the derivated cost respect to the inputs.

		t: learning time step.
	*/
	virtual Tensor3D BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate, size_t t) = 0;

	virtual LayerShape GetLayerShape() const = 0;

	virtual std::string GetName() const = 0;
	virtual std::string ToString() const = 0;
	virtual std::string ToDebugString() const { assert(false && "Debug string not implemented!"); return ""; }
	virtual std::string Summarize() const { assert(false && "Summarize not implemented!"); return "Unknown layer"; }

	virtual ActivationFunciton GetActivationFunction() const = 0;
	virtual size_t GetLearnableParams() const = 0;
	virtual std::string GetSepcialParams() const = 0;

	virtual void FromString(const std::string& data) = 0;

	std::shared_ptr<Layer> NextLayer;
};

namespace_end
