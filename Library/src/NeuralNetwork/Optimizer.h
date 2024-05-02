#pragma once
#include <sstream>
#include <math.h>
#include <memory>
#include <string>

#include "Core.h"
#include "../Math/Tensor2D.h"
#include "../Math/Operation.h"


namespace_start

enum OptimizerType
{
	None=-1,
	SGD, Adam,
};


class Optimizer
{
public:
	virtual ~Optimizer() { }

	virtual void Update(Tensor* params, Tensor* gradient, float learningRate) = 0;

	virtual std::string GetName() const = 0;
	virtual std::string ToString() const = 0;
	virtual void FromString(const std::string& fromString) = 0;

	virtual void ToHost() = 0;
	virtual void ToDevice() = 0;
};


class SGDOptimizer : public Optimizer
{
public:
	SGDOptimizer(size_t numParams) { }
	SGDOptimizer(const std::string& fromString) { FromString(fromString); }
	virtual std::string GetName() const override { return "SGDOptimizer"; }
	static std::string ClassName() { return "SGDOptimizer"; }

	virtual void Update(Tensor* params, Tensor* gradient, float learningRate) override;
	virtual std::string ToString() const override;
	virtual void FromString(const std::string& fromString) override { }

	virtual void ToHost() override { }
	virtual void ToDevice() override { }
};


class AdamOptimizer : public Optimizer
{
public:
	AdamOptimizer(size_t numParams) : m_TrainingTimeStep(1), m_FirstMoments(numParams, 1), m_SecondMoments(numParams, 1) { }
	AdamOptimizer(const std::string& fromString) { FromString(fromString); }
	virtual std::string GetName() const override { return "AdamOptimizer"; }
	static std::string ClassName() { return "AdamOptimizer"; }

	virtual void Update(Tensor* params, Tensor* gradient, float learningRate) override;
	virtual std::string ToString() const override;
	virtual void FromString(const std::string& fromString) override;

	virtual void ToHost() override;
	virtual void ToDevice() override;
private:
	size_t m_TrainingTimeStep;
	Tensor2D m_FirstMoments;
	Tensor2D m_SecondMoments;
};


class OptimizerFactory
{
public:
	OptimizerFactory(OptimizerType type) : m_Type(type) { }
	OptimizerFactory() { }

	std::unique_ptr<Optimizer> Get(const std::string& fromString)
	{
		std::stringstream ss(fromString);
		std::string name;
		ss >> name;
		std::string remaining;
		std::getline(ss, remaining);

		if (name == SGDOptimizer::ClassName())		return std::make_unique<SGDOptimizer>(remaining);
		if (name == AdamOptimizer::ClassName())		return std::make_unique<AdamOptimizer>(remaining);

		throw std::exception("Unknown optimizer type!");
	}

	std::unique_ptr<Optimizer> Get(size_t numParams) const
	{
		switch (m_Type)
		{
		case OptimizerType::SGD:			return std::make_unique<SGDOptimizer>(numParams);
		case OptimizerType::Adam:			return std::make_unique<AdamOptimizer>(numParams);
		default:
			break;
		}

		throw std::exception("Unknown optimizer type!");
	}
private:
	OptimizerType m_Type = None;
};

namespace_end