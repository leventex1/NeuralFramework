#pragma once
#include <Mogi.h>
#include <MogiDataset.h>
#include "CostFunctionFactory.h"


class Trainer
{
public:
	Trainer(
		mogi::Model* model,
		mogi::dataset::Dataset* training,
		mogi::dataset::Dataset* testing,
		CostFunctionFactory costFunctionFactory,
		bool useDevice
	);

	virtual void Train(
		size_t epochs,
		float startLearningRate,
		float endLearningRate
	);

	// Returns the average cost and set successRate. If there is no successRate in training set it to -1.
	virtual float Validate(float* successRate=nullptr) const = 0;  

protected:
	mogi::Model* m_Model;
	mogi::dataset::Dataset* m_TrainingDataset;
	mogi::dataset::Dataset* m_TestingDataset;
	CostFunctionFactory m_CostFunctionFactory;
	bool m_UseDeivce = false;
};
