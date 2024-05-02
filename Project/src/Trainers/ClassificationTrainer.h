#pragma once
#include <Mogi.h>
#include <MogiDataset.h>
#include "Trainer.h"
#include "CostFunctionFactory.h"


class ClassificationTrainer : public Trainer
{
public:
	ClassificationTrainer(
		mogi::Model* model,
		mogi::dataset::Dataset* training,
		mogi::dataset::Dataset* testing,
		CostFunctionFactory costFunctionFactory,
		bool useDevice
	);

	virtual float Validate(float* succesRate) const;
};
