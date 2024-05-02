#include <iostream>
#include "AutoencoderTrainer.h"


AutoencoderTrainer::AutoencoderTrainer(
	mogi::Model* model,
	mogi::dataset::Dataset* training,
	mogi::dataset::Dataset* testing,
	CostFunctionFactory costFunctionFactory,
	bool useDevice
) : Trainer(model, training, testing, costFunctionFactory, useDevice)
{
}

float AutoencoderTrainer::Validate(float* successRate) const
{
	if(successRate)
		*successRate = -1.0f;

	if (m_UseDeivce)
	{
		m_Model->ToDevice();
	}

	float cost = 0.0f;
	for (size_t i = 0; i < m_TestingDataset->GetEpochSize(); i++)
	{
		mogi::dataset::Sample testingSample = m_TestingDataset->GetSample();
		m_TestingDataset->Next();

		if (m_UseDeivce)
		{
			testingSample.Input.ToDevice();
		}

		mogi::Tensor3D output = m_Model->FeedForward(testingSample.Input);
		output.ToHost();

		mogi::CostFunction loss = m_CostFunctionFactory.Build(testingSample.Label);
		cost += loss.Cost(output);
	}

	return cost / m_TestingDataset->GetEpochSize();
}