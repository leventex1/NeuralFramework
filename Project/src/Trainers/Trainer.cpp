#include <iostream>
#include <iomanip>
#include "Timer.h"
#include "Trainer.h"


Trainer::Trainer(
	mogi::Model* model,
	mogi::dataset::Dataset* training,
	mogi::dataset::Dataset* testing,
	CostFunctionFactory costFunctionFactory,
	bool useDevice
)
	: m_Model(model), m_TrainingDataset(training), m_TestingDataset(testing), m_CostFunctionFactory(costFunctionFactory), m_UseDeivce(useDevice)
{
	int errorAt = -1;
	if (!m_Model->IsModelCorrect(&errorAt))
	{
		std::cout << "Model is not defined correctly! Error at layer: (" << errorAt << ")" << std::endl;
		throw -1;
	}
}

void Trainer::Train(
	size_t epochs,
	float startLearningRate,
	float endLearningRate
)
{
	const size_t loadingBarTotal = 20;  // total number of steps for the loading bar.

	if (!m_TrainingDataset->IsModelCompatible(*m_Model))
	{
		std::cout << "The dataset is not compatible with the data!" << std::endl;
		return;
	}

	if (m_UseDeivce)
	{
		m_Model->ToDevice();
	}

	for (size_t e = 0; e < epochs; e++)
	{
		float learningRate = startLearningRate + ((float)e / (float)epochs) * (endLearningRate - startLearningRate);
		float avgLoss = 0.0f;
		float avgStep = 0.0f;

		std::cout << "Epoch " << std::setw(7) << "(" + std::to_string(e + 1) + "/" + std::to_string(epochs) + ") ";
		std::cout << "[" << std::string(loadingBarTotal, ' ') << "] " << "0%" << " loss: " << avgLoss << " step: " << avgStep << "[ms]";

		Timer timer;
		for (size_t t = 0; t < m_TrainingDataset->GetEpochSize(); t += 1)
		{
			Timer stepTimer;
			mogi::dataset::Sample trainingSample = m_TrainingDataset->GetSample();
			m_TrainingDataset->Next();

			if (m_UseDeivce)
			{
				trainingSample.Input.ToDevice();
				trainingSample.Label.ToDevice();
			}

			mogi::CostFunction loss = m_CostFunctionFactory.Build(trainingSample.Label);
			m_Model->BackPropagation(trainingSample.Input, loss, learningRate, t);

			mogi::Tensor3D output = m_Model->FeedForward(trainingSample.Input);
			avgLoss *= t;
			avgLoss += loss.Cost(output);
			avgLoss /= t + 1;

			float stepDuration = stepTimer.GetTime() * 1000;
			avgStep *= t;
			avgStep += stepDuration;
			avgStep /= t + 1;

			if ((t % std::max(size_t(1), (m_TrainingDataset->GetEpochSize() / 100)) == 0) || (t == (m_TrainingDataset->GetEpochSize() - 1)))
			{
				float status = std::min((float)t / (float)(m_TrainingDataset->GetEpochSize() - 1), 1.0f);
				size_t loadingStatus = status * loadingBarTotal;
				std::cout << "\r" << "Epoch " << std::setw(7) << "(" + std::to_string(e + 1) + "/" + std::to_string(epochs) + ") ";
				std::cout << "[" << std::string(loadingStatus, '=') << std::string(loadingBarTotal - loadingStatus, ' ') << "] ";
				std::cout << (size_t)(status * 100) << "%" << " loss: " << avgLoss << " step: " << (int)avgStep << "[ms]";
			}
		}
		double duration = timer.GetTime();
		std::cout << " duration: " << duration << "[s]";

		float successRate = 0.0f;
		float averageCost = Validate(&successRate);
		
		std::cout << " Average cost: " << averageCost << " " << (successRate > -0.9f ? "Success rate: " + std::to_string(successRate) : "") << std::endl;

		m_TrainingDataset->Shuffle();
	}
}