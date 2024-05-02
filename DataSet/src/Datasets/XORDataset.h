#pragma once
#include "../Dataset.h"


namespace_dataset_start

class DATASET_API XORDataset : public Dataset
{
public:
	XORDataset();

	virtual SampleShape GetSampleShape() const;

	virtual Sample GetSample() const;
	virtual size_t GetEpochSize() const { return 4; }

	virtual void Next();
	virtual void Shuffle();

private:
	std::vector<mogi::Tensor2D> m_Samples;
	std::vector<mogi::Tensor2D> m_Labels;
	size_t m_SampleIndex;
};

namespace_dataset_end
