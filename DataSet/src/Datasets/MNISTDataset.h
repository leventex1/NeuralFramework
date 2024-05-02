#pragma once
#include <string>
#include <fstream>
#include <vector>

#include "../Dataset.h"


namespace_dataset_start

class DATASET_API MNISTDataset : public Dataset
{
public:
	MNISTDataset(const std::string& imagesFilePath, const std::string& labelsFilePath);

	virtual SampleShape GetSampleShape() const;

	virtual Sample GetSample() const;
	virtual size_t GetEpochSize() const { return m_EpochSize; }

	virtual void Next();
	virtual void Shuffle();

	void Display() const;
	static void Display(const Tensor2D& tensor, int label = -1);

private:
	size_t LoadImages(const std::string& filePath);
	size_t LoadLabels(const std::string& filePath);

private:
	std::vector<Tensor2D> m_Images;
	std::vector<unsigned char> m_Labels;
	size_t m_EpochSize;
	size_t m_SampleIndex;
};

namespace_dataset_end
