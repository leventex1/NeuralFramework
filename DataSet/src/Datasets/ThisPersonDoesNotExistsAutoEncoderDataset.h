#pragma once
#include <string>
#include <fstream>
#include <vector>

#include "../Dataset.h"
#include <opencv2/opencv.hpp>


namespace_dataset_start

class DATASET_API ThisPersonDoesNotExistsAutoEncoderDataset : public Dataset
{
public:
	ThisPersonDoesNotExistsAutoEncoderDataset(const std::string& folderPath, size_t numImages);

	virtual SampleShape GetSampleShape() const;

	virtual Sample GetSample() const;
	virtual size_t GetEpochSize() const { return m_EpochSize; }

	virtual void Next();
	virtual void Shuffle();

	static void Display(const Tensor2D& tensor);

private:
	size_t LoadImages(const std::string& folderPath, size_t numImages);

private:
	size_t m_Height, m_Width;
	std::vector<Tensor2D> m_Images;
	size_t m_EpochSize;
	size_t m_SampleIndex;
};

namespace_dataset_end
#pragma once
