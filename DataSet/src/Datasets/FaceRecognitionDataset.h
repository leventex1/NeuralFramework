#pragma once
#include <string>
#include <fstream>
#include <vector>

#include "../Dataset.h"
#include <opencv2/opencv.hpp>


namespace_dataset_start

class DATASET_API FaceRecognitionDataset : public Dataset
{
public:
	FaceRecognitionDataset(
		const std::string& positiveFolderPath, const std::string& positivePrefix, size_t positiveOffset, size_t positiveNumImages,
		const std::string& negativeFolderPath, const std::string& negativePrefix, size_t negativeOffset, size_t negativeNumImages
	);

	virtual SampleShape GetSampleShape() const;

	virtual Sample GetSample() const;
	virtual size_t GetEpochSize() const { return m_EpochSize; }

	virtual void Next();
	virtual void Shuffle();

	static void Display(const Tensor2D& tensor);

private:
	void LoadImages(const std::string& folderPath, const std::string& prefix, size_t offset, size_t numImages, unsigned char label);

private:
	size_t m_Height, m_Width;
	std::vector<Tensor2D> m_Images;
	std::vector<unsigned char> m_Labels;
	size_t m_SampleIndex;
	size_t m_EpochSize;
};

namespace_dataset_end
#pragma once
