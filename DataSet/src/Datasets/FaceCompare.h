#pragma once
#include <string>
#include <fstream>
#include <vector>

#include "../Dataset.h"
#include <opencv2/opencv.hpp>


namespace_dataset_start

class DATASET_API FaceCompare : public Dataset
{
public:
	FaceCompare(
		const std::string& positiveFolderPath, size_t positiveOffset, size_t positiveNumImages,
		const std::string& negativeFolderPath, size_t negativeOffset, size_t negativeNumImages,
		size_t epochsize
	);

	virtual SampleShape GetSampleShape() const;

	virtual Sample GetSample() const;
	virtual size_t GetEpochSize() const { return m_EpochSize; }

	Sample GetSample(int positiveIndex, int negativeIndex) const;
	Sample GetSample(int positiveIndex, const Tensor2D& input) const;

	virtual void Next();
	virtual void Shuffle();

	static void Display(const Tensor2D& tensor);

private:
	std::vector<Tensor2D> LoadImages(const std::string& folderPath, size_t offset, size_t numImages);

private:
	size_t m_Height, m_Width;
	std::vector<Tensor2D> m_Positives;
	std::vector<Tensor2D> m_Negatives;
	size_t m_EpochSize;
};

namespace_dataset_end
#pragma once
