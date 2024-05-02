#pragma once
#include <string>
#include <fstream>
#include <vector>

#include "../Dataset.h"
#include <opencv2/opencv.hpp>


namespace_dataset_start

class DATASET_API GeneralFaces : public Dataset
{
public:
	GeneralFaces(const std::string& folderPath, size_t offset, size_t numImages, bool grayScale=false);

	virtual SampleShape GetSampleShape() const;

	virtual Sample GetSample() const;
	virtual size_t GetEpochSize() const { return m_EpochSize; }

	virtual void Next();
	virtual void Shuffle();

	static void Display(const Tensor3D& tensor);

private:
	size_t LoadImages(const std::string& folderPath, size_t offset, size_t numImages);

private:
	size_t m_Height, m_Width;
	std::vector<Tensor3D> m_Images;
	size_t m_EpochSize;
	size_t m_SampleIndex;
	bool m_GrayScale;
};

namespace_dataset_end
#pragma once
