#include "GeneralFaces.h"
#include <assert.h>
#include <random>
#include <iostream>


namespace_dataset_start

GeneralFaces::GeneralFaces(const std::string& folderPath, size_t offset, size_t numImages, bool grayScale)
	: m_SampleIndex(0), m_EpochSize(numImages), m_GrayScale(grayScale)
{
	LoadImages(folderPath, offset, numImages);
}

size_t GeneralFaces::LoadImages(const std::string& folderPath, size_t offset, size_t numImages)
{
	for (size_t i = offset; i < offset+numImages; i++)
	{
		std::string filePath = folderPath + "/image_" + std::to_string(i) + ".jpg";
		cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

		m_Height = image.rows;
		m_Width = image.cols;

		if (image.empty())
		{
			throw std::runtime_error("No file at: " + filePath);
		}

		int depth = m_GrayScale ? 1 : 3;
		mogi::Tensor3D imageTensor(image.rows, image.cols, depth);
		for (size_t i = 0; i < image.rows; i++)
		{
			for (size_t j = 0; j < image.cols; j++)
			{
				cv::Vec3b brgPixel = image.at<cv::Vec3b>(i, j);
				if (m_GrayScale)
				{
					float average = (float)(brgPixel.val[0] + brgPixel.val[1] + brgPixel.val[2]) / 3.0f;
					float normalizedAverage = average / 255.0f;
					imageTensor.SetAt(i, j, 0, normalizedAverage);
				}
				else
				{
					imageTensor.SetAt(i, j, 0, (float)brgPixel.val[0] / 255.0f);
					imageTensor.SetAt(i, j, 1, (float)brgPixel.val[1] / 255.0f);
					imageTensor.SetAt(i, j, 2, (float)brgPixel.val[2] / 255.0f);
				}
			}
		}
		m_Images.push_back(imageTensor);

	}
}

SampleShape GeneralFaces::GetSampleShape() const
{
	size_t depth = m_GrayScale ? 1 : 3;
	return
	{
		m_Height, m_Width, depth,
		m_Height, m_Width, depth
	};
}

Sample GeneralFaces::GetSample() const
{
	return
	{
		m_Images[m_SampleIndex],
		m_Images[m_SampleIndex]
	};
}

void GeneralFaces::Next()
{
	m_SampleIndex = (m_SampleIndex + 1) % m_EpochSize;
}

void GeneralFaces::Shuffle()
{
	std::random_device rd;

	for (size_t i = 0; i < m_Images.size(); i++)
	{
		size_t swapIndex = rd() % m_EpochSize;
		std::swap(m_Images[i], m_Images[swapIndex]);
	}
}

void GeneralFaces::Display(const Tensor3D& tensor)
{
	const std::string charSequence = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";

	for (size_t i = 0; i < tensor.GetRows(); i++)
	{
		for (size_t j = 0; j < tensor.GetCols(); j++)
		{
			float average = tensor.GetAt(i, j, 0) + tensor.GetAt(i, j, 1) + tensor.GetAt(i, j, 2) / 3.0f;
			float value = std::min(average, 1.0f);
			char c = charSequence.at((int)(value * (charSequence.size() - 1)));
			std::cout << c << " ";
		}
		std::cout << "\n";
	}
}

namespace_dataset_end