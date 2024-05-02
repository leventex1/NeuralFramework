#include <assert.h>
#include <random>
#include <iostream>

#include "FaceRecognitionDataset.h"


namespace_dataset_start

FaceRecognitionDataset::FaceRecognitionDataset(
	const std::string& positiveFolderPath, const std::string& positivePrefix, size_t positiveOffset, size_t positiveNumImages,
	const std::string& negativeFolderPath, const std::string& negativePrefix, size_t negativeOffset, size_t negativeNumImages
) : m_Width(0), m_Height(0), m_SampleIndex(0)
{
	LoadImages(positiveFolderPath, positivePrefix, positiveOffset, positiveNumImages, 1);
	LoadImages(negativeFolderPath, negativePrefix, negativeOffset, negativeNumImages, 0);
	m_EpochSize = m_Images.size();
	Shuffle();
}

void FaceRecognitionDataset::LoadImages(const std::string& folderPath, const std::string& prefix, size_t offset, size_t numImages, unsigned char label)
{
	for (size_t i = offset; i < offset + numImages; i++)
	{
		std::string filePath = folderPath + "/" + prefix + "_" + std::to_string(i) + ".jpg";
		cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

		if ((m_Width > 0 && m_Width != image.rows) || (m_Height > 0 && m_Height != image.cols))
		{
			throw std::runtime_error("Image dimensions are different.");
		}

		m_Height = image.rows;
		m_Width = image.cols;


		if (image.empty())
		{
			throw std::runtime_error("No file at: " + filePath);
		}

		mogi::Tensor2D imageTensor(image.rows, image.cols);
		for (size_t i = 0; i < image.rows; i++)
		{
			for (size_t j = 0; j < image.cols; j++)
			{
				cv::Vec3b brgPixel = image.at<cv::Vec3b>(i, j);
				float average = (float)(brgPixel.val[0] + brgPixel.val[1] + brgPixel.val[2]) / 3.0f;
				float normalizedAverage = average / 255.0f;
				imageTensor.SetAt(i, j, normalizedAverage);
			}
		}
		m_Images.push_back(imageTensor);
		m_Labels.push_back(label);
	}
}

SampleShape FaceRecognitionDataset::GetSampleShape() const
{
	return
	{
		m_Height, m_Width, 1,
		1, 1, 1
	};
}

Sample FaceRecognitionDataset::GetSample() const
{
	Tensor3D input(m_Height, m_Width, 1, (const float*)m_Images[m_SampleIndex].GetData(), false);
	return
	{
		input,
		{ { { (float)m_Labels[m_SampleIndex] } } }
	};
}


void FaceRecognitionDataset::Next()
{
	m_SampleIndex = (m_SampleIndex + 1) % m_EpochSize;
}

void FaceRecognitionDataset::Shuffle()
{
	std::random_device rd;

	for (size_t i = 0; i < m_Images.size(); i++)
	{
		size_t swapIndex = rd() % m_EpochSize;
		std::swap(m_Images[i], m_Images[swapIndex]);
		std::swap(m_Labels[i], m_Labels[swapIndex]);
	}
}

void FaceRecognitionDataset::Display(const Tensor2D& tensor)
{
	const std::string charSequence = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";

	for (size_t i = 0; i < tensor.GetRows(); i++)
	{
		for (size_t j = 0; j < tensor.GetCols(); j++)
		{
			float average = tensor.GetAt(i, j);
			float value = std::min(average, 1.0f);
			char c = charSequence.at((int)(value * (charSequence.size() - 1)));
			std::cout << c << " ";
		}
		std::cout << "\n";
	}
}

namespace_dataset_end