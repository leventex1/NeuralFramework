#include "ThisPersonDoesNotExistsAutoEncoderDataset.h"
#include <assert.h>
#include <random>
#include <iostream>


namespace_dataset_start

ThisPersonDoesNotExistsAutoEncoderDataset::ThisPersonDoesNotExistsAutoEncoderDataset(const std::string& folderPath, size_t numImages)
	: m_SampleIndex(0), m_EpochSize(numImages)
{
	LoadImages(folderPath, numImages);
}

size_t ThisPersonDoesNotExistsAutoEncoderDataset::LoadImages(const std::string& folderPath, size_t numImages)
{
	for (size_t i = 0; i < numImages; i++)
	{
		std::string filePath = folderPath + "/image_" + std::to_string(i + 1) + ".jpg";
		cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

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

	}	
}

SampleShape ThisPersonDoesNotExistsAutoEncoderDataset::GetSampleShape() const
{
	return
	{
		m_Height, m_Width, 1,
		m_Height, m_Width, 1
	};
}

Sample ThisPersonDoesNotExistsAutoEncoderDataset::GetSample() const
{
	SampleShape s = GetSampleShape();
	Tensor3D tensor = Tensor3D(s.InputRows, s.InputCols, s.InputDepth, (const float*)m_Images[m_SampleIndex].GetData(), false);
	return
	{
		tensor, 
		tensor
	};
}

void ThisPersonDoesNotExistsAutoEncoderDataset::Next()
{
	m_SampleIndex = (m_SampleIndex + 1) % m_EpochSize;
}

void ThisPersonDoesNotExistsAutoEncoderDataset::Shuffle()
{
	std::random_device rd;

	for (size_t i = 0; i < m_Images.size(); i++)
	{
		size_t swapIndex = rd() % m_EpochSize;
		std::swap(m_Images[i], m_Images[swapIndex]);
	}
}

void ThisPersonDoesNotExistsAutoEncoderDataset::Display(const Tensor2D& tensor)
{
	const std::string charSequence = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";

	for (size_t i = 0; i < tensor.GetRows(); i++)
	{
		for (size_t j = 0; j < tensor.GetCols(); j++)
		{
			float value = std::min(tensor.GetAt(i, j), 1.0f);
			char c = charSequence.at((int)(value * (charSequence.size() - 1)));
			std::cout << c << " ";
		}
		std::cout << "\n";
	}
}

namespace_dataset_end