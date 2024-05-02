#include <assert.h>
#include <random>
#include <iostream>

#include "FaceCompare.h"


namespace_dataset_start

FaceCompare::FaceCompare(
	const std::string& positiveFolderPath, size_t positiveOffset, size_t positiveNumImages,
	const std::string& negativeFolderPath, size_t negativeOffset, size_t negativeNumImages,
	size_t epochsize
) : m_EpochSize(epochsize), m_Width(0), m_Height(0)
{
	m_Positives = LoadImages(positiveFolderPath, positiveOffset, positiveNumImages);
	m_Negatives = LoadImages(negativeFolderPath, negativeOffset, negativeNumImages);
}

std::vector<Tensor2D> FaceCompare::LoadImages(const std::string& folderPath, size_t offset, size_t numImages)
{
	std::vector<Tensor2D> res;

	for (size_t i = offset; i < offset + numImages; i++)
	{
		std::string filePath = folderPath + "/image_" + std::to_string(i) + ".jpg";
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
		res.push_back(imageTensor);
	}

	return res;
}

SampleShape FaceCompare::GetSampleShape() const
{
	return
	{
		m_Height, m_Width, 2,
		1, 1, 1
	};
}

Sample FaceCompare::GetSample() const
{
	std::random_device rd;
	bool isPositiveSample = (rd() % 2 == 0);  // two positive image or one positive and one negative.

	int positiveIndex = rd() % m_Positives.size();
	int compareIndex = rd() % (isPositiveSample ? m_Positives.size() : m_Negatives.size());

	const Tensor2D& positive = m_Positives[positiveIndex];
	const Tensor2D& compare = (isPositiveSample ? m_Positives[compareIndex] : m_Negatives[compareIndex]);

	Tensor3D input(m_Height, m_Width, 2, 0.0f);

	Tensor2D inputPositiveWatcher = CreateWatcher(input, 0);
	Tensor2D inputCompareWatcher = CreateWatcher(input, 1);

	inputPositiveWatcher.Add(positive);
	inputCompareWatcher.Add(compare);

	return
	{
		input,
		{ { { (isPositiveSample ? 1.0f : 0.0f) }}}
	};
}

Sample FaceCompare::GetSample(int positiveIndex, int negativeIndex) const
{
	Tensor3D input(m_Height, m_Width, 2, 0.0f);
	const Tensor2D& positive = m_Positives[positiveIndex];
	const Tensor2D& compare = m_Negatives[negativeIndex];

	Tensor2D inputPositiveWatcher = CreateWatcher(input, 0);
	Tensor2D inputCompareWatcher = CreateWatcher(input, 1);

	inputPositiveWatcher.Add(positive);
	inputCompareWatcher.Add(compare);

	return
	{
		input,
		{ { { -1.0f }}}
	};
}

Sample FaceCompare::GetSample(int positiveIndex, const Tensor2D& input) const
{
	Tensor3D sample(m_Height, m_Width, 2, 0.0f);

	const Tensor2D& positive = m_Positives[positiveIndex];

	Tensor2D samplePositiveWatcher = CreateWatcher(sample, 0);
	Tensor2D sampleCompareWatcher = CreateWatcher(sample, 1);

	samplePositiveWatcher.Add(positive);
	sampleCompareWatcher.Add(input);

	return
	{
		sample,
		{ { { -1.0f }}}
	};
}

void FaceCompare::Next()
{
}

void FaceCompare::Shuffle()
{
}

void FaceCompare::Display(const Tensor2D& tensor)
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