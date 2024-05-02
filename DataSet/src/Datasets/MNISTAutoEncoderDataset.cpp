#include "MNISTAutoEncoderDataset.h"
#include <assert.h>
#include <random>
#include <iostream>


namespace_dataset_start

static int32_t ReadInt(std::ifstream& file) {
	int32_t result;
	file.read(reinterpret_cast<char*>(&result), sizeof(result));
	// Convert from big endian to little endian
	return (result >> 24) |
		((result << 8) & 0x00FF0000) |
		((result >> 8) & 0x0000FF00) |
		(result << 24);
}

MNISTAutoEncoderDataset::MNISTAutoEncoderDataset(const std::string& imagesFilePath)
	: m_SampleIndex(0)
{
	size_t imagesCount = LoadImages(imagesFilePath);

	m_EpochSize = imagesCount;
}

size_t MNISTAutoEncoderDataset::LoadImages(const std::string& filePath)
{
	std::ifstream file(filePath, std::ios::binary);
	assert(file.is_open() && "Could not open MNIST images file!");

	int magicNumber = ReadInt(file);
	int imagesCount = ReadInt(file);
	int rows = ReadInt(file);
	int cols = ReadInt(file);

	assert(magicNumber == 2051 && "Invalid MNIST image file!");

	for (int t = 0; t < imagesCount; t++)
	{
		unsigned char data[28 * 28];
		file.read((char*)data, 28 * 28);
		m_Images.push_back(Tensor2D(28, 28));
		for (size_t i = 0; i < 28; i++)
		{
			for (size_t j = 0; j < 28; j++)
			{
				m_Images[t].SetAt(i, j, (float)data[i * 28 + j] / 255);
			}
		}
	}

	file.close();

	return imagesCount;
}

SampleShape MNISTAutoEncoderDataset::GetSampleShape() const
{
	return
	{
		28, 28, 1,
		28, 28, 1
	};
}

Sample MNISTAutoEncoderDataset::GetSample() const
{
	SampleShape s = GetSampleShape();
	return
	{
		Tensor3D(s.InputRows, s.InputCols, s.InputDepth, (const float*)m_Images[m_SampleIndex].GetData(), false),
		Tensor3D(s.LabelRows, s.LabelCols, s.LabelDepth, (const float*)m_Images[m_SampleIndex].GetData(), false)
	};
}

void MNISTAutoEncoderDataset::Next()
{
	m_SampleIndex = (m_SampleIndex + 1) % m_EpochSize;
}

void MNISTAutoEncoderDataset::Shuffle()
{
	std::random_device rd;

	for (size_t i = 0; i < m_Images.size(); i++)
	{
		size_t swapIndex = rd() % m_EpochSize;
		std::swap(m_Images[i], m_Images[swapIndex]);
	}
}

void MNISTAutoEncoderDataset::Display() const
{
	Sample sample = GetSample();
	Display(CreateWatcher(sample.Input, 0));
}

void MNISTAutoEncoderDataset::Display(const Tensor2D& tensor)
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