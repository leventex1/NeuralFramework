#include "XORDataset.h"
#include <random>


namespace_dataset_start

XORDataset::XORDataset()
	: m_SampleIndex(0)
{
	m_Samples = 
	{
		{
			{ 0.0f },
			{ 0.0f }
		},
		{
			{ 1.0f },
			{ 0.0f }
		},
		{
			{ 0.0f },
			{ 1.0f }
		},
		{
			{ 1.0f },
			{ 1.0f }
		}
	};

	m_Labels = 
	{
		{
			{ 0.0f },
		},
		{
			{ 1.0f },
		},
		{
			{ 1.0f },
		},
		{
			{ 0.0f },
		}
	};

}

SampleShape XORDataset::GetSampleShape() const
{
	return {
		2, 1, 1,
		1, 1, 1
	};
}

Sample XORDataset::GetSample() const
{
	return {
		Tensor3D(2, 1, 1, m_Samples[m_SampleIndex].GetData(), false),
		{ { { m_Labels[m_SampleIndex].GetAt(0, 0) } } }
	};
}

void XORDataset::Next()
{
	m_SampleIndex = (m_SampleIndex + 1) % 4;
}

void XORDataset::Shuffle()
{
	std::random_device rd;

	for (size_t i = 0; i < 4; i++)
	{
		size_t swapIndex = rd() % 4;
		std::swap(m_Samples[i], m_Samples[swapIndex]);
		std::swap(m_Labels[i], m_Labels[swapIndex]);
	}
}

namespace_dataset_end