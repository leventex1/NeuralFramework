#include "Tensor2D.h"
#include <assert.h>
#include <stdexcept>


namespace_start

Tensor2D::Tensor2D() : m_Rows(0), m_Cols(0), Tensor() { }

Tensor2D::Tensor2D(size_t rows, size_t cols, float value, bool onDevice)
	: m_Rows(rows), m_Cols(cols), Tensor(rows * cols, value, onDevice)
{
}

Tensor2D::Tensor2D(size_t rows, size_t cols, std::function<float()> initializer)
	: m_Rows(rows), m_Cols(cols), Tensor(rows * cols, initializer)
{
}

Tensor2D::Tensor2D(const std::initializer_list<std::initializer_list<float>>& initList)
	: Tensor()
{
	m_Rows = initList.size();
	m_Cols = initList.begin()->size();

	Alloc(m_Rows * m_Cols);

	size_t index = 0;
	for (auto i = initList.begin(); i != initList.end(); i++)
	{
		assert(i->size() == m_Cols && "Tensor2D missmatched params numbers!");
		for (auto j = i->begin(); j != i->end(); j++)
		{
			Tensor::SetAt(index, (*j));
			index++;
		}
	}
}

Tensor2D::Tensor2D(const Tensor2D& other)
	: m_Rows(other.m_Rows), m_Cols(other.m_Cols), Tensor(other)
{
}

Tensor2D::Tensor2D(size_t rows, size_t cols, size_t offsetRows, size_t offsetCols, float* data, bool onDevice)
	: m_Rows(rows), m_Cols(cols), m_OffsetRows(offsetRows), m_OffsetCols(offsetCols), Tensor(data, onDevice)
{
	if (IsOnDevice() && (offsetRows != 0 || offsetCols != 0))
	{
		throw std::runtime_error("Tensor data is on device.");
	}
}

Tensor2D::Tensor2D(size_t rows, size_t cols, const float* data, bool onDevice)
	: m_Rows(rows), m_Cols(cols), Tensor(data, rows * cols, onDevice)
{
}

float Tensor2D::GetAt(size_t row, size_t col) const
{
	size_t index = CalculateIndex(row, col);
	return Tensor::GetAt(index);
}

void Tensor2D::SetAt(size_t row, size_t col, float value)
{
	size_t index = CalculateIndex(row, col);
	Tensor::SetAt(index, value);
}

size_t Tensor2D::TraverseTo(size_t s) const
{
	size_t row = s / m_Cols;
	size_t col = s % m_Cols;
	return CalculateIndex(row, col);
}

size_t Tensor2D::CalculateIndex(size_t row, size_t col) const
{
	size_t offsetRows = m_OffsetRows != 0 ? m_OffsetRows : m_Cols;
	size_t offsetCols = m_OffsetCols != 0 ? m_OffsetCols : 1;
	return row * offsetRows + col * offsetCols;
}

namespace_end