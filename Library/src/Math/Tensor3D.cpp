#include "Tensor3D.h"
#include <assert.h>


namespace_start

Tensor3D::Tensor3D() : m_Rows(0), m_Cols(0), m_Depth(0), Tensor() { }

Tensor3D::Tensor3D(size_t rows, size_t cols, size_t depth, float value, bool onDevice)
	: m_Rows(rows), m_Cols(cols), m_Depth(depth), Tensor(rows * cols * depth, value, onDevice)
{
}

Tensor3D::Tensor3D(size_t rows, size_t cols, size_t depth, std::function<float()> initializer)
	: m_Rows(rows), m_Cols(cols), m_Depth(depth), Tensor(rows * cols * depth, initializer)
{
}

Tensor3D::Tensor3D(const std::initializer_list<std::initializer_list<std::initializer_list<float>>>& initList)
	: Tensor()
{
	m_Depth = initList.size();
	m_Rows = initList.begin()->size();
	m_Cols = initList.begin()->begin()->size();

	Alloc(m_Rows * m_Cols * m_Depth);

	size_t index = 0;
	for (auto d = initList.begin(); d != initList.end(); d++)
	{
		assert(d->size() == m_Rows && "Tensor3D missmatched params rows!");

		for (auto i = d->begin(); i != d->end(); i++)
		{
			assert(i->size() == m_Cols && "Tensor3D missmatched params cols!");

			for (auto j = i->begin(); j != i->end(); j++)
			{

				Tensor::SetAt(index, (*j));
				index++;
			}
		}
	}
}

Tensor3D::Tensor3D(const Tensor3D& other)
	: m_Rows(other.m_Rows), m_Cols(other.m_Cols), m_Depth(other.m_Depth), Tensor(other)
{
}

Tensor3D::Tensor3D(size_t rows, size_t cols, size_t depth, float* data, bool onDevice)
	: m_Rows(rows), m_Cols(cols), m_Depth(depth), Tensor(data, onDevice)
{
}

Tensor3D::Tensor3D(size_t rows, size_t cols, size_t depth, const float* data, bool onDevice)
	: m_Rows(rows), m_Cols(cols), m_Depth(depth), Tensor(data, rows * cols * depth, onDevice)
{
}

Tensor3D::Tensor3D(size_t rows, size_t cols, size_t depth, Tensor&& tensor)
	: m_Rows(rows), m_Cols(cols), m_Depth(depth), Tensor(std::move(tensor))
{
}

float Tensor3D::GetAt(size_t row, size_t col, size_t depth) const
{
	size_t index = row * m_Cols + col + m_Cols * m_Rows * depth;
	return Tensor::GetAt(index);
}

void Tensor3D::SetAt(size_t row, size_t col, size_t depth, float value)
{
	size_t index = row * m_Cols + col + m_Cols * m_Rows * depth;
	Tensor::SetAt(index, value);
}

size_t Tensor3D::TraverseTo(size_t s) const
{
	return s;
}

namespace_end