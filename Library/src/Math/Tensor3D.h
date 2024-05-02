#pragma once
#include "Core.h"
#include "Tensor.h"


namespace_start

class LIBRARY_API Tensor3D : public Tensor
{
public:
	Tensor3D();
	Tensor3D(size_t rows, size_t cols, size_t depth, float value = 0.0f, bool onDevice=false);
	Tensor3D(size_t rows, size_t cols, size_t depth, std::function<float()> initializer);
	Tensor3D(const std::initializer_list<std::initializer_list<std::initializer_list<float>>>& initList);
	Tensor3D(const Tensor3D& other);

	// Watcher.
	Tensor3D(size_t rows, size_t cols, size_t depth, float* data, bool onDevice);
	// Copy.
	Tensor3D(size_t rows, size_t cols, size_t depth, const float* data, bool onDevice);
	// Swap.
	Tensor3D(size_t rows, size_t cols, size_t depth, Tensor&& tensor);

	inline virtual size_t GetSize() const { return m_Rows * m_Cols * m_Depth; }
	virtual size_t TraverseTo(size_t s) const;

	float GetAt(size_t row, size_t col, size_t deth) const;
	void SetAt(size_t row, size_t col, size_t depth, float value);

	inline size_t GetRows() const { return m_Rows; }
	inline size_t GetCols() const { return m_Cols; }
	inline size_t GetDepth() const { return m_Depth; }

private:
	size_t m_Rows, m_Cols, m_Depth;
};

namespace_end