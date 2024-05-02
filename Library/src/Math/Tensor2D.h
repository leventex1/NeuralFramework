#pragma once
#include "Core.h"
#include "Tensor.h"


namespace_start

class LIBRARY_API Tensor2D : public Tensor
{
public:
	Tensor2D();
	Tensor2D(size_t rows, size_t cols, float value = 0.0f, bool onDevice = false);
	Tensor2D(size_t rows, size_t cols, std::function<float()> initializer);
	Tensor2D(const std::initializer_list<std::initializer_list<float>>& initList);
	Tensor2D(const Tensor2D& other);

	// Watcher.
	Tensor2D(size_t rows, size_t cols, size_t offsetRows, size_t offsetCols, float* data, bool onDevice);
	// Copy.
	Tensor2D(size_t rows, size_t cols, const float* data, bool onDevice);

	inline virtual size_t GetSize() const { return m_Rows * m_Cols; }
	virtual size_t TraverseTo(size_t s) const;
	
	// Use the Getter with calculating the offsets within the data.
	float GetAt(size_t row, size_t col) const;
	// Use the Setter with calculating the offsets within the data.
	void SetAt(size_t row, size_t col, float value);

	inline size_t GetRows() const { return m_Rows; }
	inline size_t GetCols() const { return m_Cols; }

	inline size_t GetOffsetRows() const { return m_OffsetRows; }
	inline size_t GetOffsetCols() const { return m_OffsetCols; }

	size_t CalculateIndex(size_t row, size_t cols) const;

private:
	size_t m_Rows, m_Cols;
	size_t m_OffsetRows = 0, m_OffsetCols = 0;
};

namespace_end
