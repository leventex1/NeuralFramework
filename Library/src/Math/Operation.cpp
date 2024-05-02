#include "Operation.h"
#include <assert.h>
#include <random>
#include <limits>

#include <MogiAccelerator.h>

#include <future>
#include <mutex>
#include "../ThreadPool.h"


namespace_start

float Sum(const Tensor* tensor)
{
	float sum = 0.0f;
	for (size_t i = 0; i < tensor->GetSize(); i++)
	{
		size_t index = tensor->TraverseTo(i);
		sum += tensor->GetData()[index];
	}
	return sum;
}

Tensor2D SliceTensor(const Tensor3D& tensor, size_t depth)
{
	assert(depth < tensor.GetDepth() && "Depth is out of range.");

	return Tensor2D(tensor.GetRows(), tensor.GetCols(), tensor.GetData() + depth * tensor.GetRows() * tensor.GetCols(), tensor.IsOnDevice());
}

Tensor2D CreateWatcher(Tensor2D& tensor, size_t row, size_t col, size_t rows, size_t cols, size_t skipRows, size_t skipCols)
{
	assert(row + (rows - 1) * (1 + skipRows) < tensor.GetRows() &&
		col + (cols - 1) * (1 + skipCols) < tensor.GetCols()
		&& "Out of range tensor params!");

	size_t otherOffsetRows = tensor.GetOffsetRows() != 0 ? tensor.GetOffsetRows() : tensor.GetCols();
	size_t otherOffsetCols = tensor.GetOffsetCols() != 0 ? tensor.GetOffsetCols() : 1;

	float* offsetData = tensor.GetData() + tensor.CalculateIndex(row, col);

	size_t offsetRows = otherOffsetRows * (1 + skipRows);
	size_t offsetCols = otherOffsetCols * (1 + skipCols);

	return Tensor2D(rows, cols, offsetRows, offsetCols, offsetData, tensor.IsOnDevice());
}

Tensor2D CreateWatcher(Tensor3D& tensor, size_t depth)
{
	assert(depth < tensor.GetDepth() && "Depth is out of range.");

	return Tensor2D(tensor.GetRows(), tensor.GetCols(), 0, 0, tensor.GetData() + depth * tensor.GetRows() * tensor.GetCols(), tensor.IsOnDevice());
}

Tensor3D CreateWatcher(Tensor2D& tensor)
{
	return Tensor3D(tensor.GetRows(), tensor.GetCols(), 1, tensor.GetData(), tensor.IsOnDevice());
}

Tensor3D CreateWatcher(Tensor3D& tensor, size_t fromDepth, size_t depth)
{
	assert(fromDepth + depth <= tensor.GetDepth() && "Depth is out of range.");

	return Tensor3D(tensor.GetRows(), tensor.GetCols(), depth, tensor.GetData() + fromDepth * tensor.GetRows() * tensor.GetCols(), tensor.IsOnDevice());
}

Tensor2D Random2D(size_t rows, size_t cols, float min, float max, bool useCuda)
{
	if (useCuda)
	{
		return accelerator::CudaRandom2D(rows, cols, min, max);
	}
	std::random_device rd;
	Tensor2D res(rows, cols, [&]() -> float {
		float r = (float)rd() / (float)rd.max();
		return min + r * (max - min);
	});
	return res;
}

Tensor3D Random3D(size_t rows, size_t cols, size_t depth, float min, float max)
{
	std::random_device rd;
	Tensor3D res(rows, cols, depth);
	res.Map([&](float v) -> float {
		float r = (float)rd() / (float)rd.max();
		return min + r * (max - min);
	});
	return res;
}

Tensor2D Map(const Tensor2D& t, std::function<float(float v)> mapper)
{
	if (t.IsOnDevice())
	{
		throw std::runtime_error("Tensor is on device!");
	}
	Tensor2D res = t;
	for (size_t i = 0; i < res.GetRows(); i++)
		for (size_t j = 0; j < res.GetCols(); j++)
			res.SetAt(i, j, mapper(res.GetAt(i, j)));
	return res;
}

Tensor3D Map(const Tensor3D& t, std::function<float(float v)> mapper)
{
	if (t.IsOnDevice())
	{
		throw std::runtime_error("Tensor is on device!");
	}
	Tensor3D res = t;
	for (size_t k = 0; k < res.GetDepth(); k++)
		for (size_t i = 0; i < res.GetRows(); i++)
			for (size_t j = 0; j < res.GetCols(); j++)
				res.SetAt(i, j, k, mapper(res.GetAt(i, j, k)));
	return res;
}

Tensor2D Add(const Tensor2D& t1, const Tensor2D& t2)
{
	Tensor2D res = t1;
	res.Add(t2);
	return res;
}

Tensor2D Sub(const Tensor2D& t1, const Tensor2D& t2)
{
	Tensor2D res = t1;
	res.Sub(t2);
	return res;
}

Tensor2D Mult(const Tensor2D& t1, const Tensor2D& t2)
{
	Tensor2D res = t1;
	res.Mult(t2);
	return res;
}

Tensor2D Div(const Tensor2D& t1, const Tensor2D& t2)
{
	Tensor2D res = t1;
	res.Div(t2);
	return res;
}

Tensor2D Add(const Tensor2D& t1, float value)
{
	Tensor2D res = t1;
	res.Add(value);
	return res;
}

Tensor2D Sub(const Tensor2D& t1, float value)
{
	Tensor2D res = t1;
	res.Sub(value);
	return res;
}

Tensor2D Mult(const Tensor2D& t1, float value)
{
	Tensor2D res = t1;
	res.Mult(value);
	return res;
}

Tensor2D Div(const Tensor2D& t1, float value)
{
	Tensor2D res = t1;
	res.Div(value);
	return res;
}

std::pair<size_t, size_t> MaxPos(const Tensor2D& tensor)
{
	assert(tensor.GetSize() > 0 && "No value in tensor!");

	std::pair<size_t, size_t> res = { 0, 0 };
	float min = tensor.GetAt(0, 0);

	for (size_t i = 0; i < tensor.GetRows(); i++)
	{
		for (size_t j = 0; j < tensor.GetCols(); j++)
		{
			if (tensor.GetAt(i, j) > min)
			{
				min = tensor.GetAt(i, j);
				res = { i, j };
			}
		}
	}

	return res;
}

Tensor2D Transpose(const Tensor2D& t)
{
	Tensor2D res(t.GetCols(), t.GetRows());

	for (size_t i = 0; i < res.GetRows(); i++)
	{
		for (size_t j = 0; j < res.GetCols(); j++)
		{
			res.SetAt(i, j, t.GetAt(j, i));
		}
	}

	return res;
}

void AsyncMatrixMult(int startRow, int endRow, int startCol, int endCol, Tensor2D* res, const Tensor2D* left, const Tensor2D* right)
{
	for (int row = startRow; row < endRow; ++row) {
		for (int col = startCol; col < endCol; ++col) {

			float product = 0.0f;
			for (size_t t = 0; t < left->GetCols(); t++)
			{
				product += left->GetAt(row, t) * right->GetAt(t, col);
			}
			res->SetAt(row, col, product);
		}
	}
}

Tensor2D MatrixMult(const Tensor2D& left, const Tensor2D& right)
{
	if (left.IsOnDevice() != right.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (left.IsOnDevice())
	{
		return accelerator::MatrixMultCUDA(left, right);
	}

	assert(left.GetCols() == right.GetRows() && "Matrix params for matrix multiplication not math! Left.column != Right.rows");
	Tensor2D res(left.GetRows(), right.GetCols());

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int rowsPerThread = res.GetRows() / numThreads;
	int extraRows = res.GetRows() % numThreads;

	std::vector<std::future<void>> threads;
	size_t startRow = 0;
	for (size_t i = 0; i < numThreads; i++)
	{
		size_t endRow = startRow + rowsPerThread + (i < extraRows ? 1 : 0);

		threads.emplace_back(
			pool->enqueue(AsyncMatrixMult, startRow, endRow, 0, res.GetCols(), &res, &left, &right)
		);

		startRow = endRow;
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (size_t row = 0; row < res.GetRows(); row++)
	{
		for (size_t col = 0; col < res.GetCols(); col++)
		{
			float product = 0.0f;
			for (size_t t = 0; t < left.GetCols(); t++)
			{
				product += left.GetAt(row, t) * right.GetAt(t, col);
			}
			res.SetAt(row, col, product);
		}
	}
#endif // ASYNC

	return res;
}

void AsyncMatrixMultLeftTranspose(int startRow, int endRow, int startCol, int endCol, Tensor2D* res, const Tensor2D* left, const Tensor2D* right)
{
	for (int row = startRow; row < endRow; ++row) {
		for (int col = startCol; col < endCol; ++col) {

			float product = 0.0f;
			for (size_t t = 0; t < left->GetRows(); t++)
			{
				product += left->GetAt(t, row) * right->GetAt(t, col);
			}
			res->SetAt(row, col, product);
		}
	}
}

Tensor2D MatrixMultLeftTranspose(const Tensor2D& left, const Tensor2D& right)
{
	assert(left.GetRows() == right.GetRows() && "Matrix params for matrix multiplication not math! Left.rows(after taking transpose) != Right.rows");
	if (left.IsOnDevice() != right.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (left.IsOnDevice())
	{
		return accelerator::MatrixMultLeftTransposeCUDA(left, right);
	}

	Tensor2D res(left.GetCols(), right.GetCols());

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int rowsPerThread = res.GetRows() / numThreads;
	int extraRows = res.GetRows() % numThreads;

	std::vector<std::future<void>> threads;
	size_t startRow = 0;
	for (size_t i = 0; i < numThreads; i++)
	{
		size_t endRow = startRow + rowsPerThread + (i < extraRows ? 1 : 0);

		threads.emplace_back(
			pool->enqueue(AsyncMatrixMultLeftTranspose, startRow, endRow, 0, res.GetCols(), &res, &left, &right)
		);

		startRow = endRow;
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (size_t row = 0; row < res.GetRows(); row++)
	{
		for (size_t col = 0; col < res.GetCols(); col++)
		{
			float product = 0.0f;
			for (size_t t = 0; t < left.GetRows(); t++)
			{
				product += left.GetAt(t, row) * right.GetAt(t, col);
			}
			res.SetAt(row, col, product);
		}
	}
#endif

	return res;
}

void AsyncMatrixMulRightTranspose(int startRow, int endRow, int startCol, int endCol, Tensor2D* res, const Tensor2D* left, const Tensor2D* right)
{
	for (int row = startRow; row < endRow; ++row) {
		for (int col = startCol; col < endCol; ++col) {

			float product = 0.0f;
			for (size_t t = 0; t < left->GetCols(); t++)
			{
				product += left->GetAt(row, t) * right->GetAt(col, t);
			}
			res->SetAt(row, col, product);
		}
	}
}

Tensor2D MatrixMultRightTranspose(const Tensor2D& left, const Tensor2D& right)
{
	assert(left.GetCols() == right.GetCols() && "Matrix params for matrix multiplication not math! Left.rows(after taking transpose) != Right.rows");
	if (left.IsOnDevice() != right.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (left.IsOnDevice())
	{
		return accelerator::MatrixMultRightTransposeCUDA(left, right);
	}
	
	Tensor2D res(left.GetRows(), right.GetRows());

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int rowsPerThread = res.GetRows() / numThreads;
	int extraRows = res.GetRows() % numThreads;

	std::vector<std::future<void>> threads;
	size_t startRow = 0;
	for (size_t i = 0; i < numThreads; i++)
	{
		size_t endRow = startRow + rowsPerThread + (i < extraRows ? 1 : 0);

		threads.emplace_back(
			pool->enqueue(AsyncMatrixMulRightTranspose, startRow, endRow, 0, res.GetCols(), &res, &left, &right)
		);

		startRow = endRow;
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (size_t row = 0; row < res.GetRows(); row++)
	{
		for (size_t col = 0; col < res.GetCols(); col++)
		{
			float product = 0.0f;
			for (size_t t = 0; t < left.GetCols(); t++)
			{
				product += left.GetAt(row, t) * right.GetAt(col, t);
			}
			res.SetAt(row, col, product);
		}
	}
#endif

	return res;
}

size_t CalcConvSize(size_t inputSize, size_t kernelSize, size_t stride, size_t padding)
{
	return (inputSize - kernelSize + 2 * padding) / stride + 1;
}

float KernelOperation(const Tensor2D& window, const Tensor2D& kernel)
{
	assert(window.GetRows() == kernel.GetRows() && window.GetCols() == kernel.GetCols() && "Window and kernel params not match!");
	
	float res = 0.0f;
	for (size_t i = 0; i < window.GetSize(); i++)
	{
		size_t windowIndex = window.TraverseTo(i);
		size_t kernelIndex = kernel.TraverseTo(i);
		res += window.GetData()[windowIndex] * kernel.GetData()[kernelIndex];
	}
	return res;
}

void Convolution(Tensor2D& output, const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor!");

	for (size_t y = 0; y < outputRows; y++)
	{
		for (size_t x = 0; x < outputCols; x++)
		{
			float sum = 0.0f;
			for (size_t ky = 0; ky < kernel.GetRows(); ky++)
			{
				for (size_t kx = 0; kx < kernel.GetCols(); kx++)
				{
					int posY = y * stride + ky - padding;
					int posX = x * stride + kx - padding;

					if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
						sum += input.GetAt(posY, posX) * kernel.GetAt(ky, kx);
					}
				}
			}
			output.SetAt(y, x, output.GetAt(y, x) + sum);
		}
	}
}

void ConvolutionKernelFlip(Tensor2D& output, const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor!");

	for (size_t y = 0; y < outputRows; y++)
	{
		for (size_t x = 0; x < outputCols; x++)
		{
			float sum = 0.0f;
			for (size_t ky = 0; ky < kernel.GetRows(); ky++)
			{
				for (size_t kx = 0; kx < kernel.GetCols(); kx++)
				{
					int posY = y * stride + ky - padding;
					int posX = x * stride + kx - padding;

					size_t flippedKy = kernel.GetRows() - 1 - ky;
					size_t flippedKx = kernel.GetCols() - 1 - kx;

					if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
						sum += input.GetAt(posY, posX) * kernel.GetAt(flippedKy, flippedKx);
					}
				}
			}
			output.SetAt(y, x, output.GetAt(y, x) + sum);
		}
	}
}

void Convolution(Tensor2D& output, const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(input.GetDepth() == kernel.GetDepth() && "Input and kernel depth not match!");
	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor!");

	if (output.IsOnDevice() != input.IsOnDevice() || output.IsOnDevice() != kernel.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (output.IsOnDevice())
	{
		accelerator::CudaConvolution(&output, &input, &kernel, stride, padding);
		return;
	}

	for (size_t y = 0; y < outputRows; y++)
	{
		for (size_t x = 0; x < outputCols; x++)
		{
			float sum = 0.0f;
			for (size_t ky = 0; ky < kernel.GetRows(); ky++)
			{
				for (size_t kx = 0; kx < kernel.GetCols(); kx++)
				{
					for (size_t d = 0; d < kernel.GetDepth(); d++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(ky, kx, d);
						}
					}

				}
			}
			output.SetAt(y, x, output.GetAt(y, x) + sum);
		}
	}
}

void ConvolutionKernelFlip(Tensor2D& output, const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(input.GetDepth() == kernel.GetDepth() && "Input and kernel depth not match!");
	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor!");

	for (size_t y = 0; y < outputRows; y++)
	{
		for (size_t x = 0; x < outputCols; x++)
		{
			float sum = 0.0f;
			for (size_t ky = 0; ky < kernel.GetRows(); ky++)
			{
				for (size_t kx = 0; kx < kernel.GetCols(); kx++)
				{
					for (size_t d = 0; d < kernel.GetDepth(); d++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						size_t flippedKy = kernel.GetRows() - 1 - ky;
						size_t flippedKx = kernel.GetCols() - 1 - kx;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(flippedKy, flippedKx, d);
						}
					}

				}
			}
			output.SetAt(y, x, output.GetAt(y, x) + sum);
		}
	}
}

void AsyncConvolution(size_t startDepth, size_t endDepth, Tensor3D& output, const Tensor3D& input, const Tensor2D& kernel, size_t stride, size_t padding) {
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	for (size_t d = startDepth; d < endDepth; d++) {
		for (size_t y = 0; y < outputRows; y++) {
			for (size_t x = 0; x < outputCols; x++) {
				float sum = 0.0f;
				for (size_t ky = 0; ky < kernel.GetRows(); ky++) {
					for (size_t kx = 0; kx < kernel.GetCols(); kx++) {
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(ky, kx);
						}
					}
				}
				output.SetAt(y, x, d, output.GetAt(y, x, d) + sum);
			}
		}
	}
}

void Convolution(Tensor3D& output, const Tensor3D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);
	size_t outputDepth = output.GetDepth();

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor size!");
	assert(output.GetDepth() == input.GetDepth() && "Invalid output tensor depth!");

	if (output.IsOnDevice() != input.IsOnDevice() || output.IsOnDevice() != kernel.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (output.IsOnDevice())
	{
		accelerator::CudaConvolution(&output, &input, &kernel, stride, padding);
		return;
	}

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int depthPerThread = outputDepth / numThreads;
	int extraDepth = outputDepth % numThreads;

	std::vector<std::future<void>> tasks;
	size_t startDepth = 0;
	for (size_t i = 0; i < numThreads; i++) {
		size_t endDepth = startDepth + depthPerThread + (i < extraDepth ? 1 : 0);

		tasks.emplace_back(
			pool->enqueue(AsyncConvolution, startDepth, endDepth, std::ref(output), std::cref(input), std::cref(kernel), stride, padding)
		);

		startDepth = endDepth;
	}

	for (auto& task : tasks) {
		task.get(); // Wait for all tasks to complete
	}
#else
	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		for (size_t y = 0; y < outputRows; y++)
		{
			for (size_t x = 0; x < outputCols; x++)
			{
				float sum = 0.0f;
				for (size_t ky = 0; ky < kernel.GetRows(); ky++)
				{
					for (size_t kx = 0; kx < kernel.GetCols(); kx++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(ky, kx);
						}

					}
				}
				output.SetAt(y, x, d, output.GetAt(y, x, d) + sum);
			}
		}
	}
#endif
}

void ConvolutionKernelFlip(Tensor3D& output, const Tensor3D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor size!");
	assert(output.GetDepth() == input.GetDepth() && "Invalid output tensor depth!");


	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		for (size_t y = 0; y < outputRows; y++)
		{
			for (size_t x = 0; x < outputCols; x++)
			{
				float sum = 0.0f;
				for (size_t ky = 0; ky < kernel.GetRows(); ky++)
				{
					for (size_t kx = 0; kx < kernel.GetCols(); kx++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						size_t flippedKy = kernel.GetRows() - 1 - ky;
						size_t flippedKx = kernel.GetCols() - 1 - kx;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX, d) * kernel.GetAt(flippedKy, flippedKx);
						}

					}
				}
				output.SetAt(y, x, d, output.GetAt(y, x, d) + sum);
			}
		}
	}
}

void AsyncConvolutionKernelFlip(size_t startDepth, size_t endDepth, Tensor3D& output, const Tensor2D& input, const Tensor3D& kernel, size_t stride, size_t padding) {
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	for (size_t d = startDepth; d < endDepth; d++) {
		for (size_t y = 0; y < outputRows; y++) {
			for (size_t x = 0; x < outputCols; x++) {
				float sum = 0.0f;
				for (size_t ky = 0; ky < kernel.GetRows(); ky++) {
					for (size_t kx = 0; kx < kernel.GetCols(); kx++) {
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						size_t flippedKy = kernel.GetRows() - 1 - ky;
						size_t flippedKx = kernel.GetCols() - 1 - kx;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX) * kernel.GetAt(flippedKy, flippedKx, d);
						}
					}
				}
				output.SetAt(y, x, d, output.GetAt(y, x, d) + sum);
			}
		}
	}
}

void ConvolutionKernelFlip(Tensor3D& output, const Tensor2D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);
	size_t outputDepth = output.GetDepth();

	assert(output.GetRows() == outputRows && output.GetCols() == outputCols && "Invalid output tensor size!");
	assert(output.GetDepth() == kernel.GetDepth() && "Invalid output tensor depth!");

	if (output.IsOnDevice() != input.IsOnDevice() || output.IsOnDevice() != kernel.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (output.IsOnDevice())
	{
		accelerator::CudaConvolutionKernelFlip(&output, &input, &kernel, stride, padding);
		return;
	}

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int depthPerThread = outputDepth / numThreads;
	int extraDepth = outputDepth % numThreads;

	std::vector<std::future<void>> tasks;
	size_t startDepth = 0;
	for (size_t i = 0; i < numThreads; i++) {
		size_t endDepth = startDepth + depthPerThread + (i < extraDepth ? 1 : 0);

		tasks.emplace_back(
			pool->enqueue(AsyncConvolutionKernelFlip, startDepth, endDepth, std::ref(output), std::cref(input), std::cref(kernel), stride, padding)
		);

		startDepth = endDepth;
	}

	for (auto& task : tasks) {
		task.get(); // Wait for all tasks to complete
	}
#else
	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		for (size_t y = 0; y < outputRows; y++)
		{
			for (size_t x = 0; x < outputCols; x++)
			{
				float sum = 0.0f;
				for (size_t ky = 0; ky < kernel.GetRows(); ky++)
				{
					for (size_t kx = 0; kx < kernel.GetCols(); kx++)
					{
						int posY = y * stride + ky - padding;
						int posX = x * stride + kx - padding;

						size_t flippedKy = kernel.GetRows() - 1 - ky;
						size_t flippedKx = kernel.GetCols() - 1 - kx;

						if (posY >= 0 && posY < input.GetRows() && posX >= 0 && posX < input.GetCols()) {
							sum += input.GetAt(posY, posX) * kernel.GetAt(flippedKy, flippedKx, d);
						}

					}
				}
				output.SetAt(y, x, d, output.GetAt(y, x, d) + sum);
			}
		}
	}
#endif
}


Tensor2D Convolution(const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	Tensor2D output(outputRows, outputCols);

	Convolution(output, input, kernel, stride, padding);

	return output;
}

Tensor2D ConvolutionKernelFlip(const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	Tensor2D output(outputRows, outputCols);

	ConvolutionKernelFlip(output, input, kernel, stride, padding);

	return output;
}

Tensor2D Convolution(const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	Tensor2D output(outputRows, outputCols);

	Convolution(output, input, kernel, stride, padding);

	return output;
}

Tensor2D ConvolutionKernelFlip(const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding)
{
	size_t outputRows = CalcConvSize(input.GetRows(), kernel.GetRows(), stride, padding);
	size_t outputCols = CalcConvSize(input.GetCols(), kernel.GetCols(), stride, padding);

	Tensor2D output(outputRows, outputCols);

	ConvolutionKernelFlip(output, input, kernel, stride, padding);

	return output;
}

void AsyncMaxPool(size_t startDepth, size_t endDepth, Tensor3D& output, const Tensor3D& input, size_t poolHeight, size_t poolWidth) 
{
	for (size_t d = startDepth; d < endDepth; d++) 
	{
		Tensor2D slice = CreateWatcher((Tensor3D&)input, d);  // !
		for (size_t r = 0; r < output.GetRows(); r++)
		{
			for (size_t c = 0; c < output.GetCols(); c++)
			{
				Tensor2D window = CreateWatcher(slice, r * poolHeight, c * poolWidth, poolHeight, poolWidth, 0, 0);
				auto pos = MaxPos(window);

				output.SetAt(r, c, d, window.GetAt(pos.first, pos.second));
			}
		}
	}
}

Tensor3D MaxPool(const Tensor3D& input, size_t poolHeight, size_t poolWidth)
{
	Tensor3D output(input.GetRows() / poolHeight, input.GetCols() / poolWidth, input.GetDepth(), 0.0f, input.IsOnDevice());

	if (output.IsOnDevice() != input.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (input.IsOnDevice())
	{
		accelerator::CudaMaxPool(&output, &input, poolHeight, poolWidth);
		return output;
	}

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();

	// Calculate depth per thread more accurately
	int depthPerThread = output.GetDepth() / numThreads;
	int extraDepth = output.GetDepth() % numThreads; // Remaining depths that need to be distributed

	std::vector<std::future<void>> tasks;
	size_t startDepth = 0;
	for (size_t i = 0; i < numThreads; i++) {
		// Assign extra depth to the first 'extraDepth' threads
		size_t endDepth = startDepth + depthPerThread + (i < extraDepth ? 1 : 0);

		tasks.emplace_back(
			pool->enqueue(AsyncMaxPool, startDepth, endDepth, std::ref(output), std::cref(input), poolHeight, poolWidth)
		);

		startDepth = endDepth;
	}

	for (auto& task : tasks) {
		task.get();
	}

#else
	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		Tensor2D slice = CreateWatcher((Tensor3D&)input, d);  // !

		for (size_t r = 0; r < output.GetRows(); r++)
		{
			for (size_t c = 0; c < output.GetCols(); c++)
			{
				Tensor2D window = CreateWatcher(slice, r * poolHeight, c * poolWidth, poolHeight, poolWidth, 0, 0);
				auto pos = MaxPos(window);

				output.SetAt(r, c, d, window.GetAt(pos.first, pos.second));
			}
		}
	}
#endif // ASYNC

	return output;
}

void AsyncDistributeReverseMaxPool(size_t startDepth, size_t endDepth, Tensor3D& distributed, const Tensor3D& input, const Tensor3D& output, size_t poolHeight, size_t poolWidth)
{
	for (size_t d = startDepth; d < endDepth; d++)
	{
		Tensor2D inputSlice = CreateWatcher((Tensor3D&)input, d);  // !
		Tensor2D distributedSlice = CreateWatcher(distributed, d);

		for (size_t r = 0; r < output.GetRows(); r++)
		{
			for (size_t c = 0; c < output.GetCols(); c++)
			{
				Tensor2D inputWindow = CreateWatcher(inputSlice, r * poolHeight, c * poolWidth, poolHeight, poolWidth, 0, 0);
				Tensor2D distributedWindow = CreateWatcher(distributedSlice, r * poolHeight, c * poolWidth, poolHeight, poolWidth, 0, 0);
				auto pos = MaxPos(inputWindow);

				distributedWindow.SetAt(pos.first, pos.second, output.GetAt(r, c, d));
			}
		}
	}
}


void DistributeReverseMaxPool(Tensor3D& distributed, const Tensor3D& input, const Tensor3D& output, size_t poolHeight, size_t poolWidth)
{
	if (output.IsOnDevice() != input.IsOnDevice() || output.IsOnDevice() != distributed.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (input.IsOnDevice())
	{
		accelerator::CudaDistributeReverseMaxPool(&distributed, &input, &output, poolHeight, poolWidth);
		return;
	}

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();

	// Calculate depth per thread more accurately
	int depthPerThread = output.GetDepth() / numThreads;
	int extraDepth = output.GetDepth() % numThreads; // Remaining depths that need to be distributed

	std::vector<std::future<void>> tasks;
	size_t startDepth = 0;
	for (size_t i = 0; i < numThreads; i++) {
		// Assign extra depth to the first 'extraDepth' threads
		size_t endDepth = startDepth + depthPerThread + (i < extraDepth ? 1 : 0);

		tasks.emplace_back(
			pool->enqueue(AsyncDistributeReverseMaxPool, startDepth, endDepth, std::ref(distributed), std::cref(input), std::cref(output), poolHeight, poolWidth)
		);

		startDepth = endDepth;
	}

	for (auto& task : tasks) {
		task.get();
	}
#else
	for (size_t d = 0; d < distributed.GetDepth(); d++)
	{
		Tensor2D inputSlice = CreateWatcher((Tensor3D&)input, d);  // !
		Tensor2D distributedSlice = CreateWatcher(distributed, d);

		for (size_t r = 0; r < output.GetRows(); r++)
		{
			for (size_t c = 0; c < output.GetCols(); c++)
			{
				Tensor2D inputWindow = CreateWatcher(inputSlice, r * poolHeight, c * poolWidth, poolHeight, poolWidth, 0, 0);
				Tensor2D distributedWindow = CreateWatcher(distributedSlice, r * poolHeight, c * poolWidth, poolHeight, poolWidth, 0, 0);
				auto pos = MaxPos(inputWindow);

				distributedWindow.SetAt(pos.first, pos.second, output.GetAt(r, c, d));
			}
		}
	}
#endif // ASYNC
}

void AsyncNearestUpsample(size_t startDepth, size_t endDepth, Tensor3D& output, const Tensor3D& input, size_t upsampleHeight, size_t upsampleWidth)
{
	for (size_t d = startDepth; d < endDepth; d++)
	{
		Tensor2D inputSlice = CreateWatcher((Tensor3D&)input, d);  // !
		Tensor2D outputSlice = CreateWatcher(output, d);

		for (size_t r = 0; r < input.GetRows(); r++)
		{
			for (size_t c = 0; c < input.GetCols(); c++)
			{
				float value = inputSlice.GetAt(r, c);
				Tensor2D outputWindow = CreateWatcher(outputSlice, r * upsampleHeight, c * upsampleWidth, upsampleHeight, upsampleWidth, 0, 0);
				for (size_t i = 0; i < outputWindow.GetSize(); i++)
				{
					size_t index = outputWindow.TraverseTo(i);
					outputWindow.GetData()[index] = value;
				}
			}
		}
	}
}

Tensor3D NearestUpsample(const Tensor3D& input, size_t upsampleHeight, size_t upsampleWidth)
{
	Tensor3D output(input.GetRows() * upsampleHeight, input.GetCols() * upsampleWidth, input.GetDepth(), 0.0f, input.IsOnDevice());

	if (input.IsOnDevice())
	{
		accelerator::CudaNearestUpsample(&output, &input, upsampleHeight, upsampleWidth);
		return output;
	}

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int depthPerThread = output.GetDepth() / numThreads;
	int extraDepth = output.GetDepth() % numThreads;

	std::vector<std::future<void>> tasks;
	size_t startDepth = 0;
	for (size_t i = 0; i < numThreads; i++) {
		size_t endDepth = startDepth + depthPerThread + (i < extraDepth ? 1 : 0);

		tasks.emplace_back(
			pool->enqueue(AsyncNearestUpsample, startDepth, endDepth, std::ref(output), std::cref(input), upsampleHeight, upsampleWidth)
		);

		startDepth = endDepth;
	}

	for (auto& task : tasks) {
		task.get();
	}
#else
	for (size_t d = 0; d < input.GetDepth(); d++)
	{
		Tensor2D inputSlice = CreateWatcher((Tensor3D&)input, d);  // !
		Tensor2D outputSlice = CreateWatcher(output, d);

		for (size_t r = 0; r < input.GetRows(); r++)
		{
			for (size_t c = 0; c < input.GetCols(); c++)
			{
				float value = inputSlice.GetAt(r, c);
				Tensor2D outputWindow = CreateWatcher(outputSlice, r * upsampleHeight, c * upsampleWidth, upsampleHeight, upsampleWidth, 0, 0);
				for (size_t i = 0; i < outputWindow.GetSize(); i++)
				{
					size_t index = outputWindow.TraverseTo(i);
					outputWindow.GetData()[index] = value;
				}
			}
		}
	}
#endif // ASYNC

	return output;
}

void AsyncDistributeReverseNearestUpsample(size_t startDepth, size_t endDepth, Tensor3D& distributed, const Tensor3D& output, size_t upsampleHeight, size_t upsampleWidth)
{
	for (size_t d = startDepth; d < endDepth; d++)
	{
		Tensor2D distributedSlice = CreateWatcher(distributed, d);
		Tensor2D costSlice = CreateWatcher((Tensor3D&)output, d);

		for (size_t r = 0; r < distributed.GetRows(); r++)
		{
			for (size_t c = 0; c < distributed.GetCols(); c++)
			{
				Tensor2D outputWindow = CreateWatcher(costSlice, r * upsampleHeight, c * upsampleWidth, upsampleHeight, upsampleWidth, 0, 0);

				float value = 0.0f;
				for (size_t i = 0; i < outputWindow.GetSize(); i++)
				{
					size_t index = outputWindow.TraverseTo(i);
					value += outputWindow.GetData()[index];
				}

				distributedSlice.SetAt(r, c, value);
			}
		}
	}
}

void DistributeReverseNearestUpsample(Tensor3D& distributed, const Tensor3D& output, size_t upsampleHeight, size_t upsampleWidth)
{

	if (distributed.IsOnDevice())
	{
		accelerator::CudaDistributeReverseNearestUpsample(&distributed, &output, upsampleHeight, upsampleWidth);
		return;
	}

#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int depthPerThread = output.GetDepth() / numThreads;
	int extraDepth = output.GetDepth() % numThreads;

	std::vector<std::future<void>> tasks;
	size_t startDepth = 0;
	for (size_t i = 0; i < numThreads; i++) {
		size_t endDepth = startDepth + depthPerThread + (i < extraDepth ? 1 : 0);

		tasks.emplace_back(
			pool->enqueue(AsyncDistributeReverseNearestUpsample, startDepth, endDepth, std::ref(distributed), std::cref(output), upsampleHeight, upsampleWidth)
		);

		startDepth = endDepth;
	}

	for (auto& task : tasks) {
		task.get();
	}
#else
	for (size_t d = 0; d < distributed.GetDepth(); d++)
	{
		Tensor2D distributedSlice = CreateWatcher(distributed, d);
		Tensor2D costSlice = CreateWatcher((Tensor3D&)output, d);

		for (size_t r = 0; r < distributed.GetRows(); r++)
		{
			for (size_t c = 0; c < distributed.GetCols(); c++)
			{
				Tensor2D outputWindow = CreateWatcher(costSlice, r * upsampleHeight, c * upsampleWidth, upsampleHeight, upsampleWidth, 0, 0);

				float value = 0.0f;
				for (size_t i = 0; i < outputWindow.GetSize(); i++)
				{
					size_t index = outputWindow.TraverseTo(i);
					value += outputWindow.GetData()[index];
				}

				distributedSlice.SetAt(r, c, value);
			}
		}
	}
#endif // ASYNC
}

void AsyncDropOut(size_t start, size_t end, Tensor* output, const Tensor* input, float dropoutRate, float retentionProb, Tensor* dropOutMask)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distr(0.0f, 1.0f);

	for (size_t i = start; i < end; i++)
	{
		float r = distr(gen);
		if (r > dropoutRate)
		{
			output->GetData()[i] = input->GetData()[i] / retentionProb;
			if (dropOutMask)
			{
				dropOutMask->GetData()[i] = 1.0f;
			}
		}
	}
}

Tensor3D DropOut(const Tensor3D& input, float dropoutRate, Tensor3D* dropOutMask)
{
	float retentionProb = 1.0f - dropoutRate;
	Tensor3D output = Tensor3D(input.GetRows(), input.GetCols(), input.GetDepth(), 0.0f, input.IsOnDevice());

	if (dropOutMask && (dropOutMask->GetRows() != input.GetRows() || dropOutMask->GetCols() != input.GetCols() || dropOutMask->GetDepth() != input.GetDepth()))
	{
		throw std::runtime_error("Dropout max dimension not match with input's.");
	}

	if (dropOutMask && dropOutMask->IsOnDevice() != input.IsOnDevice())
	{
		throw std::runtime_error("Tensors are not on the same device.");
	}

	if (output.IsOnDevice())
	{
		mogi::accelerator::CudaDropOut(&output, &input, dropoutRate, retentionProb, dropOutMask);
		return output;
	}
#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = pool->GetNumThreads();
	int dataPerThread = output.GetSize() / numThreads;
	int extraData= output.GetSize() % numThreads;

	std::vector<std::future<void>> tasks;
	size_t startDepth = 0;
	for (size_t i = 0; i < numThreads; i++) {
		size_t endDepth = startDepth + dataPerThread + (i < extraData ? 1 : 0);

		tasks.emplace_back(
			pool->enqueue(AsyncDropOut, startDepth, endDepth, &output, &input, dropoutRate, retentionProb, dropOutMask)
		);

		startDepth = endDepth;
	}

	for (auto& task : tasks) {
		task.get();
	}
#else
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distr(0.0f, 1.0f);

	for (size_t i = 0; i < input.GetSize(); i++)
	{
		float r = distr(gen);
		if (r > dropoutRate)
		{
			output.GetData()[i] = input.GetData()[i] / retentionProb;
		}
	}

#endif // ASYNC
	return output;
}

namespace_end