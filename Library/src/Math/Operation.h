#pragma once
#include <functional>
#include "Core.h"
#include "Tensor2D.h"
#include "Tensor3D.h"


namespace_start

LIBRARY_API float Sum(const Tensor* tensor);

LIBRARY_API Tensor2D SliceTensor(const Tensor3D& tensor, size_t depth);
LIBRARY_API Tensor2D CreateWatcher(Tensor2D& tensor, size_t row, size_t col, size_t rows, size_t cols, size_t skipRows, size_t skipCols);
LIBRARY_API Tensor3D CreateWatcher(Tensor2D& tensor);
LIBRARY_API Tensor2D CreateWatcher(Tensor3D& tensor, size_t depth);
LIBRARY_API Tensor3D CreateWatcher(Tensor3D& tensor, size_t fromDepth, size_t depth);

LIBRARY_API Tensor2D Random2D(size_t rows, size_t cols, float min, float max, bool useCuda=false);
LIBRARY_API Tensor3D Random3D(size_t rows, size_t cols, size_t depth, float min, float max);

LIBRARY_API Tensor2D Map(const Tensor2D& t, std::function<float(float v)> mapper);
LIBRARY_API Tensor3D Map(const Tensor3D& t, std::function<float(float v)> mapper);

LIBRARY_API Tensor2D Add(const Tensor2D& t1, const Tensor2D& t2);
LIBRARY_API Tensor2D Sub(const Tensor2D& t1, const Tensor2D& t2);
LIBRARY_API Tensor2D Mult(const Tensor2D& t1, const Tensor2D& t2);
LIBRARY_API Tensor2D Div(const Tensor2D& t1, const Tensor2D& t2);
LIBRARY_API Tensor2D Add(const Tensor2D& t1, float value);
LIBRARY_API Tensor2D Sub(const Tensor2D& t1, float value);
LIBRARY_API Tensor2D Mult(const Tensor2D& t1, float value);
LIBRARY_API Tensor2D Div(const Tensor2D& t1, float value);

LIBRARY_API std::pair<size_t, size_t> MaxPos(const Tensor2D& tensor);

LIBRARY_API Tensor2D Transpose(const Tensor2D& t);

LIBRARY_API Tensor2D MatrixMult(const Tensor2D& left, const Tensor2D& right);
// Calculates the matrix multiplication, but take the left matrix as a transpose matrix without extra calculation.
LIBRARY_API Tensor2D MatrixMultLeftTranspose(const Tensor2D& left, const Tensor2D& right);
// Calculates the matrix multiplication, but take the right matrix as a transpose matrix without extra calculation.
LIBRARY_API Tensor2D MatrixMultRightTranspose(const Tensor2D& left, const Tensor2D& right);

LIBRARY_API size_t CalcConvSize(size_t inputSize, size_t kernelSize, size_t stride, size_t padding);
LIBRARY_API float KernelOperation(const Tensor2D& window, const Tensor2D& kernel);

// Preform a convolutional operation on input with kernel with a zero padding and adds to output.
LIBRARY_API void Convolution(Tensor2D& output, const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding);
// Preform a convolutional operation on input with a 180 flipped kernel with a zero padding and adds to the output.
LIBRARY_API void ConvolutionKernelFlip(Tensor2D& output, const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding);
// Preform a convolutional operation in 3D with zero padding, sum up all the outputs from the 2D slices into the 2D output, adds the output.
LIBRARY_API void Convolution(Tensor2D& output, const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding);
// Preform a convolutional operation in 3D with a 180 flipped kernel and zero padding, sum up all the outputs from the 2D slices into the 2D output, adds to the output.
LIBRARY_API void ConvolutionKernelFlip(Tensor2D& output, const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding);
// Preform a convolutional operation in all 2D slices in the input with the 2D kernel, with zero padding. Adds to the output.
LIBRARY_API void Convolution(Tensor3D& output, const Tensor3D& input, const Tensor2D& kernel, size_t stride, size_t padding);
LIBRARY_API void ConvolutionKernelFlip(Tensor3D& output, const Tensor3D& input, const Tensor2D& kernel, size_t stride, size_t padding);

// Preform a convolutional operation on the input with all 2D kernel slices, with zero padding. Adds to the output.
LIBRARY_API void ConvolutionKernelFlip(Tensor3D& output, const Tensor2D& input, const Tensor3D& kernel, size_t stride, size_t padding);


// Preform a convolutional operation on input with kernel with a zero padding.
LIBRARY_API Tensor2D Convolution(const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding);
LIBRARY_API Tensor2D ConvolutionKernelFlip(const Tensor2D& input, const Tensor2D& kernel, size_t stride, size_t padding);
LIBRARY_API Tensor2D Convolution(const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding);
LIBRARY_API Tensor2D ConvolutionKernelFlip(const Tensor3D& input, const Tensor3D& kernel, size_t stride, size_t padding);

LIBRARY_API Tensor3D MaxPool(const Tensor3D& input, size_t poolHeight, size_t poolWidth);
LIBRARY_API void DistributeReverseMaxPool(Tensor3D& distributed, const Tensor3D& input, const Tensor3D& output, size_t poolHeight, size_t poolWidth);

LIBRARY_API Tensor3D NearestUpsample(const Tensor3D& input, size_t upsampleHeight, size_t upsampleWidth);
LIBRARY_API void DistributeReverseNearestUpsample(Tensor3D& distributed, const Tensor3D& output, size_t upsampleHeight, size_t upsampleWidht);

LIBRARY_API Tensor3D DropOut(const Tensor3D& input, float dropoutRate, Tensor3D* dropOutMask=nullptr);

namespace_end
