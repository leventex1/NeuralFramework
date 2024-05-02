#pragma once

#include "src/AcceleratorCore.h"

#include <Mogi.h>


namespace_accelerator_start

LIBRARY_ACCELERATOR_API void CudaDealloc(float* devicePtr);
LIBRARY_ACCELERATOR_API float* CudaAlloc(size_t count);
LIBRARY_ACCELERATOR_API void CudaCopyHostToDevice(float* device, const float* host, size_t size);
LIBRARY_ACCELERATOR_API void CudaCopyDeviceToHost(float* host, const float* device, size_t size);
LIBRARY_ACCELERATOR_API void CudaCopyDeviceToDevice(float* deviceDst, const float* deviceSrc, size_t size);
LIBRARY_ACCELERATOR_API void CudaCopyHostToDevice(Tensor* device, const Tensor* host);
LIBRARY_ACCELERATOR_API void CudaCopyDeviceToHost(Tensor* host, const Tensor* device);
LIBRARY_ACCELERATOR_API void CudaCopyDeviceToDevice(Tensor* deviceDst, const Tensor* deviceSrc);
LIBRARY_ACCELERATOR_API void CudaMemSet(Tensor* device, size_t size, float value);

LIBRARY_ACCELERATOR_API Tensor2D CudaRandom2D(size_t rows, size_t cols, float min, float max);

LIBRARY_ACCELERATOR_API void CudaAdd(Tensor* device, float value);
LIBRARY_ACCELERATOR_API void CudaSub(Tensor* device, float value);
LIBRARY_ACCELERATOR_API void CudaMult(Tensor* device, float value);
LIBRARY_ACCELERATOR_API void CudaDiv(Tensor* device, float value);
LIBRARY_ACCELERATOR_API void CudaAdd(Tensor* device, const Tensor* other);
LIBRARY_ACCELERATOR_API void CudaSub(Tensor* device, const Tensor* other);
LIBRARY_ACCELERATOR_API void CudaMult(Tensor* device, const Tensor* other);
LIBRARY_ACCELERATOR_API void CudaDiv(Tensor* device, const Tensor* other);

LIBRARY_ACCELERATOR_API void CudaSigmoid(Tensor* device);
LIBRARY_ACCELERATOR_API void CudaDiffSigmoid(Tensor* device);
LIBRARY_ACCELERATOR_API void CudaRelU(Tensor* device, float alpha);
LIBRARY_ACCELERATOR_API void CudaDiffRelU(Tensor* device, float alpha);

LIBRARY_ACCELERATOR_API void CudaAdamOptimization(Tensor* params, Tensor* gradients, Tensor* firstMoments, Tensor* secondMoments, float b1, float b2, float ep, size_t timeStep, float learningRate);
LIBRARY_ACCELERATOR_API Tensor2D CudaCorrectedMoments(Tensor* moments, float b, float timeStep);
LIBRARY_ACCELERATOR_API Tensor2D CudaCorrectedGradient(Tensor* firstMoments, Tensor* secondMoments, float learningRate, float epsilon = 1.0f / 100000000);

LIBRARY_ACCELERATOR_API Tensor2D MatrixMultCUDA(const Tensor2D& left, const Tensor2D& right);
LIBRARY_ACCELERATOR_API Tensor2D MatrixMultRightTransposeCUDA(const Tensor2D& left, const Tensor2D& right);
LIBRARY_ACCELERATOR_API Tensor2D MatrixMultLeftTransposeCUDA(const Tensor2D& left, const Tensor2D& right);

LIBRARY_ACCELERATOR_API void CudaConvolution(Tensor2D* output, const Tensor3D* input, const Tensor3D* kernel, size_t stride, size_t padding);
LIBRARY_ACCELERATOR_API void CudaConvolution(Tensor3D* output, const Tensor3D* input, const Tensor2D* kernel, size_t stride, size_t padding);
LIBRARY_ACCELERATOR_API void CudaConvolutionKernelFlip(Tensor3D* output, const Tensor2D* input, const Tensor3D* kernel, size_t stride, size_t padding);

LIBRARY_ACCELERATOR_API void CudaMaxPool(Tensor3D* output, const Tensor3D* input, size_t poolHeight, size_t poolWidth);
LIBRARY_ACCELERATOR_API void CudaDistributeReverseMaxPool(Tensor3D* distributed, const Tensor3D* input, const Tensor3D* output, size_t poolHeight, size_t poolWidth);

LIBRARY_ACCELERATOR_API void CudaNearestUpsample(Tensor3D* output, const Tensor3D* input, size_t upsampleHeight, size_t upsampleWidth);
LIBRARY_ACCELERATOR_API void CudaDistributeReverseNearestUpsample(Tensor3D* distributed, const Tensor3D* output, size_t upsampleHeight, size_t upsampleWidth);

LIBRARY_ACCELERATOR_API void CudaDropOut(Tensor* output, const Tensor* input, float dropoutRate, float retentionProb, Tensor* dropOutMask=nullptr);

LIBRARY_ACCELERATOR_API Tensor3D CudaCrossEntropyLoss(const Tensor3D* target, const Tensor3D* predictions);

namespace_accelerator_end