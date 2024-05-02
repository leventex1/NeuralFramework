#include "cuda_runtime.h"


__global__ void CopyKernel(float* tensor, float value, size_t size);
__global__ void CopyKernel(float* tensorDst, const float* tensorSrc, size_t size);

__global__ void AddKernel(float* tensor, size_t size, float value);
__global__ void SubKernel(float* tensor, size_t size, float value);
__global__ void MultKernel(float* tensor, size_t size, float value);
__global__ void DivKernel(float* tensor, size_t size, float value);

__global__ void AddKernel(float* tensor, const float* other, size_t size);
__global__ void SubKernel(float* tensor, const float* other, size_t size);
__global__ void MultKernel(float* tensor, const float* other, size_t size);
__global__ void DivKernel(float* tensor, const float* other, size_t size);

__global__ void SigmoidKernel(float* tensor, size_t size);
__global__ void DiffSigmoidKernel(float* tensor, size_t size);
__global__ void RelUKernel(float* tensor, size_t size, float alpha);
__global__ void DiffRelUKernel(float* tensor, size_t size, float alpha);

__global__ void AdamOptimizationKernel(float* params, float* gradients, float* firstMoments, float* secondMoments, size_t size, float b1, float b2, float epsilon, size_t timeStep, float learningRate);
__global__ void CorrectedMomentsKernel(float* tensor, size_t size, float b, float timeStep);
__global__ void CorrectedGradientKernel(float* res, float* firstMoments, float* secondMoments, size_t size, float learningRate, float epsilon);

__global__ void MatrixMultKernel(const float* A, const float* B, float* C, int aRows, int aCols, int bCols);
__global__ void MatrixMultRightTranposeKernel(const float* A, const float* B, float* C, int aRows, int aCols, int bCols);
__global__ void MatrixMultLeftTranposeKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BCols);

__global__ void ConvolutionKernel(
	const float* input, const float* kernel, float* output,
	size_t inputRows, size_t inputCols, size_t inputDepth,
	size_t kernelRows, size_t kernelCols, size_t kernelDepth,
	size_t outputRows, size_t outputCols,
	size_t stride, size_t padding);
__global__ void Convolution3DKernel(
	const float* input, const float* kernel, float* output,
	size_t inputRows, size_t inputCols, size_t inputDepth,
	size_t kernelRows, size_t kernelCols,
	size_t outputRows, size_t outputCols, size_t outputDepth,
	size_t stride, size_t padding);
__global__ void Convolution3DBackKernelFlipKernel(
	const float* input, const float* kernel, float* output,
	size_t inputRows, size_t inputCols,
	size_t kernelRows, size_t kernelCols, size_t kernelDepth,
	size_t outputRows, size_t outputCols, size_t outputDepth,
	size_t stride, size_t padding);

__global__ void MaxPoolKernel(
	const float* input, float* output,
	size_t inputRows, size_t inputCols, size_t inputDepth,
	size_t outputRows, size_t outputCols, size_t outputDepth,
	size_t poolHeight, size_t poolWidth);
__global__ void DistributedReverseMaxPoolKernel(
	const float* input, const float* output, float* distributed,
	size_t inputRows, size_t inputCols, size_t inputDepth,
	size_t outputRows, size_t outputCols, size_t outputDepth,
	size_t distributedRows, size_t distributedCols, size_t distributedDepth,
	size_t poolHeight, size_t poolWidth);

__global__ void NearestUpsampleKernel(
	const float* input, float* output,
	size_t inputRows, size_t inputCols, size_t inputDepth,
	size_t outputRows, size_t outputCols, size_t outputDepth,
	size_t upsampleHeight, size_t upsampleWidth);
__global__ void DistributeReverseNearestUpsampleKernel(
	const float* output, float* distributed,
	size_t outputRows, size_t outputCols, size_t outputDepth,
	size_t distributedRows, size_t distributedCols, size_t distributedDepth,
	size_t upsampleHeight, size_t upsampleWidth);

__global__ void DropOutKernel(const float* input, float* output, size_t size, float dropoutRate, float retentionProb, float* dropOutMask);

__global__ void CrossEntropyLossKernel(const float* targets, const float* predictions, float* loss, size_t size);

__global__ void RandomizeKernel(float* output, size_t size, float min, float max);