#include "MogiAccelerator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

#include "src/Kernels.h"


namespace_accelerator_start

void CudaDealloc(float* devicePtr)
{
    cudaFree(devicePtr);
}

float* CudaAlloc(size_t count)
{
    float* devicePtr = nullptr;
    size_t size = count * sizeof(float);
    cudaMalloc(&devicePtr, size);
    return devicePtr;
}

void CudaCopyHostToDevice(float* device, const float* host, size_t size)
{
    cudaMemcpy(device, host, size * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaCopyDeviceToHost(float* host, const float* device, size_t size)
{
    cudaMemcpy(host, device, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void CudaCopyDeviceToDevice(float* deviceDst, const float* deviceSrc, size_t size)
{
    cudaMemcpy(deviceDst, deviceSrc, size * sizeof(float), cudaMemcpyDeviceToDevice);
}

void CudaCopyHostToDevice(Tensor* device, Tensor* host)
{
    cudaMemcpy(device->GetData(), host->GetData(), host->GetSize() * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaCopyDeviceToHost(Tensor* host, Tensor* device)
{
    cudaMemcpy(host->GetData(), device->GetData(), device->GetSize() * sizeof(float), cudaMemcpyDeviceToHost);
}

void CudaCopyDeviceToDevice(Tensor* deviceDst, const Tensor* deviceSrc)
{
    int blockSize = std::min(size_t(256), deviceSrc->GetSize());
    int numBlocks = (deviceSrc->GetSize() + blockSize - 1) / blockSize;
    CopyKernel << <numBlocks, blockSize >> > (deviceDst->GetData(), deviceSrc->GetData(), deviceSrc->GetSize());
    cudaDeviceSynchronize();
}

void CudaMemSet(Tensor* device, size_t size, float value)
{
    int blockSize = std::min(size_t(256), size);
    int numBlocks = (size + blockSize - 1) / blockSize;
    CopyKernel<<<numBlocks, blockSize>>>(device->GetData(), value, size);
    cudaDeviceSynchronize();
}

float* ToDevicePtr(const Tensor2D& tensor, bool copy=true)
{
    float* dPtr;
    size_t size = tensor.GetSize() * sizeof(float);

    cudaMalloc(&dPtr, size);
    if (copy)
    {
        cudaMemcpy(dPtr, tensor.GetData(), size, cudaMemcpyHostToDevice);
    }

    return dPtr;
}

void CopyToHost(Tensor2D& dest, float* deviceSrource)
{
    cudaMemcpy(dest.GetData(), deviceSrource, dest.GetSize() * sizeof(float), cudaMemcpyDeviceToHost);
}

Tensor2D CudaRandom2D(size_t rows, size_t cols, float min, float max)
{
    Tensor2D res(rows, cols, 0.0f, true);
    int blockSize = std::min(size_t(256), res.GetSize());
    int numBlocks = (res.GetSize() + blockSize - 1) / blockSize;
    RandomizeKernel<<<numBlocks, blockSize>>>(res.GetData(), res.GetSize(), min, max);
    return res;
}

void CudaAdd(Tensor* device, float value)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    AddKernel << <numBlocks, blockSize >> > (device->GetData(), device->GetSize(), value);
}

void CudaSub(Tensor* device, float value)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    SubKernel << <numBlocks, blockSize >> > (device->GetData(), device->GetSize(), value);
}

void CudaMult(Tensor* device, float value)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    MultKernel << <numBlocks, blockSize >> > (device->GetData(), device->GetSize(), value);
}

void CudaDiv(Tensor* device, float value)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    DivKernel << <numBlocks, blockSize >> > (device->GetData(), device->GetSize(), value);
}

void CudaAdd(Tensor* device, const Tensor* other)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    AddKernel << <numBlocks, blockSize >> > (device->GetData(), other->GetData(), device->GetSize());
}

void CudaSub(Tensor* device, const Tensor* other)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    SubKernel << <numBlocks, blockSize >> > (device->GetData(), other->GetData(), device->GetSize());
}

void CudaMult(Tensor* device, const Tensor* other)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    MultKernel << <numBlocks, blockSize >> > (device->GetData(), other->GetData(), device->GetSize());
}

void CudaDiv(Tensor* device, const Tensor* other)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    DivKernel << <numBlocks, blockSize >> > (device->GetData(), other->GetData(), device->GetSize());
}

void CudaSigmoid(Tensor* device)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    SigmoidKernel << <numBlocks, blockSize >> > (device->GetData(), device->GetSize());
}

void CudaDiffSigmoid(Tensor* device)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    DiffSigmoidKernel << <numBlocks, blockSize >> > (device->GetData(), device->GetSize());
}

void CudaRelU(Tensor* device, float alpha)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    RelUKernel << <numBlocks, blockSize >> > (device->GetData(), device->GetSize(), alpha);
}

void CudaDiffRelU(Tensor* device, float alpha)
{
    int blockSize = std::min(size_t(256), device->GetSize());
    int numBlocks = (device->GetSize() + blockSize - 1) / blockSize;
    DiffRelUKernel << <numBlocks, blockSize >> > (device->GetData(), device->GetSize(), alpha);
}

void CudaAdamOptimization(Tensor* params, Tensor* gradients, Tensor* firstMoments, Tensor* secondMoments, float b1, float b2, float ep, size_t timeStep, float learningRate)
{
    int blockSize = std::min(size_t(256), params->GetSize());
    int numBlocks = (params->GetSize() + blockSize - 1) / blockSize;
    AdamOptimizationKernel<<<numBlocks, blockSize>>>(
        params->GetData(), gradients->GetData(), firstMoments->GetData(), secondMoments->GetData(), 
        params->GetSize(), b1, b2, ep, timeStep, learningRate);
}

Tensor2D CudaCorrectedMoments(Tensor* moments, float b, float timeStep)
{
    Tensor2D res(moments->GetSize(), 1, (const float*)moments->GetData(), true);
    int blockSize = std::min(size_t(256), moments->GetSize());
    int numBlocks = (moments->GetSize() + blockSize - 1) / blockSize;
    CorrectedMomentsKernel<<<numBlocks, blockSize>>>(res.GetData(), res.GetSize(), b, timeStep);
    return res;
}

Tensor2D CudaCorrectedGradient(Tensor* firstMoments, Tensor* secondMoments, float learningRate, float epsilon)
{
    Tensor2D res(firstMoments->GetSize(), 1, 0.0f, true);
    int blockSize = std::min(size_t(256), firstMoments->GetSize());
    int numBlocks = (firstMoments->GetSize() + blockSize - 1) / blockSize;
    CorrectedGradientKernel<<<numBlocks, blockSize>>>(res.GetData(), firstMoments->GetData(), secondMoments->GetData(), res.GetSize(), learningRate, epsilon);
    return res;
}

Tensor2D MatrixMultCUDA(const Tensor2D& left, const Tensor2D& right)
{
    if (left.GetCols() != right.GetRows())
    {
        throw -1;
    }

    Tensor2D res(left.GetRows(), right.GetCols(), 0.0f, true);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((right.GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (left.GetRows()  + threadsPerBlock.y - 1) / threadsPerBlock.y);
    MatrixMultKernel<< <blocksPerGrid, threadsPerBlock >> > (left.GetData(), right.GetData(), res.GetData(), left.GetRows(), left.GetCols(), right.GetCols());

    return res;
}

Tensor2D MatrixMultRightTransposeCUDA(const Tensor2D& left, const Tensor2D& right)
{
    if (left.GetCols() != right.GetCols())
    {
        throw -1;
    }

    Tensor2D res(left.GetRows(), right.GetRows(), 0.0f, true);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((right.GetRows() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (left.GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatrixMultRightTranposeKernel << <blocksPerGrid, threadsPerBlock >> > (left.GetData(), right.GetData(), res.GetData(), left.GetRows(), left.GetCols(), right.GetRows());

    return res;
}

Tensor2D MatrixMultLeftTransposeCUDA(const Tensor2D& left, const Tensor2D& right)
{
    if (left.GetRows() != right.GetRows())
    {
        throw - 1;
    }

    Tensor2D res(left.GetCols(), right.GetCols(), 0.0f, true);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((right.GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (left.GetCols() + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatrixMultLeftTranposeKernel<< <blocksPerGrid, threadsPerBlock >> > (left.GetData(), right.GetData(), res.GetData(), left.GetRows(), left.GetCols(), right.GetCols());

    return res;
}

void CudaConvolution(Tensor2D* output, const Tensor3D* input, const Tensor3D* kernel, size_t stride, size_t padding)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (output->GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output->GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    ConvolutionKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input->GetData(), kernel->GetData(), output->GetData(), 
        input->GetRows(), input->GetCols(), input->GetDepth(),
        kernel->GetRows(), kernel->GetCols(), kernel->GetDepth(),
        output->GetRows(), output->GetCols(),
        stride, padding);
}

void CudaConvolution(Tensor3D* output, const Tensor3D* input, const Tensor2D* kernel, size_t stride, size_t padding)
{
    dim3 threadsPerBlock(16, 16, 4);
    dim3 blocksPerGrid(
        (output->GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x, 
        (output->GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (output->GetDepth() + threadsPerBlock.z - 1) / threadsPerBlock.z);
    Convolution3DKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input->GetData(), kernel->GetData(), output->GetData(),
        input->GetRows(), input->GetCols(), input->GetDepth(),
        kernel->GetRows(), kernel->GetCols(),
        output->GetRows(), output->GetCols(), output->GetDepth(),
        stride, padding);
}

void CudaConvolutionKernelFlip(Tensor3D* output, const Tensor2D* input, const Tensor3D* kernel, size_t stride, size_t padding)
{
    dim3 threadsPerBlock(16, 16, 4);
    dim3 blocksPerGrid(
        (output->GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output->GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (output->GetDepth() + threadsPerBlock.z - 1) / threadsPerBlock.z);
    Convolution3DBackKernelFlipKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input->GetData(), kernel->GetData(), output->GetData(),
        input->GetRows(), input->GetCols(),
        kernel->GetRows(), kernel->GetCols(), kernel->GetDepth(),
        output->GetRows(), output->GetCols(), output->GetDepth(),
        stride, padding);
}

void CudaMaxPool(Tensor3D* output, const Tensor3D* input, size_t poolHeight, size_t poolWidth)
{
    dim3 threadsPerBlock(16, 16, 4);
    dim3 blocksPerGrid(
        (output->GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output->GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (output->GetDepth() + threadsPerBlock.z - 1) / threadsPerBlock.z);
    MaxPoolKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input->GetData(), output->GetData(),
        input->GetRows(), input->GetCols(), input->GetDepth(),
        output->GetRows(), output->GetCols(), output->GetDepth(),
        poolHeight, poolWidth);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }
}

void CudaDistributeReverseMaxPool(Tensor3D* distributed, const Tensor3D* input, const Tensor3D* output, size_t poolHeight, size_t poolWidth)
{
    dim3 threadsPerBlock(16, 16, 4);
    dim3 blocksPerGrid(
        (output->GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output->GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (output->GetDepth() + threadsPerBlock.z - 1) / threadsPerBlock.z);
    DistributedReverseMaxPoolKernel << <blocksPerGrid, threadsPerBlock >> > (
        input->GetData(), output->GetData(), distributed->GetData(),
        input->GetRows(), input->GetCols(), input->GetDepth(),
        output->GetRows(), output->GetCols(), output->GetDepth(),
        distributed->GetRows(), distributed->GetCols(), distributed->GetDepth(),
        poolHeight, poolWidth);
}

void CudaNearestUpsample(Tensor3D* output, const Tensor3D* input, size_t upsampleHeight, size_t upsampleWidth)
{
    dim3 threadsPerBlock(16, 16, 4);
    dim3 blocksPerGrid(
        (output->GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output->GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (output->GetDepth() + threadsPerBlock.z - 1) / threadsPerBlock.z);
    NearestUpsampleKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input->GetData(), output->GetData(),
        input->GetRows(), input->GetCols(), input->GetDepth(),
        output->GetRows(), output->GetCols(), output->GetDepth(),
        upsampleHeight, upsampleWidth);
}

void CudaDistributeReverseNearestUpsample(Tensor3D* distributed, const Tensor3D* output, size_t upsampleHeight, size_t upsampleWidth)
{
    dim3 threadsPerBlock(16, 16, 4);
    dim3 blocksPerGrid(
        (distributed->GetCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (distributed->GetRows() + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (distributed->GetDepth() + threadsPerBlock.z - 1) / threadsPerBlock.z);
    DistributeReverseNearestUpsampleKernel<<<blocksPerGrid, threadsPerBlock>>>(
        output->GetData(), distributed->GetData(),
        output->GetRows(), output->GetCols(), output->GetDepth(),
        distributed->GetRows(), distributed->GetCols(), distributed->GetDepth(),
        upsampleHeight, upsampleHeight);
}

void CudaDropOut(Tensor* output, const Tensor* input, float dropoutRate, float retentionProb, Tensor* dropOutMask)
{
    int blockSize = std::min(size_t(256), output->GetSize());
    int numBlocks = (output->GetSize() + blockSize - 1) / blockSize;
    DropOutKernel<<<numBlocks, blockSize>>>(
        input->GetData(), output->GetData(), output->GetSize(), dropoutRate, retentionProb, dropOutMask ? dropOutMask->GetData() : nullptr);
}

Tensor3D CudaCrossEntropyLoss(const Tensor3D* target, const Tensor3D* predictions)
{
    Tensor3D loss = Tensor3D(target->GetRows(), target->GetCols(), target->GetDepth(), 0.0f, true);
    int blockSize = std::min(size_t(256), loss.GetSize());
    int numBlocks = (loss.GetSize() + blockSize - 1) / blockSize;
    CrossEntropyLossKernel<<<numBlocks, blockSize>>>(
        target->GetData(), predictions->GetData(), loss.GetData(), loss.GetSize());
    return loss;
}

namespace_accelerator_end