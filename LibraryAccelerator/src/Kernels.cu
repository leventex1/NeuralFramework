#include "Kernels.h"

#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <math.h>
#include <float.h>


__global__ void CopyKernel(float* tensor, float value, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        tensor[idx] = value;
    }
}

__global__ void CopyKernel(float* tensorDst, const float* tensorSrc, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        tensorDst[idx] = tensorSrc[idx];
    }
}

__global__ void AddKernel(float* tensor, size_t size, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        tensor[idx] = tensor[idx] + value;
    }
}

__global__ void SubKernel(float* tensor, size_t size, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        tensor[idx] = tensor[idx] - value;
    }
}

__global__ void MultKernel(float* tensor, size_t size, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        tensor[idx] = tensor[idx] * value;
    }
}

__global__ void DivKernel(float* tensor, size_t size, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        tensor[idx] = tensor[idx] / value;
    }
}

__global__ void AddKernel(float* tensor, const float* other, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        tensor[idx] = tensor[idx] + other[idx];
    }
}

__global__ void SubKernel(float* tensor, const float* other, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        tensor[idx] = tensor[idx] - other[idx];
    }
}

__global__ void MultKernel(float* tensor, const float* other, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        tensor[idx] = tensor[idx] * other[idx];
    }
}

__global__ void DivKernel(float* tensor, const float* other, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        tensor[idx] = tensor[idx] / other[idx];
    }
}

__global__ void SigmoidKernel(float* tensor, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        tensor[idx] = 1.0f / (1.0f + exp(-tensor[idx]));
    }
}

__global__ void DiffSigmoidKernel(float* tensor, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float sigm = 1.0f / (1.0f + exp(-tensor[idx]));
        tensor[idx] = sigm * (1.0f - sigm);
    }
}

__global__ void RelUKernel(float* tensor, size_t size, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        tensor[idx] = tensor[idx] > 0 ? tensor[idx] : alpha * tensor[idx];
    }
}

__global__ void DiffRelUKernel(float* tensor, size_t size, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        tensor[idx] = tensor[idx] > 0 ? 1.0f : alpha;
    }
}

__global__ void AdamOptimizationKernel(float* params, float* gradients, float* firstMoments, float* secondMoments, size_t size, float b1, float b2, float epsilon, size_t timeStep, float learningRate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        firstMoments[idx] = b1 * firstMoments[idx] + (1.0f - b1) * gradients[idx];
        secondMoments[idx] = b2 * secondMoments[idx] + (1.0f - b2) * gradients[idx] * gradients[idx];

        float correctedFirstMoment = firstMoments[idx] / (1.0f - pow(b1, timeStep));
        float correctedSecondMoment = secondMoments[idx] / (1.0f - pow(b2, timeStep));

        float correctedGradient = learningRate * correctedFirstMoment / (sqrt(correctedSecondMoment) + epsilon);

        params[idx] = params[idx] - correctedGradient;
    }
}

__global__ void CorrectedMomentsKernel(float* tensor, size_t size, float b, float timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        tensor[idx] = tensor[idx] / (1.0f - pow(b, timeStep));
    }
}

__global__ void CorrectedGradientKernel(float* res, float* firstMoments, float* secondMoments, size_t size, float learningRate, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        res[idx] = learningRate * firstMoments[idx] / (sqrt(secondMoments[idx]) + epsilon);
    }
}

__global__ void MatrixMultKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ARows && col < BCols) {
        float sum = 0.0;
        for (int i = 0; i < ACols; ++i) {
            sum += A[row * ACols + i] * B[i * BCols + col];
        }
        C[row * BCols + col] = sum;
    }
}

__global__ void MatrixMultRightTranposeKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BRows)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ARows && col < BRows) {
        float sum = 0.0f;
        for (int e = 0; e < ACols; ++e) {
            sum += A[row * ACols + e] * B[col * ACols + e]; // Accessing B as if it's transposed
        }
        C[row * BRows + col] = sum;
    }
}

__global__ void MatrixMultLeftTranposeKernel(const float* A, const float* B, float* C, int ARows, int ACols, int BCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ACols && col < BCols) { // A is transposed, use ACols for row checks
        float sum = 0.0f;
        for (int e = 0; e < ARows; ++e) { // ARows is used here, reflecting the transposed dimension
            sum += A[e * ACols + row] * B[e * BCols + col]; // Access A as transposed
        }
        C[row * BCols + col] = sum;
    }
}

__global__ void ConvolutionKernel(
    const float* input, const float* kernel, float* output,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t kernelRows, size_t kernelCols, size_t kernelDepth,
    size_t outputRows, size_t outputCols,
    size_t stride, size_t padding)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputCols && y < outputRows)
    {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelRows; ky++)
        {
            for (int kx = 0; kx < kernelCols; kx++)
            {
                for (int d = 0; d < kernelDepth; d++)
                {
                    int posY = y * stride + ky - padding;
                    int posX = x * stride + kx - padding;

                    if (posY >= 0 && posY < inputRows && posX >= 0 && posX < inputCols) {
                        int inputIndex = d * inputRows * inputCols + (posY * inputCols + posX);
                        int kernelIndex = d * kernelRows * kernelCols + (ky * kernelCols + kx);
                        sum += input[inputIndex] * kernel[kernelIndex];
                    }
                }
            }
        }
        int outputIndex = y * outputCols + x;
        output[outputIndex] += sum;
    }
}

__global__ void Convolution3DKernel(
    const float* input, const float* kernel, float* output,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t kernelRows, size_t kernelCols,
    size_t outputRows, size_t outputCols, size_t outputDepth,
    size_t stride, size_t padding)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < outputCols && y < outputRows && d < outputDepth)
    {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelRows; ky++)
        {
            for (int kx = 0; kx < kernelCols; kx++)
            {
                int posY = y * stride + ky - padding;
                int posX = x * stride + kx - padding;

                if (posY >= 0 && posY < inputRows && posX >= 0 && posX < inputCols) {
                    int inputIndex = d * inputRows * inputCols + (posY * inputCols + posX);
                    int kernelIndex = ky * kernelCols + kx;
                    sum += input[inputIndex] * kernel[kernelIndex];
                }
            }
        }
        int outputIndex = d * outputRows * outputCols + (y * outputCols + x);
        output[outputIndex] += sum;
    }
}

__global__ void Convolution3DBackKernelFlipKernel(
    const float* input, const float* kernel, float* output,
    size_t inputRows, size_t inputCols,
    size_t kernelRows, size_t kernelCols, size_t kernelDepth,
    size_t outputRows, size_t outputCols, size_t outputDepth,
    size_t stride, size_t padding)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < outputCols && y < outputRows && d < outputDepth)
    {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelRows; ky++)
        {
            for (int kx = 0; kx < kernelCols; kx++)
            {
                int posY = y * stride + ky - padding;
                int posX = x * stride + kx - padding;

                size_t flippedKy = kernelRows - 1 - ky;
                size_t flippedKx = kernelCols - 1 - kx;

                if (posY >= 0 && posY < inputRows && posX >= 0 && posX < inputCols) {
                    int inputIndex = posY * inputCols + posX;
                    int kernelIndex = d * kernelRows * kernelCols + (ky * kernelCols + kx);
                    sum += input[inputIndex] * kernel[kernelIndex];
                }
            }
        }
        int outputIndex = d * outputRows * outputCols + (y * outputCols + x);
        output[outputIndex] += sum;
    }
}

__global__ void MaxPoolKernel(
    const float* input, float* output,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t outputRows, size_t outputCols, size_t outputDepth,
    size_t poolHeight, size_t poolWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < outputCols && y < outputRows && d < outputDepth) {
        float maxVal = -FLT_MAX;
        for (int py = 0; py < poolHeight; py++)
        {
            for (int px = 0; px < poolWidth; px++)
            {
                int inputY = y * poolHeight + py;
                int inputX = x * poolWidth + px;

                if (inputY < inputRows && inputX < inputCols) {
                    int inputIndex = d * inputRows * inputCols + (inputY * inputCols + inputX);
                    float val = input[inputIndex];
                    if (val > maxVal)
                    {
                        maxVal = val;
                    }
                }
            }
        }
        int outputIndex = d * outputRows * outputCols + (y * outputCols + x);
        output[outputIndex] = maxVal;
    }
}

__global__ void DistributedReverseMaxPoolKernel(
    const float* input, const float* output, float* distributed,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t outputRows, size_t outputCols, size_t outputDepth,
    size_t distributedRows, size_t distributedCols, size_t distributedDepth,
    size_t poolHeight, size_t poolWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < outputCols && y < outputRows && d < outputDepth) {
        float maxVal = -FLT_MAX;
        int poolX = 0, poolY = 0;
        for (int py = 0; py < poolHeight; py++)
        {
            for (int px = 0; px < poolWidth; px++)
            {
                int inputY = y * poolHeight + py;
                int inputX = x * poolWidth + px;

                if (inputY < inputRows && inputX < inputCols) {
                    int inputIndex = d * inputRows * inputCols + (inputY * inputCols + inputX);
                    float val = input[inputIndex];
                    if (val > maxVal)
                    {
                        maxVal = val;
                        poolX = px;
                        poolY = py;
                    }
                }
            }
        }
        int distributeY = y * poolHeight + poolY;
        int distributeX = x * poolWidth + poolX;
        int outputIndex = d * outputRows * outputCols + (y * outputCols + x);
        int distributedIndex = d * distributedRows * distributedCols + (distributeY * distributedCols + distributeX);
        distributed[distributedIndex] = output[outputIndex];
    }
}

__global__ void NearestUpsampleKernel(
    const float* input, float* output,
    size_t inputRows, size_t inputCols, size_t inputDepth,
    size_t outputRows, size_t outputCols, size_t outputDepth,
    size_t upsampleHeight, size_t upsampleWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < outputCols && y < outputRows && d < outputDepth) {
        int inputY = y / upsampleHeight;
        int inputX = x / upsampleWidth;
        int inputD = d;

        int outputIndex = (y * outputCols + x) + d * outputCols * outputRows;
        int inputIndex = (inputY * inputCols + inputX) + d * inputCols * inputRows;

        output[outputIndex] = input[inputIndex];
    }
}

__global__ void DistributeReverseNearestUpsampleKernel(
    const float* output, float* distributed,
    size_t outputRows, size_t outputCols, size_t outputDepth,
    size_t distributedRows, size_t distributedCols, size_t distributedDepth,
    size_t upsampleHeight, size_t upsampleWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < distributedCols && y < distributedRows && d < distributedDepth) {
        float value = 0.0f;
        for (int uy = 0; uy < upsampleHeight; uy++)
        {
            for (int ux = 0; ux < upsampleWidth; ux++)
            {
                int outputY = y * upsampleHeight + uy;
                int outputX = x * upsampleWidth + ux;

                int outputIndex = (outputY * outputCols + outputX) + d * outputRows * outputCols;
                value += output[outputIndex];
            }
        }
        int distributedIndex = (y * distributedCols + x) + d * distributedRows * distributedCols;
        distributed[distributedIndex] = value;
    }
}

__global__ void DropOutKernel(const float* input, float* output, size_t size, float dropoutRate, float retentionProb, float* dropOutMask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(1234, idx, 0, &state);

        float r = curand_uniform(&state);
        if (r > dropoutRate)
        {
            output[idx] = input[idx] / retentionProb;
            if (dropOutMask)
            {
                dropOutMask[idx] = 1.0f;
            }
        }
    }
}

__global__ void RandomizeKernel(float* output, size_t size, float min, float max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(1234, idx, 0, &state);
        float r = curand_uniform(&state);
        output[idx] = min + (max - min) * r;
    }
}

__global__ void CrossEntropyLossKernel(const float* targets, const float* predictions, float* loss, size_t size)
{
    const float epsilon = 0.00000001f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if(predictions[idx] > epsilon)
        {
            loss[idx] = (targets[idx] / predictions[idx]) / (float)size;
        }
        else
        {
            loss[idx] = 0;
        }
    }
}