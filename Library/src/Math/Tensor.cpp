#include "Tensor.h"
#include <assert.h>
#include <sstream>

#include <vector>
#include <future>

#include "../ThreadPool.h"
#include <MogiAccelerator.h>


namespace_start

Tensor::Tensor() : m_Data(nullptr), m_IsWatcher(false), m_IsOnDevice(false) { }

Tensor::~Tensor()
{
	Dealloc();
}

Tensor::Tensor(size_t size, float value, bool onDevice)
	: m_Data(nullptr), m_IsWatcher(false), m_IsOnDevice(onDevice)
{
	Alloc(size);

	if (m_IsOnDevice)
	{
		accelerator::CudaMemSet(this, size, value);
	}
	else
	{
		for (int i = 0; i < size; i++)
		{
			m_Data[i] = value;
		}
	}
}

Tensor::Tensor(size_t size, std::function<float()> initializer)
	: m_Data(nullptr), m_IsWatcher(false), m_IsOnDevice(false)
{
	Alloc(size);
	for (int i = 0; i < size; i++)
	{
		m_Data[i] = initializer();
	}
}

Tensor::Tensor(const Tensor& other)
	: m_Data(nullptr), m_IsWatcher(false), m_IsOnDevice(other.m_IsOnDevice)
{
	size_t size = other.GetSize();
	Alloc(size);

	if (m_IsOnDevice)
	{
		accelerator::CudaCopyDeviceToDevice(this, &other);
	}
	else {
#ifdef ASYNC
		ThreadPool* pool = ThreadPool::GetInstance();
		int numThreads = std::min(pool->GetNumThreads(), size);
		int dataPerThread = size / numThreads;
		int extradata = size % numThreads;

		std::vector<std::future<void>> threads;
		size_t startIndex = 0;
		for (size_t i = 0; i < numThreads; i++)
		{
			size_t endIndex = startIndex + dataPerThread + (i < extradata ? 1 : 0);

			threads.emplace_back(
				pool->enqueue([startIndex, endIndex, this, &other] {
					for (size_t i = startIndex; i < endIndex; ++i)
					{
						size_t otherTrueIndex = other.TraverseTo(i);
						m_Data[i] = other.GetData()[otherTrueIndex];
					}
					})
			);

			startIndex = endIndex;
		}

		for (auto& thread : threads) {
			thread.get();
		}
#else
		for (int i = 0; i < size; i++)
		{
			size_t otherTrueIndex = other.TraverseTo(i);
			m_Data[i] = other.m_Data[otherTrueIndex];
		}
#endif // ASYNC
	}
}

Tensor* Tensor::operator=(const Tensor& other)
{
	Dealloc();
	m_IsWatcher = false;
	m_IsOnDevice = other.m_IsOnDevice;
	size_t size = other.GetSize();
	Alloc(size);
	
	if (m_IsOnDevice)
	{
		accelerator::CudaCopyDeviceToDevice(this, &other);
	}
	else
	{
		for(int i = 0; i < size; i++)
		{
			size_t otherTrueIndex = other.TraverseTo(i);
			m_Data[i] = other.m_Data[otherTrueIndex];
		}
	}
	return this;
}

Tensor::Tensor(Tensor&& other) noexcept
	: m_Data(nullptr), m_IsWatcher(other.m_IsWatcher), m_IsOnDevice(other.m_IsOnDevice)
{
	m_Data = other.m_Data;
	other.m_Data = nullptr;
}

Tensor::Tensor(float* m_Watching, bool onDevice)
	: m_IsWatcher(true), m_Data((float*)m_Watching), m_IsOnDevice(onDevice)
{
}

Tensor::Tensor(const float* data, size_t size, bool onDevice)
	: m_IsOnDevice(onDevice)
{
	Alloc(size);
	if(m_IsOnDevice)
	{
		accelerator::CudaCopyDeviceToDevice(m_Data, data, size);
	}
	else
	{
		for (int i = 0; i < size; i++)
		{
			m_Data[i] = data[i];
		}
	}
}

void Tensor::ToDevice()
{
	if (m_IsOnDevice)
		return;

	if (IsWatcher())
	{
		throw std::runtime_error("Data ptr is a watcher, copy the data first.");
	}

	float* devicePtr = accelerator::CudaAlloc(GetSize());
	accelerator::CudaCopyHostToDevice(devicePtr, m_Data, GetSize());
	
	Dealloc();
	m_Data = devicePtr;

	m_IsOnDevice = true;
}

void Tensor::ToHost()
{
	if (!m_IsOnDevice)
		return;

	if (IsWatcher())
	{
		throw std::runtime_error("Data ptr is a watcher, copy the data first.");
	}

	float* hostPtr = new float[GetSize()];
	accelerator::CudaCopyDeviceToHost(hostPtr, m_Data, GetSize());

	Dealloc();
	m_Data = hostPtr;

	m_IsOnDevice = false;
}

void AsyncMap(int startIndex, int endIndex, Tensor* t, std::function<float(float v)> mapper)
{
	for (size_t i = startIndex; i < endIndex; i++)
	{
		size_t trueIndex = t->TraverseTo(i);
		t->GetData()[trueIndex] = mapper(t->GetData()[trueIndex]);
	}
}

void Tensor::Map(std::function<float(float v)> mapper)
{
	if (m_IsOnDevice)
	{
		throw std::runtime_error("Kernel code does not support lambda functions!");
	}
	else
	{
#ifdef ASYNC
		ThreadPool* pool = ThreadPool::GetInstance();
		int numThreads = std::min(pool->GetNumThreads(), GetSize());
		int dataPerThread = GetSize() / numThreads;
		int extradata = GetSize() % numThreads;

		std::vector<std::future<void>> threads;
		size_t startIndex = 0;
		for (size_t i = 0; i < numThreads; i++)
		{
			size_t endIndex = startIndex + dataPerThread + (i < extradata ? 1 : 0);

			threads.emplace_back(
				pool->enqueue(AsyncMap, startIndex, endIndex, this, mapper)
			);

			startIndex = endIndex;
		}

		for (auto& thread : threads) {
			thread.get();
		}
#else
		for (int i = 0; i < GetSize(); i++)
		{
			size_t trueIndex = TraverseTo(i);
			m_Data[trueIndex] = mapper(m_Data[trueIndex]);
		}
#endif // ASYNC
	}
}

void AsyncElementWise(int startIndex, int endIndex, Tensor* v1, const Tensor* v2, std::function<float(float v1, float v2)> operation)
{
	for (size_t i = startIndex; i < endIndex; i++)
	{
		size_t trueIndex = v1->TraverseTo(i);
		size_t otherTrueIndex = v2->TraverseTo(i);
		v1->GetData()[trueIndex] = operation(v1->GetData()[trueIndex], v2->GetData()[otherTrueIndex]);
	}
}

void Tensor::ElementWise(const Tensor& other, std::function<float(float v1, float v2)> operation)
{
	assert(GetSize() == other.GetSize() && "Tensor sizes not match!");

	if (m_IsOnDevice)
	{
		throw std::runtime_error("Kernel code does not support lambda functions!");
	}
	else
	{
#ifdef ASYNC
	ThreadPool* pool = ThreadPool::GetInstance();
	int numThreads = std::min(pool->GetNumThreads(), GetSize());
	int dataPerThread = GetSize() / numThreads;
	int extradata = GetSize() % numThreads;

	std::vector<std::future<void>> threads;
	size_t startIndex = 0;
	for (size_t i = 0; i < numThreads; i++)
	{
		size_t endIndex = startIndex + dataPerThread + (i < extradata ? 1 : 0);

		threads.emplace_back(
			pool->enqueue(AsyncElementWise, startIndex, endIndex, this, &other, operation)
		);

		startIndex = endIndex;
	}

	for (auto& thread : threads) {
		thread.get();
	}
#else
	for (int i = 0; i < GetSize(); i++)
	{
		size_t trueIndex = TraverseTo(i);
		size_t otherTrueIndex = other.TraverseTo(i);
		m_Data[trueIndex] = operation(m_Data[trueIndex], other.m_Data[otherTrueIndex]);
	}
#endif // ASYNC
	}
}

float Tensor::GetAt(size_t i) const
{
	if (m_IsOnDevice)
	{
		throw std::runtime_error("Tensor data is on device.");
	}

	assert((i < GetSize() || m_IsWatcher) && "Out of index error!");
	return m_Data[i];
}

void Tensor::SetAt(size_t i, float value)
{
	if (m_IsOnDevice)
	{
		throw std::runtime_error("Tensor data is on device.");
	}

	assert((i < GetSize() || m_IsWatcher) && "Out of index error!");
	m_Data[i] = value;
}

Tensor& Tensor::Add(const Tensor& other) 
{ 
	if (m_IsOnDevice)
	{
		accelerator::CudaAdd(this, &other);
	}
	else
	{
		ElementWise(other, [](float v1, float v2) -> float { return v1 + v2; }); 
	}
	return *this; 
}

Tensor& Tensor::Sub(const Tensor& other) 
{ 
	if (m_IsOnDevice)
	{
		accelerator::CudaSub(this, &other);
	}
	else
	{
		ElementWise(other, [](float v1, float v2) -> float { return v1 - v2; }); 
	}
	return *this; 
}

Tensor& Tensor::Mult(const Tensor& other) 
{ 
	if (m_IsOnDevice)
	{
		accelerator::CudaMult(this, &other);
	}
	else
	{
		ElementWise(other, [](float v1, float v2) -> float { return v1 * v2; }); 
	}
	return *this; 
}

Tensor& Tensor::Div(const Tensor& other) 
{ 
	if(m_IsOnDevice)
	{
		accelerator::CudaDiv(this, &other);
	}
	else
	{
		ElementWise(other, [](float v1, float v2) -> float { return v1 / v2; }); 
	}
	return *this; 
}


Tensor& Tensor::Add(float value)
{
	if (m_IsOnDevice)
	{
		accelerator::CudaAdd(this, value);
	}
	else
	{
		Map([value](float v) -> float { return v + value; });
	}
	return *this;
}

Tensor& Tensor::Sub(float value)
{
	if (m_IsOnDevice)
	{
		accelerator::CudaSub(this, value);
	}
	else
	{
		Map([value](float v) -> float { return v - value; });
	}
	return *this;
}

Tensor& Tensor::Mult(float value)
{
	if (m_IsOnDevice)
	{
		accelerator::CudaMult(this, value);
	}
	else
	{
		Map([value](float v) -> float { return v * value; });
	}
	return *this;
}

Tensor& Tensor::Div(float value)
{
	if (m_IsOnDevice)
	{
		accelerator::CudaDiv(this, value);
	}
	else
	{
		Map([value](float v) -> float { return v / value; });
	}
	return *this;
}

std::string Tensor::ToString() const
{
	if (m_IsOnDevice)
	{
		throw std::runtime_error("Tensor data is on device.");
	}

	std::stringstream ss;
	for(size_t i = 0; i < GetSize(); i++)
	{
		size_t trueIndex = TraverseTo(i);
		ss << (i == 0 ? "" : " ") << m_Data[trueIndex];
	}
	return ss.str();
}

void Tensor::Alloc(size_t size)
{
	Dealloc();
	if (m_IsOnDevice)
	{
		m_Data = accelerator::CudaAlloc(size);
	}
	else
	{
		m_Data = new float[size];
	}
}

void Tensor::Dealloc()
{
	if (m_Data && !m_IsWatcher)
	{
		if (m_IsOnDevice)
		{
			accelerator::CudaDealloc(m_Data);
		}
		else
		{
			delete[] m_Data;
		}
	}
	m_Data = nullptr;
}

namespace_end
