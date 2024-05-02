#include "ThreadPool.h"


namespace_start

ThreadPool* ThreadPool::s_Instance = nullptr;

ThreadPool* ThreadPool::GetInstance()
{
    if (!s_Instance)
    {
        size_t numThreads = std::thread::hardware_concurrency();
        s_Instance = new ThreadPool(numThreads);
    }

    return s_Instance;
}

ThreadPool::ThreadPool(size_t threads)
    : stop(false), m_NumThreads(threads)
{
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this]
            {
                for (;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
    );
}

ThreadPool::~ThreadPool()
{
    // Joins all threads
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers)
        worker.join();
}

namespace_end