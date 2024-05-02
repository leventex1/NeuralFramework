#pragma once
#include <chrono>

class Timer {
public:
	Timer() {
		m_StartTime = std::chrono::system_clock::now();
	}
	void Restart()
	{
		m_StartTime = std::chrono::system_clock::now();
	}
	double GetTime()
	{
		std::chrono::time_point<std::chrono::system_clock> endTime = std::chrono::system_clock::now();
		return std::chrono::duration<double>(endTime - m_StartTime).count();
	}
private:
	std::chrono::time_point<std::chrono::system_clock> m_StartTime;
};
