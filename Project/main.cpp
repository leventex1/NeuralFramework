#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

#include <Mogi.h>
#include <MogiDataset.h>

#include "src/App.h"
#include "src/Utils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

class Timer {
public:
	Timer() {
		m_StartTime = std::chrono::system_clock::now();
	}
	double GetTime()
	{
		std::chrono::time_point<std::chrono::system_clock> endTime = std::chrono::system_clock::now();
		return std::chrono::duration<double>(endTime - m_StartTime).count();
	}
	~Timer()
	{
		std::chrono::time_point<std::chrono::system_clock> endTime = std::chrono::system_clock::now();
		double duration = std::chrono::duration<double>(endTime - m_StartTime).count();
		std::cout << "Duration: " << duration * 1000 << "ms" << std::endl;
	}
private:
	std::chrono::time_point<std::chrono::system_clock> m_StartTime;
};

int main(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}

	/*
		Create a modell for XOR training.
	*/
	mogi::Model simpleModel;
	simpleModel.AddLayer(std::make_shared<mogi::DenseLayer>(2, 3, mogi::Sigmoid(), mogi::Xavier(2, 3)));
	simpleModel.AddLayer(std::make_shared<mogi::DenseLayer>(3, 1, mogi::Sigmoid(), mogi::Xavier(3, 1)));

	/*
		Get an XOR dataset.
	*/
	mogi::dataset::XORDataset dataset;

	


	return 0;
}