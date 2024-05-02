#include "App.h"
#include <sstream>

#include "Trainers/ClassificationTrainer.h"

#include <Mogi.h>
#include <MogiDataset.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>


void App()
{
	std::string line;
	do
	{
		std::cout << "command: ";
		std::getline(std::cin, line);

		std::stringstream ss(line);
		std::string command;
		ss >> command;
		std::stringstream params(ss.str().substr(command.size()));

		if (command == "new")
		{
			std::string name;
			params >> name;
			if (name.size())
			{
				int numCollectedImages = GeatherFacialImages("Datasets/FacialImages", name);
				std::cout << "Collected " << numCollectedImages << " # images" << std::endl;
			}
			else
			{
				std::cout << "Provide a name for the datasets. \"new test_name\"" << std::endl;
			}
		}
		else if (command == "train")
		{
			std::string modelName, datasetName;
			int numImages;
			if (params >> modelName && params >> datasetName && params >> numImages)
			{
				TrainOneShotFacialRecognizer("Datasets/FacialImages", datasetName, numImages, "Models/" + modelName);
			}
			else
			{
				std::cout << "Provide a model/dataset name and the number of images the dataset holds. \"train model_name.txt dataset_name 400\"" << std::endl;
			}
		}
		else if (command == "test")
		{
			std::string modelName;
			if (params >> modelName)
			{
				TestFacialRecognizer("Models/" + modelName);
			}
			else
			{
				std::cout << "Provide a model name. \"test model_name.txt\"" << std::endl;
			}
		}

	} while (line != "exit");
}


int GeatherFacialImages(const std::string& folderPath, const std::string& name)
{
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Error opening video capture device" << std::endl;
		return -1;
	}

	std::cout << "Press s to capture a training image and press ESC to save and exit." << std::endl;
	

	cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);

	std::vector<cv::Mat> images;

	while (true) {
		cv::Mat frame;
		cv::Mat croppedFrame;
		cap >> frame; // Capture a new frame from the camera

		if (frame.empty()) {
			std::cerr << "Error: captured frame is empty." << std::endl;
			break;
		}

		cv::resize(frame, croppedFrame, cv::Size(512, 512));
		int size = 128 + 64;
		cv::Rect regionOfIntrest(croppedFrame.rows/2 - size/2, croppedFrame.cols/2 - size/2, size, size);
		croppedFrame = croppedFrame(regionOfIntrest);


		cv::cvtColor(croppedFrame, croppedFrame, cv::COLOR_BGR2GRAY);

		int key = cv::waitKey(10) & 0xFF;
		if (key == 's') {
			std::cout << "Captured image: " << images.size() + 1 << std::endl;

			cv::Mat data;
			cv::resize(croppedFrame, data, cv::Size(128, 128));
			images.push_back(data);
		}
		else if (key == 27) {	// ESC
			std::cout << "Saving " << images.size() << " images..." << std::endl;
			for(int i = 0; i < images.size(); i++)
			{
				cv::imwrite(folderPath + "/" + name + "_" + std::to_string(i) + ".jpg", images[i]);
			}

			break;
		}

		cv::imshow("Input", croppedFrame);
	}

	cap.release();
	cv::destroyAllWindows();

	return images.size();
}


void TrainOneShotFacialRecognizer(const std::string& folderPath, const std::string& name, int numImages, const std::string& modelName)
{
	std::cout << "Datasets loading..." << std::endl;
	mogi::dataset::FaceRecognitionDataset trainingDataset(
		folderPath, name, 0, numImages,
		"Datasets/GeneralFacesCropped", "image", 0, numImages
	);
	mogi::dataset::FaceRecognitionDataset testDataset(
		folderPath, name, 0, numImages * 0.2f,
		"Datasets/GeneralFacesCropped", "image", 0, numImages * 0.2f
	);
	std::cout << "Datasets loaded!" << std::endl;

	mogi::Model model;
	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(128, 128, 1, 3, 3, 32, 1, mogi::RelU(), mogi::He(3 * 3 * 2), false));
	model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(128, 128, 32, 2, 2));
	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(64, 64, 32, 3, 3, 64, 1, mogi::RelU(), mogi::He(3 * 3 * 32), false));
	model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(64, 64, 64, 2, 2));
	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(32, 32, 64, 3, 3, 128, 1, mogi::RelU(), mogi::He(3 * 3 * 64), false));
	model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(32, 32, 128, 2, 2));
	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(16, 16, 128, 3, 3, 128, 1, mogi::RelU(), mogi::He(3 * 3 * 128), false));
	model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(16, 16, 128, 2, 2));

	model.AddLayer(std::make_shared<mogi::ReshapeLayer>(8, 8, 128, 8 * 8 * 128, 1, 1));
	model.AddLayer(std::make_shared<mogi::DenseLayer>(8 * 8 * 128, 128, mogi::RelU(), mogi::Xavier(8 * 8 * 128, 128)));
	model.AddLayer(std::make_shared<mogi::DenseLayer>(128, 1, mogi::Sigmoid(), mogi::Xavier(128, 1)));

	model.InitializeOptimizer(mogi::OptimizerFactory(mogi::Adam));
	model.Summarize();

	ClassificationTrainer trainer(&model, &trainingDataset, &testDataset, CostFunctionFactory(CostFunctionType::BinaryCrossEntropyLoss), true);

	trainer.Train(1, 0.0001f, 0.0001f);

	model.ToHost();
	std::cout << "Saving modell..." << std::endl;
	model.Save(modelName);
	std::cout << "Modell saved!" << std::endl;
}


void TestFacialRecognizer(const std::string& modelPath)
{
	std::cout << "Loading modell..." << std::endl;
	mogi::Model model(modelPath);
	std::cout << "Modell loaded!" << std::endl;
	model.Summarize();

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Error opening video capture device" << std::endl;
		return;
	}
	std::string faceCascadePath = "C:/Dev/Szakdolgozat/Project/Dependencies/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceCascadePath)) {
		std::cerr << "Error: Could not load face cascade classifier." << std::endl;
		return;
	}

	std::cout << "Press v to verify with the modell or ESC to exit." << std::endl;

	cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);

	while (true) {
		cv::Mat frame;
		cv::Mat croppedFrame;
		cap >> frame; // Capture a new frame from the camera

		if (frame.empty()) {
			std::cerr << "Error: captured frame is empty." << std::endl;
			break;
		}

		cv::resize(frame, croppedFrame, cv::Size(512, 512));
		int size = 128 + 64;
		cv::Rect regionOfIntrest(croppedFrame.rows / 2 - size / 2, croppedFrame.cols / 2 - size / 2, size, size);
		croppedFrame = croppedFrame(regionOfIntrest);
		cv::cvtColor(croppedFrame, croppedFrame, cv::COLOR_BGR2GRAY);

		/*std::vector<cv::Rect> faces;
		faceCascade.detectMultiScale(croppedFrame, faces);
		if (faces.size())
			croppedFrame = croppedFrame(faces[0]);
		cv::resize(croppedFrame, croppedFrame, cv::Size(128, 128));*/

		int key = cv::waitKey(10) & 0xFF;
		if (key == 'v') {
			cv::Mat data;
			cv::resize(croppedFrame, data, cv::Size(128, 128));

			mogi::Tensor3D input(128, 128, 1);
			for (size_t i = 0; i < input.GetRows(); i++)
			{
				for (size_t j = 0; j < input.GetCols(); j++)
				{
					auto value = data.at<uchar>(i, j);
					input.SetAt(i, j, 0, value / 255.0f);
				}
			}

			mogi::Tensor3D output = model.FeedForward(input);
			if (output.GetAt(0, 0, 0) >= 0.8f)
			{
				std::cout << "Verified: ";
			}
			std::cout << output.ToString() << std::endl;
		}
		else if (key == 27) {	// ESC
			break;
		}

		cv::resize(croppedFrame, croppedFrame, cv::Size(512, 512));

		cv::imshow("Input", croppedFrame);
	}

	cap.release();
	cv::destroyAllWindows();

}