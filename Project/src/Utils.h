#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

static void CroppImage()
{
	for (int i = 0; i < 400; i++)
	{
		cv::Mat img = cv::imread("Datasets/GeneralFaces256x256/image_" + std::to_string(i) + ".jpg", cv::IMREAD_COLOR);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		int size = 128 + 64;
		cv::Rect regionOfIntrest(img.rows / 2 - size / 2, img.cols / 2 - size / 2, size, size);
		img = img(regionOfIntrest);
		cv::resize(img, img, cv::Size(128, 128));

		cv::imwrite("Datasets/GeneralFacesCropped/image_" + std::to_string(i) + ".jpg", img);
	}
}


static void FindFaces()
{

	std::string faceCascadePath = "C:/Dev/Szakdolgozat/Project/Dependencies/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(faceCascadePath)) {
		std::cerr << "Error: Could not load face cascade classifier." << std::endl;
		return;
	}

	for (int i = 0; i < 643; i++)
	{
		cv::Mat img = cv::imread("Datasets/FacialImages/levente_" + std::to_string(i) + ".jpg", cv::IMREAD_COLOR);

		std::vector<cv::Rect> faces;
		faceCascade.detectMultiScale(img, faces);
		if (faces.size())
			img = img(faces[0]);
		cv::resize(img, img, cv::Size(128, 128));

		cv::imwrite("Datasets/FacialImagesCroppedCenteredAroundFaces/image_" + std::to_string(i) + ".jpg", img);
	}
}