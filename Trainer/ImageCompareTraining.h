#pragma once
#include <iostream>

#include <Mogi.h>
#include <MogiDataset.h>
#include "src/ClassificationTrainer.h"


void ImageCompareTraining()
{
	mogi::dataset::FaceCompare trainingDataset(
		"Datasets/Verification/Levente_middle", 0, 200,
		"Datasets/GeneralFaces", 0, 200,
		100
	);

	mogi::dataset::FaceCompare testingDataset(
		"Datasets/Verification/Levente_middle", 200, 30,
		"Datasets/GeneralFaces", 200, 30,
		30
	);

	mogi::Model model;
	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(128, 128, 2, 3, 3, 32, 1, mogi::RelU(), mogi::He(3 * 3 * 2), false));
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

	for (size_t i = 0; i < 10; i++)
	{
		mogi::dataset::Sample testSample = testingDataset.GetSample();
		testingDataset.Next();

		mogi::Tensor3D output = model.FeedForward(testSample.Input);
		std::cout << "Target=" << testSample.Label.ToString() << " Prediction=" << output.ToString() << std::endl;
	}

	ClassificationTrainer trainer(&model, &trainingDataset, &testingDataset, CostFunctionFactory(CostFunctionType::BinaryCrossEntropyLoss), true);


	trainer.Train(5, 0.0001f, 0.0001f);

	model.ToHost();

	std::cout << "Saving model..." << std::endl;
	model.Save("Models/LeventeFaceVerificator_middlerange-01.txt");
	std::cout << "Model saved!" << std::endl;

	for (size_t i = 0; i < 10; i++)
	{
		mogi::dataset::Sample testSample = testingDataset.GetSample();
		testingDataset.Next();

		mogi::Tensor3D output = model.FeedForward(testSample.Input);
		std::cout << "Target=" << testSample.Label.ToString() << " Prediction=" << output.ToString() << std::endl;
	}

}
