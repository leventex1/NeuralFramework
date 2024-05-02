#pragma once
#include <iostream>

#include <Mogi.h>
#include <MogiDataset.h>
#include "src/AutoencoderTrainer.h"
#include "src/Timer.h"


void ThisPersonDoesNotExitsTraining()
{
	mogi::Model model;
	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(128, 128, 1, 3, 3, 32, 1, mogi::RelU(), mogi::He(3 * 3 * 1), false));
	model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(128, 128, 32, 2, 2));
	
	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(64, 64, 32, 3, 3, 64, 1, mogi::RelU(), mogi::He(3 * 3 * 32), false));
	model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(64, 64, 64, 2, 2));

	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(32, 32, 64, 3, 3, 128, 1, mogi::RelU(), mogi::He(3 * 3 * 64), false));
	model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(32, 32, 128, 2, 2));

	// Bottle neck

	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(16, 16, 128, 3, 3, 128, 1, mogi::RelU(), mogi::He(3 * 3 * 128), false));
	model.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(16, 16, 128, 2, 2));

	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(32, 32, 128, 3, 3, 64, 1, mogi::RelU(), mogi::He(3 * 3 * 128), false));
	model.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(32, 32, 64, 2, 2));

	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(64, 64, 64, 3, 3, 32, 1, mogi::RelU(), mogi::He(3 * 3 * 64), false));
	model.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(64, 64, 32, 2, 2));

	model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(128, 128, 32, 3, 3, 1, 1, mogi::Sigmoid(), mogi::Xavier(3 * 3 * 32, 1), false));

	model.InitializeOptimizer(mogi::OptimizerFactory(mogi::Adam));
	model.Summarize();

	//model.Save("Models/ThisPersonDoesNotExists-Conv-AutoEncoder-01.txt");

	std::cout << "Loading datasets..." << std::endl;
	mogi::dataset::ThisPersonDoesNotExistsAutoEncoderDataset testingDataset("Datasets/ThisPersonDoesNotExists/images", 100);
	mogi::dataset::ThisPersonDoesNotExistsAutoEncoderDataset trainingDataset("Datasets/ThisPersonDoesNotExists/trainImages", 1000);
	std::cout << "Datasets loaded!" << std::endl;
	
	AutoencoderTrainer trainer(&model, &trainingDataset, &testingDataset, CostFunctionFactory(MeanSuareError), true);

	float cost = trainer.Validate();
	std::cout << "Average cost before training: " << cost << std::endl;

	trainer.Train(10, 0.001f, 0.001f);

	model.ToHost();
	model.Save("Models/ThisPersonDoesNotExists-Conv-AutoEncoder-02.txt");

	for (size_t i = 0; i < 2; i++)
	{
		auto sample = testingDataset.GetSample();
		auto output = model.FeedForward(sample.Input);

		mogi::dataset::ThisPersonDoesNotExistsAutoEncoderDataset::Display(mogi::CreateWatcher(sample.Input, 0));
		mogi::dataset::ThisPersonDoesNotExistsAutoEncoderDataset::Display(mogi::CreateWatcher(output, 0));
		std::cout << "==============\n";
		testingDataset.Next();
	}

}