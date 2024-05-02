#pragma once
#include <iostream>

#include <Mogi.h>
#include <MogiDataset.h>
#include "src/AutoencoderTrainer.h"
#include "src/Timer.h"

void GeneralFaceTraining()
{
	mogi::Model model("Models/GeneralFace-Conv-AutoEncoder-01-1-1-2.txt");
	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(128, 128, 1, 5, 5, 32, 2, mogi::RelU(0.001), mogi::He(3 * 3 * 1), false));
	//model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(128, 128, 32, 2, 2));

	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(64, 64, 32, 5, 5, 64, 2, mogi::RelU(0.001), mogi::He(3 * 3 * 64), false));
	//model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(64, 64, 64, 2, 2));

	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(32, 32, 64, 5, 5, 128, 2, mogi::RelU(0.001), mogi::He(3 * 3 * 128), false));
	//model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(32, 32, 128, 2, 2));

	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(16, 16, 128, 3, 3, 128, 1, mogi::RelU(0.001), mogi::He(3 * 3 * 128), false));
	//model.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(16, 16, 128, 2, 2));

	//// Bottle neck

	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(8, 8, 128, 3, 3, 128, 1, mogi::RelU(0.001), mogi::He(3 * 3 * 512), false));
	//model.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(8, 8, 128, 2, 2));

	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(16, 16, 128, 5, 5, 128, 2, mogi::RelU(0.001), mogi::He(3 * 3 * 512), false));
	//model.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(16, 16, 128, 2, 2));

	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(32, 32, 128, 5, 5, 64, 2, mogi::RelU(0.001), mogi::He(3 * 3 * 256), false));
	//model.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(32, 32, 64, 2, 2));

	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(64, 64, 64, 5, 5, 32, 2, mogi::RelU(0.001), mogi::He(3 * 3 * 128), false));
	//model.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(64, 64, 32, 2, 2));

	//model.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(128, 128, 32, 5, 5, 1, 2, mogi::Sigmoid(), mogi::Xavier(3 * 3 * 64, 1), false));

	//model.InitializeOptimizer(mogi::OptimizerFactory(mogi::Adam));
	model.Summarize();


	std::cout << "Loading datasets..." << std::endl;
	mogi::dataset::GeneralFaces testingDataset("Datasets/GeneralFaces", 10000, 100, false);
	mogi::dataset::GeneralFaces trainingDataset("Datasets/GeneralFaces", 6000, 1000, false);
	trainingDataset.Shuffle();
	std::cout << "Datasets loaded!" << std::endl;


	AutoencoderTrainer trainer(&model, &trainingDataset, &testingDataset, CostFunctionFactory(MeanSuareError), true);

	/*{
		Timer t;
		float cost = trainer.Validate();
		std::cout << "Average cost before training: " << cost << std::endl;
		std::cout << "duration: " << t.GetTime() << std::endl;
	}*/

	trainer.Train(1, 0.0001f, 0.0001f);

	model.ToHost();

	std::cout << "Saving model..." << std::endl;
	model.Save("Models/GeneralFace-Conv-AutoEncoder-01-1-1-3.txt");
	std::cout << "Model saved!" << std::endl;
}
