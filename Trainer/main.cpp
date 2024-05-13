#include <iostream>

#include "ThisPersonDoesNotExistsTraining.h"
#include "GeneralFacesTraining.h"
#include "ImageCompareTraining.h"
#include "FaceRecognitionTraining.h"

#include "src/Timer.h"
#include "src/ClassificationTrainer.h"


int main(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}


	//ThisPersonDoesNotExitsTraining();
	//GeneralFaceTraining();
	//ImageCompareTraining();
	//FaceRecognitionTraining();

	{
		/*
			Create a modell for XOR training.
		*/
		mogi::Model simpleModel;
		simpleModel.AddLayer(std::make_shared<mogi::DenseLayer>(2, 3, mogi::Sigmoid(), mogi::Xavier(2, 3)));
		simpleModel.AddLayer(std::make_shared<mogi::DenseLayer>(3, 1, mogi::Sigmoid(), mogi::Xavier(3, 1)));
		simpleModel.InitializeOptimizer(mogi::OptimizerFactory(mogi::OptimizerType::SGD));
		simpleModel.Summarize();

		/*
			Get an XOR dataset.
		*/
		mogi::dataset::XORDataset dataset;

		/*
			Create a trainer.
		*/
		ClassificationTrainer trainer(
			&simpleModel,											// Train this model.
			&dataset,												// On this dataset.
			&dataset,												// Test on this dataset.
			CostFunctionFactory(CostFunctionType::MeanSuareError),	// On BinaryCrossEntropyLoss
			false);													// Dont use Cuda

		// Create tesing function.
		auto testing = [&simpleModel, &dataset] {
			for (int i = 0; i < dataset.GetEpochSize(); i++)
			{
				mogi::dataset::Sample sample = dataset.GetSample();
				dataset.Next();

				mogi::Tensor3D output = simpleModel.FeedForward(sample.Input);
				std::cout << 
					"Input: (" << sample.Input.ToString() <<
					"), Label: (" << sample.Label.ToString() <<
					") Prediction: (" << output.ToString() << 
					")\n";
			}
		};

		testing();

		/*
			Train 1000 epochs, with a learning rate of 1.0 at the first epoch and
			a learning rate of 0.1 at the end of the training (interpolating between).
		*/
		//trainer.Train(1000, 1.0f, 0.1f);

		testing();
	}

	{
		mogi::Model simpleModel;
		simpleModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(28, 28, 1, 3, 3, 8, 1, mogi::RelU(0.0f), mogi::He(3 * 3)));
		simpleModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(28, 28, 8, 2, 2));
		simpleModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(14, 14, 8, 3, 3, 10, 1, mogi::RelU(0.0f), mogi::He(3 * 3)));
		simpleModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(14, 14, 10, 2, 2));
		simpleModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(7, 7, 10, 3, 3, 12, 1, mogi::RelU(0.0f), mogi::He(3 * 3)));
		simpleModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(7, 7, 12, 7 * 7 * 12, 1, 1));
		simpleModel.AddLayer(std::make_shared<mogi::DenseLayer>(7 * 7 * 12, 10, mogi::Sigmoid(), mogi::Xavier(7 * 7 * 12, 10)));
		simpleModel.InitializeOptimizer(mogi::OptimizerFactory(mogi::OptimizerType::SGD));
		simpleModel.Summarize();

		std::string path = "C:/Dev/Szakdolgozat/Project/Trainer/Datasets/MNIST/";
		mogi::dataset::MNISTDataset testingDataset(path+"t10k-images.idx3-ubyte", path+"t10k-labels.idx1-ubyte");
		mogi::dataset::MNISTDataset trainingDataset(path+"train-images.idx3-ubyte", path+"train-labels.idx1-ubyte");

		ClassificationTrainer trainer(
			&simpleModel,											// Train this model.
			&trainingDataset,										// On this dataset.
			&testingDataset,										// Test on this dataset.
			CostFunctionFactory(CostFunctionType::MeanSuareError),	// On BinaryCrossEntropyLoss
			true);													// Dont use Cuda


		auto testing = [&simpleModel, &testingDataset] {
			for (int i = 0; i < 10; i++)
			{
				mogi::dataset::Sample sample = testingDataset.GetSample();
				testingDataset.Next();

				mogi::Tensor3D output = simpleModel.FeedForward(sample.Input);
				std::cout <<
					"Label: (" << sample.Label.ToString() <<
					") Prediction: (" << output.ToString() <<
					")\n";
			}
		};

		testing();

		trainer.Train(1, 0.01f, 0.01f);
		simpleModel.ToHost();

		testing();
		simpleModel.Save("test.txt");
	}


	return 0;
}