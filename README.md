# Thesis Project

## Intorduction
This repository is created for my thesis project where I describe and introduce the convolutional neural network modells for facial recognition. The libraries were wirtten in *c++* and use the *CUDA* platform for optimization. There are 5 project in this repositoriy, 2 of them create the main part of the neural network library (`Library`, `LibraryAccelerator`). 

<br />

1. `Library`: Contains all the necessary classes and components to build neural networks.
    - Tensor classes (*2D, 3D*) with unitily functions and tensor operations.
    - Neural network layers (*Dense,- Convolutional,- Dropout,- MaxPooling,- NearestUpsampling,- Reshape,- Softmax-Layer*).
    - Activation functions (*Sigmoid, ReLU*), easily integrable with other ones.
    - Loss functions (*MeanSquareError, CrossEntropyLoss, BinaryCrossEntropyLoss*), easily integrable with other ones.
    - Weight initializatsion functinos (*HE, Xavier*).
    - Learning optimizers (*SGD, Adam*).
    - Model class that binds all the network parts together.

2. `LibraryAccelerator`: implements some useful tensor operation with high paralelism. Uses *CUDA* platform.

3. `DataSet`: Implements some interfaces for commonly used datasets for loading, shuffling and sampling the data (*XORDataset, MNISTDataset, MNISTAutoEncoderDataset, GeneraFaces, FaceCompare, ...*). Except of the XORDataset, all of them need to be provided with a database source, that can be loaded from disk.

4. `Trainer`: Implements some interfaces for commonly used training sequences (*ClassificationTrainer, AutoEncoderTrainer*).

5. `Project`: Just for testing purposes.

<br />

The `DataSet`, `Trainer`, `Project` projects uses the `OpenCV` (dependency) library for loading images from disk if needed. The `Library` and `LibraryAccelerator` projects compile into a dinamic library (*.dll*), that all the other prjects use. Also the `Library` project depends on the `LibraryAccelerator`.

<br />

## Getting started

A basic presentation of a modell training on the XOR dataset using the library. This sample can be found in the `Trainer`/main.cpp file.

```cpp
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
	&simpleModel,						// Train this model.
	&dataset,						// On this dataset.
	&dataset,						// Test on this dataset.
	CostFunctionFactory(CostFunctionType::MeanSuareError),	// On MeanSquareError.
	false);							// Dont use Cuda.

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
trainer.Train(1000, 1.0f, 0.1f);

testing();
```

For MNIST or other datasets, just replace the modell and the testing and training datasets.

```cpp
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
```

This is how the MNISTDataset class can be used to load the MNIST dataset from disk.

```cpp
mogi::dataset::MNISTDataset testingDataset(path+"t10k-images.idx3-ubyte", path+"t10k-labels.idx1-ubyte");
mogi::dataset::MNISTDataset trainingDataset(path+"train-images.idx3-ubyte", path+"train-labels.idx1-ubyte");
```

The modell can be saved and loaded.

```cpp
modell.Save("myModell.txt");
mogi::Model myModell("myModell.txt");
```

