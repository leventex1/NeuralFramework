#pragma once

#include "Core.h"

#include "src/ThreadPool.h"

#include "src/Math/Tensor2D.h"
#include "src/Math/Tensor3D.h"
#include "src/Math/Operation.h"

#include "src/NeuralNetwork/ActivationF.h"
#include "src/NeuralNetwork/CostF.h"
#include "src/NeuralNetwork/Initializer.h"
#include "src/NeuralNetwork/DenseLayer.h"
#include "src/NeuralNetwork/ConvolutionalLayer.h"
#include "src/NeuralNetwork/ReshapeLayer.h"
#include "src/NeuralNetwork/MaxPoolingLayer.h"
#include "src/NeuralNetwork/NearestUpsamplingLayer.h"
#include "src/NeuralNetwork/SoftmaxLayer.h"
#include "src/NeuralNetwork/DropoutLayer.h"
#include "src/NeuralNetwork/Model.h"

namespace_start

void LIBRARY_API TestLibrary();

namespace_end