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
	FaceRecognitionTraining();
	return 0;
}