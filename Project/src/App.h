#pragma once
#include <string>

void App();
int GeatherFacialImages(const std::string& folderPath, const std::string& name);
void TrainOneShotFacialRecognizer(const std::string& folderPath, const std::string& name, int numImages, const std::string& modelName);
void TestFacialRecognizer(const std::string& modelPath);