#include <iostream>

#include "Tests.h"


namespace_start

void TestLibrary()
{
	std::cout << "Test in debug mode\n";

	std::cout << "MatrixMult: ";
	TestMatrixMult();
	std::cout << std::endl;

	std::cout << "MatrixMultRightTranspose: ";
	TestMatrixMultRightTranspose();
	std::cout << std::endl;

	std::cout << "MatrixMultLeftTranspose: ";
	TestMatrixMultLeftTranspose();
	std::cout << std::endl;
}

namespace_end