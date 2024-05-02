#include "Tests.h"
#include <assert.h>
#include <iostream>

#include "Mogi.h"


namespace_start

void TestElementWiseMult()
{

}

void TestMatrixMult()
{
	{
		Tensor2D t1 =
		{
			{ 0.0f, 1.0f, 2.0f }, { 3.0f, 4.0f, 5.0f }, { 6.0f, 7.0f, 8.0f },
			{ -0.0f, -1.0f, -2.0f }, { -3.0f, -4.0f, -5.0f }, { -6.0f, -7.0f, -8.0f }
		};
		Tensor2D t2 =
		{
			{ 0.0f, 1.0f, 2.0f }, { 3.0f, 4.0f, 5.0f }, { 6.0f, 7.0f, 8.0f },
			{ -0.0f, -1.0f, -2.0f }, { -3.0f, -4.0f, -5.0f }, { -6.0f, -7.0f, -8.0f }
		};

		assert(Compare(&t1, &t2));
	}
	std::cout << "+";

	{
		Tensor2D left =
		{
			{ 0.0f, 0.0f, 0.0f },
			{ 0.0f, 0.0f, 0.0f },
		};
		Tensor2D right =
		{
			{ 0.0f },
			{ 0.0f },
			{ 0.0f }
		};
		Tensor2D res = MatrixMult(left, right);
		Tensor2D expected = {
			{0.0f},
			{0.0f}
		};
		assert(Compare(&res, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 2, 3},
			{4, 5, 6}
		};
		Tensor2D identity = {
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1}
		};
		Tensor2D result = MatrixMult(A, identity);
		assert(Compare(&result, &A));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 2},
			{3, 4}
		};
		Tensor2D scalarMatrix = {
			{3, 0},
			{0, 3}
		};
		Tensor2D expected = {
			{3, 6},
			{9, 12}
		};
		Tensor2D result = MatrixMult(A, scalarMatrix);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 2, 3},
			{4, 5, 6}
		};
		Tensor2D B = {
			{7, 8},
			{9, 10},
			{11, 12}
		};
		Tensor2D expected = {
			{58, 64},
			{139, 154}
		};
		Tensor2D result = MatrixMult(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{-1, -2},
			{3, 4}
		};
		Tensor2D B = {
			{5, 6},
			{-7, -8}
		};
		Tensor2D expected = {
			{19, 22},
			{13, 14}
		};
		Tensor2D result = MatrixMult(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, -1},
			{1, -1}
		};
		Tensor2D B = {
			{1, 1},
			{1, 1}
		};
		Tensor2D expected = {
			{0, 0},
			{0, 0}
		};
		Tensor2D result = MatrixMult(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		const int N = 100; 
		Tensor2D A(N, N, 1.0f);
		Tensor2D B(N, N, 1.0f);
		Tensor2D result = MatrixMult(A, B);
		Tensor2D expected(N, N, N);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";



	{
		Tensor2D left =
		{
			{ 0.0f, 0.0f, 0.0f },
			{ 0.0f, 0.0f, 0.0f },
		};
		Tensor2D right =
		{
			{ 0.0f },
			{ 0.0f },
			{ 0.0f }
		};
		left.ToDevice();
		right.ToDevice();
		Tensor2D res = MatrixMult(left, right);
		Tensor2D expected = {
			{0.0f},
			{0.0f}
		};
		expected.ToHost();
		res.ToHost();
		assert(Compare(&res, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 2, 3},
			{4, 5, 6}
		};
		Tensor2D identity = {
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1}
		};
		A.ToDevice();
		identity.ToDevice();
		Tensor2D result = MatrixMult(A, identity);
		result.ToHost();
		A.ToHost();
		assert(Compare(&result, &A));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 2},
			{3, 4}
		};
		Tensor2D scalarMatrix = {
			{3, 0},
			{0, 3}
		};
		A.ToDevice();
		scalarMatrix.ToDevice();
		Tensor2D result = MatrixMult(A, scalarMatrix);
		Tensor2D expected = {
			{3, 6},
			{9, 12}
		};
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 2, 3},
			{4, 5, 6}
		};
		Tensor2D B = {
			{7, 8},
			{9, 10},
			{11, 12}
		};
		A.ToDevice();
		B.ToDevice();
		Tensor2D result = MatrixMult(A, B);
		Tensor2D expected = {
			{58, 64},
			{139, 154}
		};
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{-1, -2},
			{3, 4}
		};
		Tensor2D B = {
			{5, 6},
			{-7, -8}
		};
		A.ToDevice();
		B.ToDevice();
		Tensor2D result = MatrixMult(A, B);
		Tensor2D expected = {
			{19, 22},
			{13, 14}
		};
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, -1},
			{1, -1}
		};
		Tensor2D B = {
			{1, 1},
			{1, 1}
		};
		A.ToDevice();
		B.ToDevice();
		Tensor2D result = MatrixMult(A, B);
		Tensor2D expected = {
			{0, 0},
			{0, 0}
		};
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		const int N = 1000;
		Tensor2D A(N, N, 1.0f);
		Tensor2D B(N, N, 1.0f);
		A.ToDevice();
		B.ToDevice();
		Tensor2D result = MatrixMult(A, B);
		result.ToHost();
		Tensor2D expected(N, N, N);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";
}

void TestMatrixMultRightTranspose()
{
	{
		Tensor2D A = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		Tensor2D B = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		Tensor2D expected = {
			{5, 11, 17},
			{11, 25, 39},
			{17, 39, 61}
		};
		Tensor2D result = MatrixMultRightTranspose(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D left = {
			{0.0f, 0.0f},
			{0.0f, 0.0f}
		};
		Tensor2D rightTranspose = {
			{0.0f, 0.0f},
			{0.0f, 0.0f}
		}; // This would be the transpose of a 2x2 zero matrix
		Tensor2D expected = {
			{0.0f, 0.0f},
			{0.0f, 0.0f}
		};
		Tensor2D result = MatrixMultRightTranspose(left, rightTranspose);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	// Multiplication with identity matrix represented in transposed form
	{
		Tensor2D A = {
			{1, 2, 3},
			{4, 5, 6}
		};
		Tensor2D identityTransposed = {
			{1, 0, 0},
			{0, 1, 0}
		}; // This represents the transpose of a 3x3 identity matrix truncated to 2x3 for multiplication
		Tensor2D expected = {
			{1, 2},
			{4, 5}
		}; // Only the first two columns of A are preserved
		Tensor2D result = MatrixMultRightTranspose(A, identityTransposed);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	// Multiplication resulting in a scalar (dot product)
	{
		Tensor2D A = {
			{1, 2, 3}
		};
		Tensor2D B = {
			{4, 5, 6},
		}; // B is already in a transposed form suitable for dot product
		Tensor2D expected = {
			{32} // 1*4 + 2*5 + 3*6 = 32
		};
		Tensor2D result = MatrixMultRightTranspose(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		const int N = 100;
		Tensor2D A(N, N, 1.0f);
		Tensor2D B(N, N, 1.0f);
		Tensor2D expected(N, N, static_cast<float>(N));
		Tensor2D result = MatrixMultRightTranspose(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 2, 3},
			{4, 5, 6}
		};
		Tensor2D BTransposed = {
			{7, 9, 11},
			{8, 10, 12}
		};
		A.ToDevice();
		BTransposed.ToDevice();
		Tensor2D expected = {
			{58, 64},
			{139, 154}
		};
		Tensor2D result = MatrixMultRightTranspose(A, BTransposed);
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	// Testing with negative values and device transfer
	{
		Tensor2D A = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		Tensor2D B = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		Tensor2D expected = {
			{5, 11, 17},
			{11, 25, 39},
			{17, 39, 61}
		};
		A.ToDevice();
		B.ToDevice();
		Tensor2D result = MatrixMultRightTranspose(A, B);
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	// Testing with one-dimensional vectors (as matrices) and device transfer
	{
		Tensor2D A = {
			{1, 2, 3, 4}
		}; // 1x4 matrix
		Tensor2D BTransposed = {
			{4, 3, 2, 1},
		}; // 4x1 matrix, which is the transpose of a 1x4 matrix
		A.ToDevice();
		BTransposed.ToDevice();
		Tensor2D expected = {
			{20} // Dot product: 1*4 + 2*3 + 3*2 + 4*1
		};
		Tensor2D result = MatrixMultRightTranspose(A, BTransposed);
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	// Edge case: Multiplication by a zero matrix (transposed)
	{
		Tensor2D A = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		Tensor2D BTransposed = {
			{0, 0},
			{0, 0},
			{0, 0}
		}; // This is equivalent to multiplying by a zero matrix of size 2x3
		A.ToDevice();
		BTransposed.ToDevice();
		Tensor2D expected = {
			{0, 0, 0},
			{0, 0, 0},
			{0, 0, 0}
		};
		Tensor2D result = MatrixMultRightTranspose(A, BTransposed);
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		const int N = 1000; 
		Tensor2D A = Tensor2D(N, N, 1.0f); 
		Tensor2D B = Tensor2D(N, N, 1.0f); 
		A.ToDevice();
		B.ToDevice();
		Tensor2D result = MatrixMultRightTranspose(A, B);
		Tensor2D expected = Tensor2D(N, N, N);
		result.ToHost();
		assert(Compare(&result, &expected));
	}
	std::cout << "+";
}

void TestMatrixMultLeftTranspose()
{
	{
		Tensor2D A = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		Tensor2D B = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		Tensor2D expected = {
			{35, 44},
			{44, 56}
		};
		Tensor2D result = MatrixMultLeftTranspose(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{0.0f, 0.0f},
			{0.0f, 0.0f},
			{0.0f, 0.0f}
		};
		Tensor2D B = {
			{0.0f, 0.0f},
			{0.0f, 0.0f},
			{0.0f, 0.0f}
		};
		Tensor2D expected = {
			{0.0f, 0.0f},
			{0.0f, 0.0f}
		};
		Tensor2D result = MatrixMultLeftTranspose(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 0},
			{0, 1},
			{0, 0}
		};
		Tensor2D B = {
			{1, 0},
			{0, 1},
			{0, 0}
		};
		Tensor2D expected = {
			{1, 0},
			{0, 1}
		};
		Tensor2D result = MatrixMultLeftTranspose(A, B);
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		Tensor2D B = {
			{1, 2},
			{3, 4},
			{5, 6}
		};
		A.ToDevice();
		B.ToDevice();
		Tensor2D expected = {
			{35, 44},
			{44, 56}
		};
		Tensor2D result = MatrixMultLeftTranspose(A, B);
		result.ToHost(); // Assuming the operation requires bringing the data back to host for comparison
		assert(Compare(&result, &expected));
	}
	std::cout << "+";

	{
		Tensor2D A = {
			{-1, -2},
			{-3, -4},
			{-5, -6}
		};
		Tensor2D B = {
			{-1, -2},
			{-3, -4},
			{-5, -6}
		};
		A.ToDevice();
		B.ToDevice();
		Tensor2D expected = {
			{35, 44},
			{44, 56}
		}; 
		std::cout << "+";

		{
			const int N = 100; // Smaller N for example
			Tensor2D A(N, N, 1.0f); // Original matrix before transposition
			Tensor2D B(N, N, 1.0f);
			Tensor2D expected(N, N, N);
			A.ToDevice();
			B.ToDevice();
			Tensor2D result = MatrixMultLeftTranspose(A, B);
			result.ToHost();
			assert(Compare(&result, &expected));
		}
		std::cout << "+";
	}
}

namespace_end