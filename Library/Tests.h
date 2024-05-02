#pragma once
#include "Mogi.h"


namespace_start

static bool Compare(const Tensor* left, const Tensor* right, float error = 0.00000001f)
{
	if (left->GetSize() != right->GetSize())
		return false;

	for (size_t i = 0; i < left->GetSize(); i++)
	{
		size_t leftIndex = left->TraverseTo(i);
		size_t rightIndex = right->TraverseTo(i);
		float l = left->GetData()[leftIndex];
		float r = right->GetData()[rightIndex];
		float d = l - r;
		if (d >= error)
			return false;
	}
	return true;
}

void TestElementWiseMult();

void TestMatrixMult();
void TestMatrixMultRightTranspose();
void TestMatrixMultLeftTranspose();

namespace_end