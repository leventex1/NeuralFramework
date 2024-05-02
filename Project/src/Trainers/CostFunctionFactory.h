#pragma once
#include <Mogi.h>
#include <assert.h>


enum CostFunctionType
{
	MeanSuareError = 0,
	CrossEntropyLoss,
	BinaryCrossEntropyLoss
};


class CostFunctionFactory
{
public:
	CostFunctionFactory(CostFunctionType type) : m_Type(type) { }

	mogi::CostFunction Build(const mogi::Tensor3D& label) const
	{
		switch (m_Type)	
		{
		case MeanSuareError:			return mogi::MeanSquareError(label);
		case CrossEntropyLoss:			return mogi::CrossEntropyLoss(label);
		case BinaryCrossEntropyLoss:	return mogi::BinaryCrossEntropyLoss(label);
		default:
			break;
		}
		assert(false && "Unknown cost function type!");
		return mogi::CostFunction();
	}

private:
	CostFunctionType m_Type;
};
