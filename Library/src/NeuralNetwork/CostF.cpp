#include "CostF.h"

#include <MogiAccelerator.h>


namespace_start

MeanSquareError::MeanSquareError(const Tensor3D& target)
{
	Cost = [target](const Tensor3D& output) -> float
	{
		assert((output.GetDepth() == target.GetDepth() &&
			output.GetRows() == target.GetRows() &&
			output.GetCols() == target.GetCols()
			) && "Number of parameters in the output and target must be the same!");

		Tensor3D tar = target;
		Tensor3D out = output;
		tar.ToHost();
		out.ToHost();

		float cost = 0.0f;
		for (size_t t = 0; t < tar.GetSize(); t++)
		{
			float diff = tar.GetData()[t] - out.GetData()[t];
			cost += diff * diff;
		}
		return cost / (float)tar.GetSize();
	};

	DiffCost = [target](const Tensor3D& output) -> Tensor3D {
		assert((output.GetDepth() == target.GetDepth() &&
			output.GetRows() == target.GetRows() &&
			output.GetCols() == target.GetCols()
			) && "Number of parameters in the output and target must be the same!");

		Tensor3D res = output;
		res.Sub(target);
		res.Mult(2.0f / (float)target.GetSize());

		return res;
	};
}


CrossEntropyLoss::CrossEntropyLoss(const Tensor3D& target)
{
	Cost = [target](const Tensor3D& output) -> float
	{
		assert((output.GetDepth() == target.GetDepth() &&
			output.GetRows() == target.GetRows() &&
			output.GetCols() == target.GetCols()
			) && "Number of parameters in the output and target must be the same!");

		Tensor3D tar = target;
		Tensor3D out = output;
		tar.ToHost();
		out.ToHost();

		float cost = 0.0f;
		const float ep = 0.0000001f;

		for (size_t t = 0; t < tar.GetSize(); t++)
		{
			float predVal = out.GetData()[t];
			float tarVal = tar.GetData()[t];
			cost += tarVal * log(predVal + ep);
		}

		return -1.0f * cost / tar.GetSize();
	};

	DiffCost = [target](const Tensor3D& output) -> Tensor3D
	{
		assert((output.GetDepth() == target.GetDepth() &&
			output.GetRows() == target.GetRows() &&
			output.GetCols() == target.GetCols()
			) && "Number of parameters in the output and target must be the same!");

		if (output.IsOnDevice() != target.IsOnDevice())
		{
			throw std::runtime_error("Output and target on different devices.");
		}

		if (output.IsOnDevice())
		{
			return accelerator::CudaCrossEntropyLoss(&target, &output);
		}

		Tensor3D res = target;
		res.ElementWise(output, [](float t, float p) -> float {
			if (p > 0.00000001f)
			{
				return -1.0f * t / p;
			}
			else
				return 0.0f;
		});
		res.Div(target.GetSize());
		return res;
	};
}


BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(const Tensor3D& target)
{
	Cost = [target](const Tensor3D& output) -> float
	{
		assert((output.GetDepth() == target.GetDepth() &&
			output.GetRows() == target.GetRows() &&
			output.GetCols() == target.GetCols()
			) && "Number of parameters in the output and target must be the same!");

		Tensor3D tar = target;
		Tensor3D out = output;
		tar.ToHost();
		out.ToHost();

		const float ep = 1e-7f;
		float cost = 0.0f;
		for (size_t t = 0; t < target.GetSize(); t++)
		{
			// TODO: log(0) -> Nan!
			float predVal = out.GetData()[t];
			float tarVal = tar.GetData()[t];
			cost += tarVal * log(predVal + ep) + (1.0f - tarVal) * log(1.0f - predVal + ep);
		}

		return -1.0f * cost / target.GetSize();
	};

	DiffCost = [target](const Tensor3D& output) -> Tensor3D
	{
		assert((output.GetDepth() == target.GetDepth() &&
			output.GetRows() == target.GetRows() &&
			output.GetCols() == target.GetCols()
			) && "Number of parameters in the output and target must be the same!");

		if (output.IsOnDevice() != target.IsOnDevice())
		{
			throw std::runtime_error("Output and target on different devices.");
		}

		const float ep = 1e-7f;

		if (output.IsOnDevice())
		{
			Tensor3D tar = target;
			Tensor3D out = output;
			tar.ToHost();
			out.ToHost();

			Tensor3D res = tar;
			res.ElementWise(out, [&ep](float t, float p) -> float {
				p = std::min(std::max(p, ep), 1.0f - ep);
				return -((t / p) - (1 - t) / (1 - p));
			});
			res.ToDevice();
			return res;
		}

		Tensor3D res = target;
		res.ElementWise(output, [&ep](float t, float p) -> float {
			p = std::min(std::max(p, ep), 1.0f - ep);
			return -((t / p) - (1 - t) / (1 - p));
		});
		return res;
	};
}


namespace_end