#pragma once
#include <memory>
#include <vector>
#include <string>

#include "Core.h"
#include "Layer.h"


namespace_start

struct ModelShape
{
	size_t InputRows, InputCols, InputDepth;
	size_t OutputRows, OutputCols, OutputDepth;
};

class LIBRARY_API Model
{
public:
	Model() { }
	Model(const std::string& filePath);

	void AddLayer(const std::shared_ptr<Layer>& headLayer);
	void AddLayer(const std::string& layerName, const std::string& layerFromData);

	void ToHost();
	void ToDevice();

	void InitializeOptimizer(OptimizerFactory optimizerFactory);
	Tensor3D FeedForward(const Tensor3D& inputs) const;
	Tensor3D BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate, size_t t);

	void Save(const std::string& filePath) const;
	void Load(const std::string& filePath);

	bool IsModelCorrect(int* errorAt=nullptr) const;
	ModelShape GetModelShape() const;
	void Summarize() const;

	const std::shared_ptr<Layer>& GetLayer(size_t i);

	inline const std::shared_ptr<Layer>& GetRootLayer() const { return m_RootLayer; }
private:
	std::shared_ptr<Layer> m_RootLayer = nullptr;
};

namespace_end