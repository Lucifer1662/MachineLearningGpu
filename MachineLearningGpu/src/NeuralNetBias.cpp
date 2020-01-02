#include "NeuralNetBias.h"
#include <algorithm>



NeuralNetBias::NeuralNetBias(std::initializer_list<unsigned int> activations)
{
	neurons.reserve(activations.size()-1);
	biases.reserve(activations.size() - 1);
	for (auto i = activations.begin(); i != activations.end()-1; i++)
	{
		neurons.emplace_back(*i, *(i + 1));
		biases.emplace_back(*i, 1);
	}
}

void NeuralNetBias::SetRandom(float min, float max)
{
	for (auto i = neurons.begin(); i < neurons.end(); i++) {
		i->SetRandom(min, max);
	}
	for (auto i = biases.begin(); i < biases.end(); i++) {
		i->SetRandom(min, max);
	}
	
}

void NeuralNetBias::Evaluate(const Mat& input, const Mat& output, PredefNeuralNetBias& predefined)
{
	emplace(predefined.z[0], input, '*', neurons[0]);
	emplace(predefined.z[0], predefined.z[0], '+', biases[0]);
	emplace(predefined.a[0], predefined.z[0], Sig);
	for (auto zIt = predefined.z.begin() + 1, neuronsIt = neurons.begin() + 1,
		aIt = predefined.a.begin() + 1, biasIt = biases.begin()+1;
		neuronsIt != neurons.end();
		zIt++, neuronsIt++, aIt++, biasIt++)
	{
		emplace(*zIt, *(aIt - 1), '*', *neuronsIt);
		emplace(*zIt, *zIt, '+', *biasIt);
		emplace(*aIt, *zIt, Sig);
	}
}
