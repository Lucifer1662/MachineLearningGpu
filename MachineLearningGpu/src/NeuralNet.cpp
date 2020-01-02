#include "NeuralNet.h"




NeuralNet::NeuralNet(std::initializer_list<unsigned int> activations)
{
	neurons.reserve(activations.size()-1);
	for (auto i = activations.begin(); i != activations.end()-1; i++)
	{
		neurons.emplace_back(*i, *(i + 1));
	}
}

void NeuralNet::SetRandom(float min, float max)
{
	for (auto i = neurons.begin(); i < neurons.end(); i++)
		i->SetRandom(min, max);
}

void NeuralNet::Evaluate(const Mat& input, const Mat& output, PredefNeuralNet& predefined)
{
	emplace(predefined.z[0], input, '*', neurons[0]);
	emplace(predefined.a[0], predefined.z[0], Sig);
	for (auto zIt = predefined.z.begin() + 1, neuronsIt = neurons.begin() + 1,
		aIt = predefined.a.begin() + 1;
		neuronsIt != neurons.end();
		zIt++, neuronsIt++, aIt++)
	{
		emplace(*zIt, *(aIt - 1), '*', *neuronsIt);
		emplace(*aIt, *zIt, Sig);
	}
}
