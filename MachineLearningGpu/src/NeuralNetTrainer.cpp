#include "NeuralNetTrainer.h"
#include <time.h>

NeuralNetTrainer::NeuralNetTrainer(NeuralNet& net):net(net), predefined(net.neurons.size()) {}

void NeuralNetTrainer::Train(Mat & input, Mat & output, int iterations)
{
	std::vector<Mat>& neurons = net.neurons;

	for (size_t i = 0; i < iterations; i++)
	{

		net.Evaluate(input, output, predefined);


		emplace(aError.back(), output, '-', a.back());
		//emplace(aError.back(), aError.back(), 'x', 2);
		emplace(zSigDer.back(), z.back(), SigDer);
		emplace(aDelta.back(), aError.back(), 'x', zSigDer.back());
		
		
		for (auto zSigDerIt = zSigDer.rbegin()+1,
			zIt = z.rbegin()+1,
			aDeltaIt = aDelta.rbegin(),
			aErrorIt = aError.rbegin()+1,
			neuronsIt = neurons.rbegin()
			; neuronsIt != neurons.rend()-1; 
			zSigDerIt++, zIt++, aDeltaIt++, aErrorIt++, neuronsIt++)
		{
			emplace(*aErrorIt, *aDeltaIt, DotT, *neuronsIt);
			emplace(*zSigDerIt, *zIt, SigDer);
			emplace(*(aDeltaIt+1), *aErrorIt, 'x', *zSigDerIt);
		}


		for (auto adjIt = adj.rbegin(), aIt = a.rbegin() + 1,
			aDeltaIt = aDelta.rbegin(), neuronsIt = neurons.rbegin();
			neuronsIt != neurons.rend() - 1;
			adjIt++, aIt++, aDeltaIt++, neuronsIt++)
		{
			emplace(*adjIt, *aIt, TDot, *aDeltaIt);
			emplace(*neuronsIt, *neuronsIt, '+', *adjIt);
		}

		emplace(adj.front(), input, TDot, aDelta.front());
		emplace(neurons.front(), neurons.front(), '+', adj.front());

	}

}





