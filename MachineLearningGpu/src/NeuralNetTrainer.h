#pragma once
#include "NeuralNet.h"


class NeuralNetTrainer
{
	union {
		struct { std::vector<Mat> z, a, aError, aDelta, zSigDer, adj; };
		PredefNeuralNet predefined;
	};

	NeuralNet& net;
public:
	NeuralNetTrainer(NeuralNet& net);
	~NeuralNetTrainer() {
		predefined.~PredefNeuralNet();
	
	}
	void Train(Mat& input, Mat& output, int iterations);


	template<typename... Args>
	void TrainBatch(int numOfBatches, int batchSize, int iterations,
		void (*GetInOut)(int batchSize, int iteration,
			Mat& inputs, Mat& outputs, Args...), Args... args){
		Mat inputs, outputs;
		for (size_t i = 0; i < numOfBatches; i++)
		{
			GetInOut(batchSize, i, inputs, outputs, args...);
			Train(inputs, outputs, iterations);
			std::cout << '\r' << (i+1) / (float)numOfBatches;
		}
		std::cout << std::endl;
		predefined.getResult().GetData();
	}
	

};

