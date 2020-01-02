#pragma once
#include <vector>
#include "Mat.h"

class NeuralNetTrainer;
struct PredefNeuralNet;


class NeuralNet
{
	friend class NeuralNetTrainer;
	friend struct PredefNeuralNet;
	std::vector<Mat> neurons;
public:
	NeuralNet(std::initializer_list<unsigned int> activations);
	void SetRandom(float min=-1, float max=1);
	void Evaluate(const Mat& input, const Mat& output, PredefNeuralNet& predefined);


	template<typename... Args>
	void ExamineResults(int numOfBatches, int batchSize,
		void(*processResults)(Mat& inputs, Mat& outputs, Mat& results),
		void(*GetInOut)(int batchSize, int iteration,
			Mat& inputs, Mat& outputs, Args...), Args... args);
};

struct PredefNeuralNet {
	std::vector<Mat> z, a, aError, aDelta, zSigDer, adj;
	PredefNeuralNet(int size) :z(size), a(size), aError(size),
		aDelta(size), zSigDer(size), adj(size) {};
	PredefNeuralNet(NeuralNet& net) : PredefNeuralNet(net.neurons.size()) {}

	Mat& getResult() {
		return a.back();
	}
	~PredefNeuralNet() {}

};

template<typename... Args>
void NeuralNet::ExamineResults(int numOfBatches, int batchSize,
	void(*processResults)(Mat& inputs, Mat& outputs, Mat& results),
	void(*GetInOut)(int batchSize, int iteration,
		Mat& inputs, Mat& outputs, Args...), Args... args) {

	PredefNeuralNet predefined(neurons.size());
	Mat inputs, outputs;
	for (size_t i = 0; i < numOfBatches; i++)
	{
		GetInOut(batchSize, i, inputs, outputs, args...);
		Evaluate(inputs, outputs, predefined);
		processResults(inputs, outputs, predefined.getResult());
		std::cout << '\r' << (i + 1) / (float)numOfBatches;
	}
	std::cout << std::endl;
}

