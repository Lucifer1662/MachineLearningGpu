#pragma once
#include <vector>
#include "Mat.h"

class NeuralNetTrainer;
struct PredefNeuralNet;


class NeuralNetBias
{
	friend class NeuralNetTrainer;
	friend struct PredefNeuralNetBias;
	std::vector<Mat> neurons;
	std::vector<Mat> biases;
public:
	NeuralNetBias(std::initializer_list<unsigned int> activations);
	void SetRandom(float min=-1, float max=1);
	void Evaluate(const Mat& input, const Mat& output, PredefNeuralNetBias& predefined);


	template<typename... Args>
	void ExamineResults(int numOfBatches, int batchSize,
		void(*processResults)(Mat& inputs, Mat& outputs, Mat& results),
		void(*GetInOut)(int batchSize, int iteration,
			Mat& inputs, Mat& outputs, Args...), Args... args);
};

struct PredefNeuralNetBias {
	std::vector<Mat> z, a, aError, aDelta, zSigDer, adj;
	PredefNeuralNetBias(int size) :z(size), a(size), aError(size),
		aDelta(size), zSigDer(size), adj(size) {};

	PredefNeuralNetBias(NeuralNetBias& net) : PredefNeuralNetBias(net.neurons.size()) {}

	Mat& getResult() {
		return a.back();
	}
	~PredefNeuralNetBias() {}

};

template<typename... Args>
void NeuralNetBias::ExamineResults(int numOfBatches, int batchSize,
	void(*processResults)(Mat& inputs, Mat& outputs, Mat& results),
	void(*GetInOut)(int batchSize, int iteration,
		Mat& inputs, Mat& outputs, Args...), Args... args) {

	PredefNeuralNetBias predefined(neurons.size());
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

