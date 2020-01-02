#define GLEW_STATIC
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>
#include <GLHelpers\Buffer.h>
#include <GLHelpers\Program.h>
#include <iostream>
//-----------------------------------
#include <time.h>
#include "NeuralNet.h"
#include "NeuralNetTrainer.h"
#include "Mnist.h"



void TestMat() {
	Mat m1(2, 3);
	m1.SetRandom();
	std::cout << m1 << std::endl;
	Mat m2(3, 2);
	m2.SetRandom();
	std::cout << m2 << std::endl;
	Mat m4(2, 3);
	m4.SetRandom();
	std::cout << m4 << std::endl;

	std::cout << "Dot" << std::endl;
	Mat m3 = std::move(m1 * m2);
	std::cout << m3 << std::endl;

	Mat m5(4, 3);
	m5.SetRandom();
	Mat m6(3, 1);
	m6.SetRandom();
	Mat m7(1, 3);
	m7.SetRandom();
	std::cout << m5 << std::endl;
	std::cout << m6 << std::endl;
	std::cout << m7 << std::endl;

	std::cout << "Dot Again" << std::endl;
	m3 = std::move(m5 * m6);
	std::cout << m3 << std::endl;

	std::cout << "Add" << std::endl;
	m3 = std::move(m1 + m4);
	std::cout << m3 << std::endl;

	std::cout << "Minus" << std::endl;
	m3 = std::move(m1 - m4);
	std::cout << m3 << std::endl;

	std::cout << "Mult" << std::endl;
	m3 = std::move(m1.Mult(m4));
	std::cout << m3 << std::endl;

	std::cout << "Lhs Transpose" << std::endl;
	m3 = std::move(m1.DotLhsTranspose(m4));
	std::cout << m3 << std::endl;

	std::cout << "Lhs Again Transpose" << std::endl;
	m3 = std::move(m5.DotLhsTranspose(m7));
	std::cout << m3 << std::endl;

	std::cout << "Rhs Transpose" << std::endl;
	m3 = std::move(m1.DotRhsTranspose(m4));
	std::cout << m3 << std::endl;

	std::cout << "Sigmoid" << std::endl;
	m3 = std::move(SigmoidMat(m1));
	std::cout << m3 << std::endl;

	std::cout << "Sigmoid Derivative" << std::endl;
	m3 = std::move(SigmoidDerivativeMat(m1));
	std::cout << m3 << std::endl;

	std::cout << "Additve" << std::endl;
	m1 += m4;
	std::cout << m1 << std::endl;
}


void GetInputsAndOutputs (int batchSize, int iteration,
	Mat& inputs, Mat& outputs){
	unsigned int numImages = batchSize, imageSize, numLabels = batchSize;
	inputs = read_mnist_images_Mat("train-images.idx3-ubyte", numImages, imageSize, batchSize*iteration);
	outputs = read_mnist_labels_Mat("train-labels.idx1-ubyte", numLabels, batchSize*iteration);
}

int right=0, wrong=0;

std::vector<int> lab(10);
std::vector<int> unlab(10);

void ProcessResults(Mat& inputs, Mat& outputs, Mat& res) {
	std::vector<int> predictions = GetLabelPredicted(res);
	std::vector<int> expected = GetLabels(outputs);

	for (size_t i = 0; i < predictions.size(); i++) {
		if (expected[i] == predictions[i]) {
			right++;
			lab[expected[i]] ++;
		}
		else {
			wrong++;
			unlab[expected[i]] ++;
		}
		//std::cout << expected[i] << ":  :" << predictions[i] << std::endl;
	}
}

int main(int argc, char** argv) {


	glfwInit();
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	auto window = glfwCreateWindow(600, 600, "Plot Plus", NULL, NULL);
	glfwMakeContextCurrent(window);
	glewInit();

	InitGpuOperations();
	unsigned int numImages, imageSize;
	GetImageSizeAndNum("train-images.idx3-ubyte", numImages, imageSize);

	cout.precision(3);
	cout << std::fixed;
	

	NeuralNet net({imageSize,16,16, 10 });
	net.SetRandom(-1, 1);
	

	//10 20 44
	//20 10 56
	//200 1 53
	//40 5 51 
	NeuralNetTrainer trainer(net);
	trainer.TrainBatch(20, 10, 100, GetInputsAndOutputs);
	std::cout << "Done Training" << right << std::endl;
	
	net.ExamineResults(1, numImages, ProcessResults, GetInputsAndOutputs);


	std::cout << "Right:     " << right << std::endl;
	std::cout << "Wrong:     " << wrong << std::endl;
	std::cout << "Precentage:" << right / ((float)right + wrong) * 100 << std::endl;

	std::cout << "Percentage For Each Number:" << std::endl;
	for (size_t i = 0; i < 10; i++)
	{
		std::cout << i << ": " << lab[i] + unlab[i] << "  " << lab[i] / ((float)lab[i] + unlab[i]) * 100 << std::endl;

	}
	
	int p;
	std::cin >> p;
	return 0;
}