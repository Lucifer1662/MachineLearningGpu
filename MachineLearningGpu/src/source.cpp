#define GLEW_STATIC
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>
#include <GLHelpers\Buffer.h>
#include <GLHelpers\Program.h>
//-----------------------------------
#include "Mat.h"
#include "OperationsGpu.h"
#include<iostream>



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

void TrainV1(std::vector<Mat>& neuralNet)
{

	std::vector<float> b(12);
	int i = 0;
	b[i++] = 0;
	b[i++] = 0;
	b[i++] = 1;

	b[i++] = 0;
	b[i++] = 1;
	b[i++] = 1;

	b[i++] = 1;
	b[i++] = 0;
	b[i++] = 1;

	b[i++] = 1;
	b[i++] = 1;
	b[i++] = 1;

	Mat input(4, 3);
	input.Set(b);
	std::cout << input << std::endl;


	Mat output(4, 1);
	output.Set({ 0, 1, 1, 0});
	std::cout << output << std::endl;


	Mat z1;
	Mat a1;
	Mat z2;
	Mat a2;
	Mat a2_error;
	Mat a2_delta;
	Mat a1_error;
	Mat a1_delta;
	Mat z1SigDer;
	Mat z2SigDer;
	Mat adjust1;
	Mat adjust2;

	for (size_t i = 0; i < 1000; i++)
	{
		//l0 is z0

		//z1 = std::move(input * neuralNet[0]);
		emplace(z1, input, '*', neuralNet[0]);

		//a1 = std::move(SigmoidMat(z1));
		emplace(a1, z1, Sig);
		//z2 = std::move(a1 * neuralNet[1]);
		emplace(z2, a1, '*', neuralNet[1]);
		//a2 = std::move(SigmoidMat(z2));
		emplace(a2, z2, Sig);
		


		//activation error
		//a2_error = std::move((output - a2)*2);
		emplace(a2_error, output, '-', a2);
		//a2_error.IntoMeOp<'x'>(a2_error, 2);
		emplace(a2_error, a2_error, 'x', 2);
		//a2_error *= 2;
		

	


		//the change needed to decrease the error
		//Δa2 = (output - y) * sig'(z2)
		//a2_delta = std::move(a2_error.Mult(SigmoidDerivativeMat(z2)));
		emplace(z2SigDer, z2, SigDer);
		emplace(a2_delta, a2_error, 'x', z2SigDer);
	
		//the error in l1
		//ΔC/Δa1 = Δa2 * w
		//ΔC/Δa1 = (output - y) * sig'(z2) * w1
		//but transposing the synaps matrix each training data is summed togeather
		//a1_error = std::move(a2_delta.DotRhsTranspose(neuralNet[1]));
		emplace(a1_error, a2_delta, DotT, neuralNet[1]);
	
		//Δa1 = ΔC/Δa2 * sig'(a1)
		//Δa1 = ΔC/Δa2 * sig'(a1)
		//Δa1 = (output - y) * sig'(z2) * w1 * sig'(z1)
		//a1_delta = std::move(a1_error.Mult(SigmoidDerivativeMat(z1)));
		emplace(z1SigDer, z1, SigDer);
		emplace(a1_delta, a1_error, 'x', z1SigDer);

		//applying the deltas to the synapses
		//w += a1 * Δa2
		//w += (output - y) * sig'(z2) * a1 
		//but transposing the delta is times to all
		//neuralNet[1] += std::move(a1.DotLhsTranspose(a2_delta));
		emplace(adjust2, a1, TDot, a2_delta);
		emplace(neuralNet[1], neuralNet[1], '+', adjust2);
		//w += (output - y) * sig'(z2) * w1 * sig'(z1) * a0
		//neuralNet[0] += std::move(input.DotLhsTranspose(a1_delta));
		emplace(adjust1, input, TDot, a1_delta);
		emplace(neuralNet[0], neuralNet[0], '+', adjust1);
		
	}
	z1 = std::move(input * neuralNet[0]);
	a1 = std::move(SigmoidMat(z1));
	z2 = std::move(a1 * neuralNet[1]);
	a2 = std::move(SigmoidMat(z2));
	std::cout << "result" << std::endl;
	std::cout << a2 << std::endl;
}


int main(int argc, char** argv) {
	
	glfwInit();
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	auto window = glfwCreateWindow(600, 600, "Plot Plus", NULL, NULL);
	glfwMakeContextCurrent(window);
	glewInit();

	InitGpuOperations();

	

	std::vector<Mat> neuralNet(3);
	neuralNet[0] = Mat(3, 3);
	neuralNet[1] = Mat(3, 1);
	neuralNet[0].SetRandom();
	neuralNet[1].SetRandom();
	std::cout << "Neural Net" << std::endl;
	std::cout << neuralNet[0] << std::endl;
	std::cout << neuralNet[1] << std::endl;
	Train(neuralNet);
	
	int p;
	std::cin >> p;
	return 0;
}