#pragma once
#include <GL\glew.h>
#include <glm\glm.hpp>
#include <GLHelpers\Buffer.h>
#include <GLHelpers\Program.h>
#include <vector>
#include <iostream>
#include <OperationsGpu.h>

typedef glm::tvec2<int, glm::precision::mediump> ivec2;


class Mat
{
	
	void InitBlank();
	void SetData(const std::vector<float>& data);
	void SetSize();
public:
	size_t Size() const;
	ivec2 size;
	Buffer<GL_SHADER_STORAGE_BUFFER> bo;
	Mat();
	Mat(unsigned rows, unsigned cols);
	Mat(const Mat& mat);
	Mat(Mat&& mat);
	
	//resises does not copy data
	void Resize(int rows, int cols);
	void ResizeCopy(int rows, int cols);

	~Mat();

	void SetRandom(float min = -1, float max = 1);
	void SetIdentity();
	void SetZero();
	void Set(const std::vector<float>& data);
	Mat operator* (const Mat& rhs);
	void DotMatIntoMe(const Mat& lhs, const Mat& rhs);
	Mat operator *=(const Mat& rhs);
	Mat& operator *=(float scalar);
	Mat operator* (float scalar);
	Mat operator +(const Mat& rhs);
	Mat operator +=(const Mat& rhs);	
	Mat operator -(const Mat& rhs);
	Mat operator -=(const Mat& rhs);
	Mat& operator=(Mat&& rhs);
	const Mat& operator=(const Mat& rhs);
	Mat Mult(const Mat& rhs);
	Mat DotLhsTranspose(const Mat&);
	Mat DotRhsTranspose(const Mat&);
	Mat DotBothTranspose(const Mat&);
	


	template<char op>
	void IntoMeOp(const Mat& lhs, const Mat& rhs) {
		OpRef<op>(*this, lhs, rhs);
	}
	template<char op>
	void IntoMeOp(const Mat& lhs) {
		OpRef<op>(*this, lhs);
	}
	template<char op>
	void IntoMeOp(const Mat& lhs, float rhs) {
		OpRef<op>(*this, lhs, rhs);
	}

	std::vector<float> GetData() const;
	
};




std::ostream& operator <<(std::ostream& io, const Mat& mat);

#define emplace(mat, lhs, op, ...) mat.IntoMeOp<op>(lhs,__VA_ARGS__)
	
