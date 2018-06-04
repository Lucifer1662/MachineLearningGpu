#include "Mat.h"
#include <vector>
#include <OperationsGpu.h>
#include <memory>


std::ostream& operator <<(std::ostream& io, const Mat& mat) {
	std::vector<float> data = mat.GetData();
	int i = 0;
	for (size_t x = 0; x < mat.size.x; x++)
	{
		for (size_t y = 0; y < mat.size.y; y++, i++)
		{
			io << data[i] << ", ";
		}
		io << std::endl;
	}
	return io;
}

size_t Mat::Size() const
{
	return size.x*size.y;
}

void Mat::InitBlank()
{
	bo.SetData(NULL, Size() * sizeof(float) + sizeof(size));
}

void Mat::SetData(const std::vector<float>& data)
{
	bo.SetSubData((void*)&data[0], Size() * sizeof(float), sizeof(size));
}

void Mat::SetSize()
{
	bo.SetSubData(&size, sizeof(size), 0);
}

void Mat::Resize(int rows, int cols)
{
	if (rows*cols != Size()) {
		size.x = rows;
		size.y = cols;
		InitBlank();
		SetSize();
	}
}

void Mat::ResizeCopy(int rows, int cols)
{
	if (rows*cols != Size()) {
		bo.ResizeBuffers(rows* cols * sizeof(float) + sizeof(size), Size()* sizeof(float));
		size.x = rows;
		size.y = cols;
		SetSize();
	}
}

Mat::Mat()
{
	//std::cout << "Mat()" << std::endl;
}


Mat::Mat(unsigned x, unsigned y)
{
	//std::cout << "Mat(x,y)" << std::endl;
	size.x = x;
	size.y = y;
	InitBlank();
	SetSize();
}

Mat::Mat(const Mat & mat)
{
	//std::cout << "&Mat(x,y)" << std::endl;
	*this = mat;
}

Mat::Mat(Mat && mat): bo(GL_DYNAMIC_DRAW, false)
{
	//std::cout << "&&Mat(x,y)" << std::endl;
	*this = std::move(mat);
}


Mat Mat::operator*(const Mat & rhs)
{
	return DotMat(*this, rhs);
}

void Mat::DotMatIntoMe(const Mat & lhs, const Mat & rhs)
{
	DotMat(*this, lhs, rhs);
}




Mat Mat::operator*=(const Mat & rhs)
{
	return DotMatMe(*this, rhs);
}

Mat& Mat::operator*=(float scalar)
{
	MultMatMe(*this,*this, scalar);
	return *this;
}

Mat Mat::operator*(float scalar)
{
	return MultMat(*this, scalar);
}

Mat Mat::operator+(const Mat & rhs)
{
	return AddMat(*this, rhs);
}

Mat Mat::operator+=(const Mat & rhs)
{
	return AddMatMe(*this, rhs);
}

Mat Mat::operator-(const Mat & rhs)
{
	return MinusMat(*this, rhs);
}

Mat Mat::operator-=(const Mat & rhs)
{
	return Mat();
}

Mat & Mat::operator=(Mat && rhs)
{
	this->bo = std::move(rhs.bo);
	this->size = rhs.size;
	return *this;
}

const Mat & Mat::operator=(const Mat & rhs)
{
	this->bo = rhs.bo.CreateCopy();
	this->size = rhs.size;
	std::vector<float> data = GetData();
	return *this;
}

Mat Mat::Mult(const Mat & rhs)
{
	return MultMat(*this, rhs);
}

Mat Mat::DotLhsTranspose(const Mat& rhs)
{
	return DotLhsTransposeMat(*this, rhs);
}

Mat Mat::DotRhsTranspose(const Mat& rhs)
{
	return DotRhsTransposeMat(*this, rhs);
}





std::vector<float> Mat::GetData() const
{
	std::vector<float> data(Size());
	bo.GetSubData(&data[0], sizeof(size), Size()*sizeof(float));
	return data;
}

Mat::~Mat()
{
}

void Mat::SetRandom(float min, float max)
{
	std::vector<float> vec = std::vector<float>(Size());
	for (size_t i = 0; i < Size(); i++)
	{
		vec[i] = ((rand() %RAND_MAX)/ (float)RAND_MAX)*(max - min) + min;
	}
	SetData(vec);
}

void Mat::SetIdentity()
{

}

void Mat::SetZero()
{
	bo.SetSubData(NULL, Size() * sizeof(float), sizeof(size));
}

void Mat::Set(const std::vector<float>& data)
{
	SetData(data);
}
//
//template<>
//void Mat::IntoMeOp<'*'>(const Mat& lhs, const Mat& rhs) {
//	OpRef<'*'>(*this, lhs, rhs);
//}
//
//template<>
//void Mat::IntoMeOp<'-'>(const Mat& lhs, const Mat& rhs) {
//	OpRef<'-',
//}
//
//template<>
//void Mat::IntoMeOp<'+'>(const Mat& lhs, const Mat& rhs);
//
//template<>
//void Mat::IntoMeOp<Sig>(const Mat& lhs, const Mat& rhs);
//
//template<>
//void Mat::IntoMeOp<SigDer>(const Mat& lhs, const Mat& rhs);
//
//template<>
//void Mat::IntoMeOp<TDot>(const Mat& lhs, const Mat& rhs);
//
//template<>
//void Mat::IntoMeOp<DotT>(const Mat& lhs, const Mat& rhs);