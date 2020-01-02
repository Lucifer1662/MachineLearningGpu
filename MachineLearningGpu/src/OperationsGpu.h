#pragma once
#include <GL\glew.h>
class Mat;


enum MyEnum : char
{
	Sig = 0,
	SigDer = 1,
	TDot = 2,
	DotT = 3
};


void InitGpuOperations();





template<char op>
void OpRef(Mat& res, const Mat& lhs, const Mat& rhs);
template<char op>
void OpRef(Mat& res, const Mat& lhs);
template<char op>
void OpRef(Mat& res, const Mat& lhs, float rhs);


template<> void OpRef<'*'>(Mat& res, const Mat& lhs, const Mat& rhs);
template<> void OpRef<'+'>(Mat& res, const Mat& lhs, const Mat& rhs);
template<> void OpRef<'-'>(Mat& res, const Mat& lhs, const Mat& rhs);
template<> void OpRef<'x'>(Mat& res, const Mat& lhs, const Mat& rhs);
template<> void OpRef<Sig>(Mat& res, const Mat& lhs);
template<> void OpRef<SigDer>(Mat& res, const Mat& lhs);
template<> void OpRef<TDot>(Mat& res, const Mat& lhs, const Mat& rhs);
template<> void OpRef<DotT>(Mat& res, const Mat& lhs, const Mat& rhs);

Mat DotMat(const Mat& lhs, const Mat& rhs);
void DotMat(Mat& res, const Mat& lhs, const Mat& rhs);
Mat DotMatMe(const Mat& lhs, const Mat& rhs);
Mat DotLhsTransposeMat(const Mat& lhs, const Mat& rhs);
Mat DotRhsTransposeMat(const Mat& lhs, const Mat& rhs);
Mat SigmoidMat(const Mat& mat);
Mat SigmoidDerivativeMat(const Mat& mat);
Mat AddMat(const Mat& lhs, const Mat& rhs);
Mat AddMatMe(const Mat& lhs, const Mat& rhs);
Mat MinusMat(const Mat& lhs, const Mat& rhs);
Mat MinusMatMe(const Mat& lhs, const Mat& rhs);
Mat MultMat(const Mat& lhs, const Mat& rhs);
Mat MultMat(const Mat& lhs, float scalar);
Mat MultMatMe(Mat& res, const Mat& lhs, float scalar);
Mat Transpose(const Mat& mat);
void TransposeMe(Mat& mat);


