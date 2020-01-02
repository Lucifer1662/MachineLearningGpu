#include "OperationsGpu.h"
#include "GLHelpers\Program.h"
#include <string>
#include <Mat.h>

#ifndef GGG
int dotMatProgram, dotLhsTranposeMatProgram, dotRhsTranposeMatProgram, sigmoidMatProgram, addMatProgram,
minusMatProgram, multMatProgram, scalarMultMatProgram, scalarMultLocation, sigmoidDerivativeMatProgram;
#define GGG
#endif // !GGG



void InitGpuOperations()
{

	std::string src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		uint rowsLhs;
		uint colsLhs;
		float matLhs[ ]; 
	};

	layout( std430, binding=1 ) buffer rhs
	{
		uint rowsRhs;
		uint colsRhs;
		float matRhs[ ]; 
	};

	layout( std430, binding=2 ) buffer outPut
	{
		uint rowsRes;
		uint colsRes;
		float matRes[ ]; 
	};

	void main() {
		float sum = 0;
		for(int i = 0; i < colsLhs; i++){
			sum +=  matLhs[i + gl_GlobalInvocationID.y*colsLhs] * matRhs[i*colsRhs + gl_GlobalInvocationID.x];
		}		
		matRes[gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*colsRes] = sum;
	}            

)V0G0N";
	
	dotMatProgram =  CreateProgram(src);
//--------------------------------------------------------------------------------------------------------

	src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		uint rowsLhs;
		uint colsLhs;
		float matLhs[ ]; 
	};

	layout( std430, binding=1 ) buffer rhs
	{
		uint rowsRhs;
		uint colsRhs;
		float matRhs[ ]; 
	};

	layout( std430, binding=2 ) buffer outPut
	{
		uint rowsRes;
		uint colsRes;
		float matRes[ ]; 
	};
	


	void main() {
		float sum = 0;
		uvec2 pos = gl_GlobalInvocationID.xy;
		for(int i = 0; i < rowsLhs; i++){
			sum += matLhs[i*colsLhs + pos.y] * matRhs[i*colsRhs + pos.x]; 
		}
		matRes[pos.x + pos.y*colsRes] =  sum;
	}            

)V0G0N";

	dotLhsTranposeMatProgram = CreateProgram(src);
	//--------------------------------------------------------------------------------------------------------

	src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		uint rowsLhs;
		uint colsLhs;
		float matLhs[ ]; 
	};

	layout( std430, binding=1 ) buffer rhs
	{
		uint rowsRhs;
		uint colsRhs;
		float matRhs[ ]; 
	};

	layout( std430, binding=2 ) buffer outPut
	{
		uint rowsRes;
		uint colsRes;
		float matRes[ ]; 
	};
	


	void main() {
		float sum = 0;
		uvec2 pos = gl_GlobalInvocationID.xy;
		for(int i = 0; i < colsLhs; i++){
			sum += matLhs[i + pos.y*colsLhs] * matRhs[i + pos.x* colsLhs]; 
		}
		matRes[pos.x + pos.y*colsRes] =  sum;
	}            
)V0G0N";

	dotRhsTranposeMatProgram = CreateProgram(src);
	//--------------------------------------------------------------------------------------------------------

	src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		ivec2 size;
		highp float mat[ ]; 
	};


	layout( std430, binding=1 ) buffer outPut
	{
		ivec2 sizeRes;
		highp float matRes[ ]; 
	};
	


	void main() {
		uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*sizeRes.y;
		float ex = pow(2.71828182846, mat[i]);
		if(isinf(ex))
			matRes[i] = 1;
		else
			matRes[i] = ex / (ex + 1);
	}            

)V0G0N";

	

	sigmoidMatProgram = CreateProgram(src);
	//-----------------------------------------------------------------------------


	src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		ivec2 size;
		float mat[ ]; 
	};


	layout( std430, binding=1 ) buffer outPut
	{
		ivec2 sizeRes;
		float matRes[ ]; 
	};
	


	void main() {
		uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*size.y;

		float ex = pow(2.71828182846, mat[i]);
		float sigx;
		if(isinf(ex))
			sigx = 1;
		else
			sigx = ex / (ex + 1);

		matRes[i] = sigx * (1 - sigx);
	}            

)V0G0N";



	sigmoidDerivativeMatProgram = CreateProgram(src);
	//-----------------------------------------------------------------------------

	src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		ivec2 sizeLhs;
		float matLhs[ ]; 
	};

	layout( std430, binding=1 ) buffer rhs
	{
		ivec2 sizeRhs;
		float matRhs[ ]; 
	};

	layout( std430, binding=2 ) buffer outPut
	{
		ivec2 sizeRes;
		float matRes[ ]; 
	};
	


	void main() {

		uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*sizeLhs.y;
		matRes[i] = matLhs[i] + matRhs[i];
	}            

)V0G0N";

	addMatProgram = CreateProgram(src);
//----------------------------------------------------------------------------------------------

	src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		ivec2 sizeLhs;
		float matLhs[ ]; 
	};

	layout( std430, binding=1 ) buffer rhs
	{
		ivec2 sizeRhs;
		float matRhs[ ]; 
	};

	layout( std430, binding=2 ) buffer outPut
	{
		ivec2 sizeRes;
		float matRes[ ]; 
	};

	void main() {
		uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*sizeLhs.y;
		matRes[i] = matLhs[i] - matRhs[i];
	}            

)V0G0N";

	minusMatProgram = CreateProgram(src);
//-----------------------------------------------------------------------------------

	src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		ivec2 sizeLhs;
		float matLhs[ ]; 
	};

	layout( std430, binding=1 ) buffer rhs
	{
		ivec2 sizeRhs;
		float matRhs[ ]; 
	};

	layout( std430, binding=2 ) buffer outPut
	{
		ivec2 sizeRes;
		float matRes[ ]; 
	};
	
	void main() {
		uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*sizeLhs.y;
		matRes[i] = matLhs[i] * matRhs[i];
	}            

)V0G0N";

	multMatProgram = CreateProgram(src);
//-------------------------------------------------------------------
	src = R"V0G0N(
#version 430
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;
	layout( std430, binding=0 ) buffer lhs
	{
		ivec2 size;
		float matLhs[ ]; 
	};

	layout( std430, binding=1 ) buffer outPut
	{
		ivec2 sizeRes;
		float matRes[ ]; 
	};

	uniform float scalar;
	
	void main() {
		uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*size.x;
		matRes[i] = matLhs[i] * scalar;
	}            

)V0G0N";

	scalarMultMatProgram = CreateProgram(src);
	scalarMultLocation = glGetUniformLocation(scalarMultMatProgram, "scalar");
}



//result must be correctly sized before hand
Mat& MatOperation(int program, Mat& result, const Mat & lhs, const Mat & rhs) {
	lhs.bo.BindBufferBase(0);
	rhs.bo.BindBufferBase(1);
	result.bo.BindBufferBase(2);
	glUseProgram(program);
	glDispatchCompute(result.size.y, result.size.x, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	return result;
}

Mat& MatOperation(int program, Mat& result, const Mat & lhs, const Mat & rhs, int resx, int resy) {
	result.Resize(resx, resy);
	return MatOperation(program, result, lhs, rhs);
}

Mat MatOperation(int program, const Mat & lhs, const Mat & rhs, int resx, int resy) {
	Mat res(resx, resy);
	MatOperation(program, res, lhs, rhs);
	return res;
}


Mat& MatOperation(int program,Mat& res, const Mat& mat) {
	mat.bo.BindBufferBase(0);
	res.bo.BindBufferBase(1);
	glUseProgram(program);
	glDispatchCompute(res.size.y, res.size.x, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	return res;
}

Mat& MatOperation1Var(int program, Mat& res, const Mat& mat, int resx, int resy) {
	res.Resize(resx, resy);
	return MatOperation(program, res, mat);
}

Mat MatOperation(int program, const Mat& mat, int resx, int resy) {
	Mat res(resx, resy);
	MatOperation(program, res, mat);
	return res;
}

Mat& MatOperation(int program, int scalarLocation, Mat& res, const Mat& mat, float scalar) {
	res.Resize(mat.size.x, mat.size.y);
	mat.bo.BindBufferBase(0);
	res.bo.BindBufferBase(1);
	glUseProgram(scalarMultMatProgram);
	glUniform1f(scalarLocation, scalar);
	glDispatchCompute(res.size.x, res.size.y, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	return res;
}

Mat MatOperation(int program,int scalarLocation, const Mat& mat, float scalar) {
	Mat res(mat.size.x, mat.size.y);
	MatOperation(program, scalarLocation, res, mat, scalar);
	return res;
}


Mat DotMat(const Mat & lhs, const Mat & rhs)
{
	return MatOperation(dotMatProgram, lhs, rhs, lhs.size.x, rhs.size.y);
}

void DotMat(Mat & res, const Mat & lhs, const Mat & rhs)
{
	MatOperation(dotMatProgram, res, lhs, rhs);
}

Mat DotMatMe(const Mat& lhs, const Mat & rhs)
{

	Mat res(lhs.size.x, rhs.size.y);
	lhs.bo.BindBufferBase(0);
	rhs.bo.BindBufferBase(1);
	lhs.bo.BindBufferBase(2);
	glUseProgram(dotMatProgram);
	glDispatchCompute(res.size.y, res.size.x, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	return res;
}

Mat DotLhsTransposeMat(const Mat & lhs, const Mat & rhs)
{
	return MatOperation(dotLhsTranposeMatProgram, lhs, rhs, lhs.size.y, rhs.size.y);
}

Mat DotRhsTransposeMat(const Mat & lhs, const Mat & rhs)
{
	return MatOperation(dotRhsTranposeMatProgram, lhs, rhs, lhs.size.x, rhs.size.x);
}

Mat SigmoidMat(const Mat & mat)
{
	return MatOperation(sigmoidMatProgram, mat, mat.size.x, mat.size.y);
}

Mat SigmoidDerivativeMat(const Mat & mat)
{
	return MatOperation(sigmoidDerivativeMatProgram, mat, mat.size.x, mat.size.y);
}

Mat AddMat(const Mat & lhs, const Mat & rhs)
{
	return MatOperation(addMatProgram, lhs, rhs, lhs.size.x, lhs.size.y);
}

Mat AddMatMe(const Mat & lhs, const Mat & rhs)
{
	Mat res(lhs.size.x, rhs.size.y);
	res.SetRandom();
	lhs.bo.BindBufferBase(0);
	rhs.bo.BindBufferBase(1);
	lhs.bo.BindBufferBase(2);
	glUseProgram(addMatProgram);
	glDispatchCompute(res.size.x, res.size.y, 1);
	return lhs;
}

Mat MinusMat(const Mat & lhs, const Mat & rhs)
{
	return MatOperation(minusMatProgram, lhs, rhs, lhs.size.x, lhs.size.y);
}

Mat MultMat(const Mat & lhs, const Mat & rhs)
{
	return MatOperation(multMatProgram, lhs, rhs, lhs.size.x, lhs.size.y);
}

Mat MultMat(const Mat & mat, float scalar)
{
	return MatOperation(scalarMultMatProgram, scalarMultLocation, mat, scalar);
}

Mat MultMatMe(Mat& res, const Mat & lhs, float scalar)
{
	return MatOperation(multMatProgram,scalarMultMatProgram,res, lhs, scalar);
}


template<>
void OpRef<'*'>(Mat& res, const Mat& lhs, const Mat& rhs) {
	MatOperation(dotMatProgram, res, lhs, rhs, lhs.size.x, rhs.size.y);
}

template<> void OpRef<'+'>(Mat& res, const Mat& lhs, const Mat& rhs) {
	MatOperation(addMatProgram, res, lhs, rhs, lhs.size.x, lhs.size.y);
}

template<> void OpRef<'-'>(Mat& res, const Mat& lhs, const Mat& rhs) {
	MatOperation(minusMatProgram, res, lhs, rhs, lhs.size.x, lhs.size.y);
}
template<> void OpRef<'x'>(Mat& res, const Mat& lhs, const Mat& rhs) {
	MatOperation(multMatProgram, res, lhs, rhs, lhs.size.x, lhs.size.y);
}
template<> void OpRef<'x'>(Mat& res, const Mat& lhs, float rhs) {
	MatOperation(scalarMultMatProgram,scalarMultLocation, res, lhs, rhs);
}
template<> void OpRef<Sig>(Mat& res, const Mat& lhs) {
	MatOperation1Var(sigmoidMatProgram, res, lhs, lhs.size.x, lhs.size.y);
}
template<> void OpRef<SigDer>(Mat& res, const Mat& lhs) {
	MatOperation1Var(sigmoidDerivativeMatProgram, res, lhs, lhs.size.x, lhs.size.y);
}
template<> void OpRef<TDot>(Mat& res, const Mat& lhs, const Mat& rhs) {
	MatOperation(dotLhsTranposeMatProgram, res, lhs,rhs, lhs.size.y, rhs.size.y);
}
template<> void OpRef<DotT>(Mat& res, const Mat& lhs, const Mat& rhs) {
	MatOperation(dotRhsTranposeMatProgram, res, lhs, rhs, lhs.size.x, rhs.size.x);
}
