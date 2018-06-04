#pragma once
#include <GL\glew.h>
#include <vector>

template<unsigned int BUFFER_TYPE>
class Buffer {
	unsigned int bo;
	unsigned int size;
	unsigned int usage;
	Buffer(const Buffer& buffer) = delete;
public:

	void SetUsage(unsigned int usage) {
		Buffer::usage = usage;
	}

	void CreateBuffer() {
		if (bo == -1) {
			glGenBuffers(1, &bo);
		}
	}

	Buffer(unsigned int usage = GL_DYNAMIC_DRAW, bool createBuffer = true) {
		SetUsage(usage);
		if (createBuffer)
			glGenBuffers(1, &bo);
		else
			bo = -1;
	}

	Buffer(void* data, unsigned int sizeInBytes, unsigned int usage = GL_DYNAMIC_DRAW) : Buffer(usage) {
		SetData(data, sizeInBytes);
	}

	Buffer(Buffer&& o) {
		*this = o;
	}

	Buffer&& CreateCopy() const {
		Buffer<BUFFER_TYPE> buffer;
		//create a temp buffer of type GL_COPY_READ_BUFFER
		buffer.Bind(GL_COPY_READ_BUFFER);
		glBufferData(GL_COPY_READ_BUFFER, size, NULL, GL_STATIC_COPY);
		buffer.size = size;

		//copy data to buffer
		Bind();
		glCopyBufferSubData(BUFFER_TYPE, GL_COPY_READ_BUFFER, 0, 0, size);

		return std::move(buffer);
	}

	

	inline void Bind(unsigned bufferType = BUFFER_TYPE)const {
		glBindBuffer(bufferType, bo);
	}

	inline void BindBufferBase(int i) const {
		Bind();
		glBindBufferBase(BUFFER_TYPE, i, bo);
	}

	void SetData(void* data, unsigned int sizeInBytes) {
		Bind();
		size = sizeInBytes;
		glBufferData(BUFFER_TYPE, sizeInBytes, data, usage);
	}
	void SetSubData(void* data, unsigned int sizeInBytes, unsigned int offset) {
		if (offset + sizeInBytes > size)
			ResizeBuffers(sizeInBytes, offset);
		Bind();		
		glBufferSubData(BUFFER_TYPE, offset, sizeInBytes, data);
		
	}

	int GetData(void* data) const{
		Bind();
		glGetBufferSubData(BUFFER_TYPE, 0, size, data);
		return size;
	}

	int GetSubData(void* data, unsigned offset, unsigned size) const {
		Bind();
		glGetBufferSubData(BUFFER_TYPE, offset, size, data);
		return size;
	}

	Buffer& operator=(const Buffer& buffer) = delete;
	Buffer& operator=(Buffer&& buffer) {
		if(bo != -1)
			glDeleteBuffers(1, &bo);
		bo = buffer.bo;
		size = buffer.size;
		usage = buffer.usage;
		buffer.bo = 0;
		buffer.size = 0;
		return *this;
	}

	

	void ResizeBuffers(unsigned int newSize, unsigned int copyAmount = size) {
		unsigned int vbotemp;
		glGenBuffers(1, &vbotemp);
		Bind();

		if (copyAmount > 0) {
			//create a temp buffer of type GL_COPY_READ_BUFFER
			
			glBindBuffer(GL_COPY_READ_BUFFER, vbotemp);
			glBufferData(GL_COPY_READ_BUFFER, newSize, NULL, GL_STATIC_COPY);

			//copy data from vbo1 to it
			glCopyBufferSubData(BUFFER_TYPE, GL_COPY_READ_BUFFER, 0, 0, copyAmount);
			glBindBuffer(BUFFER_TYPE, vbotemp);
			//glVertexPointer(3, GL_FLOAT, 0, (char *)NULL);

			glDeleteBuffers(1, &bo);
		}
		else {
			glBindBuffer(BUFFER_TYPE, vbotemp);
			glBufferData(BUFFER_TYPE, newSize, NULL, usage);
		}
		bo = vbotemp;
		size = newSize;
		Bind();
	}

	~Buffer() {
		glDeleteBuffers(1, &bo);
	}


};
