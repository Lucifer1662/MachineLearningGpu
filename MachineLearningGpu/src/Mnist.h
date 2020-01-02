#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

typedef unsigned char uchar;
using std::ifstream;
using std::vector;
using std::string;
using std::ios;
using std::cout;
using std::endl;


int reverseInt (int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};


vector<vector<uchar>> read_mnist_images(const string& full_path, unsigned int& number_of_images, unsigned int& image_size) {


	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		vector<vector<uchar>> _dataset(number_of_images);
		for (int i = 0; i < number_of_images; i++) {
			_dataset[i] = vector<uchar>(image_size);
			file.read((char *)_dataset[i].data(), image_size);
		}
		file.close();
		return _dataset;
	}
	else {
		throw std::runtime_error("Cannot open file `" + full_path + "`!");
	}

}

vector<vector<float>> read_mnist_images_float(const string& full_path, unsigned int& number_of_images, unsigned int& image_size) {
	

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		vector<vector<float>> _dataset(number_of_images);
		vector<uchar> buffer(image_size);
		for (int i = 0; i < number_of_images; i++) {
			file.read((char *)buffer.data(), image_size);

			_dataset[i] = vector<float>(image_size);
			for (size_t j = 0; j < image_size; j++)
				_dataset[i][j] = buffer[j] / 256.0f;
			
		}
		file.close();
		return _dataset;
	}
	else {
		throw std::runtime_error("Cannot open file `" + full_path + "`!");
	}

}

Mat read_mnist_images_Mat(const string& full_path, unsigned int& number_of_images, unsigned int& image_size, unsigned int offset = 0) {
	

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");
		unsigned int tempNum;
		file.read((char *)&tempNum, sizeof(number_of_images)), tempNum = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;
		if (number_of_images == 0)
			number_of_images = tempNum;
		

		file.seekg(offset * sizeof(char) * image_size, ios::cur);
		vector<float> _dataset(image_size*number_of_images);
		float* _datasetPtr = _dataset.data();
		vector<uchar> buffer(image_size);
		for (int i = 0; i < number_of_images; i++) {
			file.read((char *)buffer.data(), image_size);

			uchar* bufPtr = buffer.data();
			for (size_t j = i * image_size, end = j + image_size;
				j < end; j++, bufPtr++, _datasetPtr++) {
				*_datasetPtr = *bufPtr / 256.0f;
				//int l = 0;
			}

		}
		file.close();
		return Mat(number_of_images, image_size, _dataset);
	}
	else {
		throw std::runtime_error("Cannot open file `" + full_path + "`!");
	}

}




vector<uchar> read_mnist_labels(string full_path, unsigned int& number_of_labels) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};


	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049)
			throw std::runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		vector<uchar> _dataset(number_of_labels);
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	}
	else {
		throw std::runtime_error("Unable to open file `" + full_path + "`!");
	}
}


Mat read_mnist_labels_Mat(string full_path, unsigned int& number_of_labels, unsigned int offset = 0) {
	
	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049)
			throw std::runtime_error("Invalid MNIST label file!");

		unsigned int tempNum;
		file.read((char *)&tempNum, sizeof(tempNum)), tempNum = reverseInt(tempNum);
		if (number_of_labels == 0)
			number_of_labels = tempNum;

		vector<uchar> _dataset(number_of_labels);
		file.seekg(offset*sizeof(char), ios::cur);
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		//a single row is different outputs
		//a single column is single output
		
		vector<float> data(number_of_labels * 10);
		
		for (size_t x = 0, i = 0; x < number_of_labels; x++)
		{
			for (size_t y = 0; y < 10; y++, i++)
			{
				data[i] = _dataset[x] == y? 1 : 0;
			}
		}
		return Mat(number_of_labels, 10, data);
	}
	else {
		throw std::runtime_error("Unable to open file `" + full_path + "`!");
	}
}




void PrintImage(vector<vector<uchar>> images, vector<uchar> labels) {
	vector<vector<double>> ar;
	int a = 0;
	int b = 0;
	int c = 0;

	std::cout << std::fixed;

	for (size_t i = 0; i < 1; i++)
	{
		vector<uchar>& d = images[i];
		/*for (size_t y = 0; y < 28; y++)
		{
			for (size_t x = 0; x < 28; x++)
			{
				if ((d[y * 28 + x] / 256.0f) > 0.8f)
					cout << 1;
				else
					cout << " ";

				cout << " ";
				//d++;
			}
			cout << endl;
		}*/
		cout << (int)(labels[i]) << endl << endl;
	}

}

void GetImageSizeAndNum(const string& full_path, unsigned int& number_of_images, unsigned int& image_size) {
	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");
		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;
	}
}


vector<int> GetLabelPredicted(Mat& mat) {
	vector<int> results(mat.size.x);
	std::vector<float> data = mat.GetData();
	int i = 0;
	float max;
	for (size_t x = 0; x < mat.size.x; x++)
	{
		max = 0;
		for (size_t y = 0; y < mat.size.y; y++, i++)
		{
			if (data[i] > max) {
				max = data[i];
				results[x] = y;
			}
		}
	}
	return results;
}

vector<int> GetLabels(Mat& mat) {
	vector<int> results;
	results.reserve(mat.size.x);
	std::vector<float> data = mat.GetData();
	int i = 0;
	for (size_t x = 0, i = 0; x < mat.size.x; x++)
		for (size_t y = 0; y < 10; y++)
			if (data[i++]) {
				results.emplace_back(y);
				i += 9 - y;
				break;
			}
	return results;
}

