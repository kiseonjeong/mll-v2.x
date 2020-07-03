#include "stdafx.h"
#include "Mlldataset.h"

namespace mll
{
	mlldataset::mlldataset() : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();
	}

	mlldataset::mlldataset(const string path, const string separator, const labelpos mode) : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();

		// Open a dataset
		open(path, separator, mode);
	}

	mlldataset::mlldataset(const algmat& X, const algmat& T) : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();

		// Set the dataset
		set(X, T);
	}

	mlldataset::mlldataset(const algmat& X) : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();

		// Set the dataset
		set(X);
	}

	mlldataset::mlldataset(const mlldataset& obj) : X(_X), T(_T), dimension(_dimension), nsample(_nsample), C(_C), nclass(_nclass), slope(_slope), intercept(_intercept)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	mlldataset::~mlldataset()
	{
		// Clear the object
		clearObject();
	}

	mlldataset& mlldataset::operator=(const mlldataset& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	const algmat& mlldataset::operator[](const int idx) const
	{
		// Check the index
		assert(idx >= 0 && idx < 3);

		// Get the vector matrix
		switch (idx)
		{
		case 0: return _X;			// feature vector
		case 1: return _T;			// target vector
		default: return _C;			// class vector
		}
	}

	inline void mlldataset::setObject()
	{
		// Set the parameters
		setType(*this);
		_dimension = -1;
		_nsample = -1;
		_nclass = -1;

		// Clear the memories
		_X.release();
		_T.release();
		_C.release();
	}

	inline void mlldataset::copyObject(const object& obj)
	{
		// Do down casting
		mlldataset* _obj = (mlldataset*)&obj;

		// Copy the parameters
		_dimension = _obj->_dimension;
		_nsample = _obj->_nsample;
		_nclass = _obj->_nclass;

		// Copy the memories
		_X = _obj->_X;
		_T = _obj->_T;
		_C = _obj->_C;
	}

	inline void mlldataset::clearObject()
	{
		// Clear the memories
		_X.release();
		_T.release();
		_C.release();
	}

	const int mlldataset::open(const string path, const string separator, const labelpos mode)
	{
		// Create a dataset reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Read the dataset
		vector<algmat> datasetX;
		vector<algmat> datasetT;
		string lineStr;
		string trimStr;
		vector<string> splitStr;
		bool firstData = true;
		while (!reader.eof())
		{
			// Read and check a string value
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			splitStr = split(trimStr, separator);
			if (trimStr == "" || splitStr.size() == 1)
			{
				continue;
			}

			// Check the label position
			if (mode == LABEL_REAR || mode == LABEL_FRONT)
			{
				// Check a first string value
				if (firstData == true)
				{
					// Initialize a feature dimension
					_dimension = (int)splitStr.size() - 1;
					firstData = false;
				}
				else
				{
					// Check a feature dimension
					assert(splitStr.size() - 1 == _dimension);
					if (splitStr.size() - 1 != _dimension)
					{
						return -1;
					}
				}

				// Check the label position
				algmat sampleX(msize(1, _dimension));
				algmat sampleT(msize(1, 1));
				if (mode == LABEL_REAR)
				{
					for (int i = 0; i < _dimension; i++)
					{
						sampleX(i) = atof(splitStr[i].c_str());
					}
					sampleT(0) = atof(splitStr[_dimension].c_str());
				}
				else
				{
					for (int i = 1; i < 1 + _dimension; i++)
					{
						sampleX(i - 1) = atof(splitStr[i].c_str());
					}
					sampleT(0) = atof(splitStr[0].c_str());
				}

				// Save the dataset
				datasetX.push_back(sampleX);
				datasetT.push_back(sampleT);
			}
			else
			{
				// Check a first string value
				if (firstData == true)
				{
					// Initialize a feature dimension
					_dimension = (int)splitStr.size();
					firstData = false;
				}
				else
				{
					// Check a feature dimension
					assert(splitStr.size() == _dimension);
					if (splitStr.size() != _dimension)
					{
						return -1;
					}
				}

				// Check the label position
				algmat sampleX(msize(1, _dimension));
				for (int i = 0; i < _dimension; i++)
				{
					sampleX(i) = atof(splitStr[i].c_str());
				}

				// Save the dataset
				datasetX.push_back(sampleX);
				datasetT.push_back(sampleX);
			}
		}
		reader.close();

		// Set the dataset
		_X = algmat::append(datasetX);
		_T = algmat::append(datasetT);

		// Check the label position
		if (mode == LABEL_REAR || mode == LABEL_FRONT)
		{
			// Set information
			_dimension = _X.cols;
			_nsample = _X.rows;
			_C = _T.uniq();
			_nclass = _C.length();
		}
		else
		{
			// Set information
			_dimension = _X.cols;
			_nsample = _X.rows;
			_C = algmat::zeros(msize(1));
			_nclass = _C.length();
		}

		return 0;
	}

	void mlldataset::set(const algmat& X, const algmat& T)
	{
		// Check the vector length
		assert(X.rows == T.rows);

		// Set the dataset
		_X = X;
		_T = T;

		// Set information
		_dimension = _X.cols;
		_nsample = _X.rows;
		_C = _T.uniq();
		_nclass = _C.length();
	}

	void mlldataset::set(const algmat& X)
	{
		// Set the dataset
		_X = X;
		_T = X;

		// Set information
		_dimension = _X.cols;
		_nsample = _X.rows;
		_C = algmat::zeros(msize(1));
		_nclass = _C.length();
	}

	const bool mlldataset::empty() const
	{
		// Check the dataset is empty or not
		if (_X.empty() == true || _T.empty() == true)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	void mlldataset::shuffle(const int iter)
	{
		// Check a sample size
		if (_X.rows > 1)
		{
			// Set a iteration for shuffling
			int maxIter = iter;
			if (iter < 0)
			{
				maxIter = _X.rows;
			}

			// Create a random number generator
			random_device rd;
			mt19937 gen(rd());
			uniform_int_distribution<int> dist(0, _X.rows - 1);

			// Shuffle the dataset
			int count = 0;
			while (count < maxIter)
			{
				// Randomly select an index
				int i = dist(gen);
				int j = i;
				while (j == i)
				{
					j = dist(gen);
				}

				// Swap the sample data
				_X.swap(i, j);
				_T.swap(i, j);
				count++;
			}
		}
	}

	void mlldataset::scale()
	{
		// Create the vector memories for the feature scaling
		_slope = algmat::zeros(msize(1, _X.cols));
		_intercept = algmat::zeros(msize(1, _X.cols));

		// Normalize the dataset using the feature scaling
		const double epsilon = 1e-5;
		for (int i = 0; i < _X.cols; i++)
		{
			// Find the min, max values
			double minVal = _X(0, i);
			double maxVal = _X(0, i);
			for (int j = 0; j < _X.rows; j++)
			{
				if (_X(j, i) < minVal)
				{
					minVal = _X(j, i);
				}
				if (_X(j, i) > maxVal)
				{
					maxVal = _X(j, i);
				}
			}

			// Normalize the feature value
			const double denom = max(maxVal - minVal, epsilon);
			for (int j = 0; j < _X.rows; j++)
			{
				_X(j, i) = (_X(j, i) - minVal) / denom;
			}

			// Save the scaling value
			_slope(i) = denom;
			_intercept(i) = minVal;
		}
	}

	void mlldataset::normalize()
	{
		// Create the vector memories for the standard score
		_slope = algmat::zeros(msize(1, _X.cols));
		_intercept = algmat::zeros(msize(1, _X.cols));

		// Normalize the dataset using the standard score
		const double epsilon = 1e-5;
		for (int i = 0; i < _X.cols; i++)
		{
			// Calculate a mean value
			double mean = 0.0;
			for (int j = 0; j < _X.rows; j++)
			{
				mean += _X(j, i);
			}
			mean /= _X.rows;

			// Calculate a variance value
			double var = 0.0;
			for (int j = 0; j < _X.rows; j++)
			{
				var += (_X(j, i) - mean) * (_X(j, i) - mean);
			}
			var /= _X.rows;

			// Normalize the feature value
			const double denom = max(sqrt(var), epsilon);
			for (int j = 0; j < _X.rows; j++)
			{
				_X(j, i) = (_X(j, i) - mean) / denom;
			}

			// Save the scaling value
			_slope(i) = denom;
			_intercept(i) = mean;
		}
	}

	const vector<mlldataset> mlldataset::subdata(const int subsize, const bool shuffling) const
	{
		// Check the dataset is empty or not
		assert(empty() == false);

		// Check a size of the dataset
		assert(subsize > 0);

		// Initialize the parameters
		int nsub = (int)ceil((double)_nsample / subsize);

		// Set sub-dataset length information
		ndarray<int, 1> length(dim(1, nsub));
		for (int i = 0; i < nsub; i++)
		{
			if (_nsample / (subsize * (i + 1)) > 0)
			{
				length[i] = subsize;
			}
			else
			{
				length[i] = _nsample % subsize;

			}
		}

		// Check the shuffling flag
		ndarray<int, 1> index(dim(1, _nsample), -1);
		if (shuffling == true && _nclass > 1)
		{
			// Set sub-dataset index information
			ndarray<bool, 1> flag(dim(1, _nsample));
			flag.set(false);
			int count = 0;
			while (count < _nsample)
			{
				int target = count;
				while (true)
				{
					bool selection = false;
					for (int i = 0; i < _nsample; i++)
					{
						if (flag[i] == false && _T(i) == _C(target % _nclass))
						{
							flag[i] = true;
							index[count] = i;
							selection = true;
							count++;
							break;
						}
					}
					if (selection == false)
					{
						target++;
					}
					else
					{
						break;
					}
				}
			}
		}
		else
		{
			// Set sub-dataset index information
			for (int i = 0; i < _nsample; i++)
			{
				index[i] = i;
			}
		}

		// Generate the sub-dataset
		vector<mlldataset> subset;
		if (_nclass > 1)
		{
			for (int i = 0; i < nsub; i++)
			{
				algmat subX(msize(length[i], _dimension));
				for (int j = 0, l = i * subsize; j < subX.rows; j++, l++)
				{
					for (int k = 0; k < subX.cols; k++)
					{
						subX(j, k) = _X(index[l], k);
					}
				}
				algmat subT(msize(length[i], 1));
				for (int j = 0, l = i * subsize; j < subT.rows; j++, l++)
				{
					for (int k = 0; k < subT.cols; k++)
					{
						subT(j, k) = _T(index[l], k);
					}
				}
				subset.push_back(mlldataset(subX, subT));
			}
		}
		else
		{
			for (int i = 0; i < nsub; i++)
			{
				algmat subX(msize(length[i], _dimension));
				for (int j = 0, l = i * subsize; j < subX.rows; j++, l++)
				{
					for (int k = 0; k < subX.cols; k++)
					{
						subX(j, k) = _X(index[l], k);
					}
				}
				subset.push_back(mlldataset(subX));
			}
		}

		// Check the shuffling flag
		if (shuffling == true)
		{
			for (int i = 0; i < nsub; i++)
			{
				subset[i].shuffle();
			}
		}

		return subset;
	}
}