#include "stdafx.h"
#include "KNearestNeighbor.h"

namespace mll
{
	KNN::KNN()
	{
		// Set an object
		setObject();
	}

	KNN::KNN(const int K, const measure& meas)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(K, meas);
	}

	KNN::KNN(const mlldataset& dataset, const int K, const measure& meas)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset, K, meas);
	}

	KNN::KNN(const KNN& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	KNN::~KNN()
	{
		// Clear the object
		clearObject();
	}

	KNN& KNN::operator=(const KNN& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	inline void KNN::setObject()
	{
		// Set the parameters
		setType(*this);
		K = 1;
		meas = nullptr;
	}

	inline void KNN::copyObject(const object& obj)
	{
		// Do down casting
		KNN* _obj = (KNN*)&obj;

		// Copy the parameters
		K = _obj->K;

		// Copy the memories
		copyMeasure(*_obj->meas);
		minVec = _obj->minVec;
		maxVec = _obj->maxVec;
		M = _obj->M;
		X = _obj->X;
		T = _obj->T;
		C = _obj->C;
	}

	inline void KNN::clearObject()
	{
		// Clear the memories
		if (meas != nullptr)
		{
			delete meas;
		}
		minVec.release();
		maxVec.release();
		M.release();
		X.release();
		T.release();
		C.release();
	}

	void KNN::condition(const int K, const measure& meas)
	{
		// Set a condition
		this->K = K;
		copyMeasure(meas);
	}

	void KNN::createMeasure(const int type, const bool nflag, const double slope, const double intercept, const double p)
	{
		// Check the old memory
		if (this->meas != nullptr)
		{
			delete this->meas;
		}

		// Create a measurement
		switch (type)
		{
		case KNN_MEASURE_EUCLIDEAN: this->meas = new euclidean(nflag, slope, intercept); break;
		case KNN_MEASURE_MANHATTAN: this->meas = new manhattan(nflag, slope, intercept); break;
		case KNN_MEASURE_CHEBYCHEV: this->meas = new chebychev(nflag, slope, intercept); break;
		case KNN_MEASURE_MINKOWSKI: this->meas = new minkowski(nflag, slope, intercept, p); break;
		case KNN_MEASURE_COSINE: this->meas = new cosine(); break;
		case KNN_MEASURE_CORRELATION: this->meas = new correlation(); break;
		default: this->meas = nullptr; break;
		}
	}

	void KNN::copyMeasure(const measure& meas)
	{
		// Create a measurement using the input variables
		createMeasure(meas.type, meas.nflag, meas.slope, meas.intercept, meas.p);
	}

	void KNN::train(const mlldataset& dataset)
	{
		// Backup the dataset
		X = dataset[0];			// Feature vector
		T = dataset[1];			// Target vector
		C = dataset[2];			// Class vector

		// Calculate min, max vectors
		minVec = algmat::min(X, 0);
		maxVec = algmat::max(X, 0);

		// Check the measurement type
		if (meas->type == KNN_MEASURE_CORRELATION)
		{
			// Calculate a mean vector
			M = algmat::mean(X, 0);
		}
		else
		{
			// Initialize a mean vector
			M = algmat::zeros(msize(1, X.cols));
		}

		// Check the normalization flag
		if (meas->nflag == true)
		{
			convertScale();
		}
	}

	void KNN::train(const mlldataset& dataset, const int K, const measure& meas)
	{
		// Set a train condition
		condition(K, meas);

		// Train the dataset
		train(dataset);
	}

	void KNN::convertScale()
	{
		// Convert ccale on the dataset
		for (int i = 0; i < X.cols; i++)
		{
			for (int j = 0; j < X.rows; j++)
			{
				X(j, i) = (X(j, i) - minVec(i) / (maxVec(i) - minVec(i)));
				X(j, i) = meas->slope * X(j, i)+ meas->intercept;
			}
		}
	}

	const double KNN::predict(const algmat& x)
	{
		// Check the normalization flag
		algmat xp = x;
		if (meas->nflag == true)
		{
			convertScale(xp);
		}

		// Calculate and sort distances
		vector<double> distance;
		vector<double> label;
		for (int i = 0; i < X.rows; i++)
		{
			// Calculate a distance
			double delta = meas->calculate(X.submat(i) - M, xp - M);

			// Sort the distances
			bool maxFlag = true;
			for (int j = 0; j < (int)distance.size(); j++)
			{
				if (distance[j] > delta)
				{
					distance.insert(distance.begin() + j, delta);
					label.insert(label.begin() + j, T(i));
					maxFlag = false;
					break;
				}
			}

			// Check the max flag
			if (maxFlag == true)
			{
				distance.push_back(delta);
				label.push_back(T(i));
			}
		}

		// Vote the class using nearest samples
		ndarray<int, 1> vote(dim(1, C.length()), 0);
		for (int i = 0; i < K; i++)
		{
			for (int j = 0; j < C.length(); j++)
			{
				if (label[i] == C(j))
				{
					vote[j]++;
				}
			}
		}

		// Get an argument of the maxima label
		int maxValue = vote[0];
		int maxIndex = 0;
		for (int i = 1; i < C.length(); i++)
		{
			if (maxValue < vote[i])
			{
				maxValue = vote[i];
				maxIndex = i;
			}
		}

		return C(maxIndex);
	}

	void KNN::convertScale(algmat& x) const
	{
		// Convert scale on the sample data
		for (int i = 0; i < x.length(); i++)
		{
			x(i) = (x(i) - minVec(i)) / (maxVec(i) - minVec(i));
			x(i) = meas->slope * x(i) + meas->intercept;
		}
	}

	const int KNN::open(const string path, const string prefix)
	{
		// Open measurement information
		if (openMeasureInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open label information
		if (openLabelInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open sample information
		if (openSampleInfo(path, prefix) != 0)
		{
			return -1;
		}

		return 0;
	}

	const int KNN::openMeasureInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("KNN_MEASURE_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		bool nflag = false;
		double slope = 0.0;
		double intercept = 0.0;
		double p = 0.0;
		int type = KNN_MEASURE_UNKNOWN;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Normalization")
			{
				if (atoi(splitStrs[1].c_str()) == 1)
				{
					nflag = true;
				}
				else
				{
					nflag = false;
				}
				continue;
			}
			if (splitStrs[0] == "Slope")
			{
				slope = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Intercept")
			{
				intercept = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "p")
			{
				p = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Measure_Type")
			{
				type = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "K")
			{
				K = atoi(splitStrs[1].c_str());
				continue;
			}
		}
		reader.close();

		// Create a measurement for distance calculation
		createMeasure(type, nflag, slope, intercept, p);

		return 0;
	}

	const int KNN::openLabelInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("KNN_LABEL_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Num_C")
			{
				C = algmat(msize(1, atoi(splitStrs[1].c_str())), 0.0);
				while (!reader.eof())
				{
					// Check the end of the section
					getline(reader, lineStr);
					trimStr = trim(lineStr);
					if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
					{
						break;
					}

					// Check a string format
					splitStrs = split(trimStr, "=");
					if (splitStrs.size() == 1)
					{
						continue;
					}

					// Check the key format
					vector<string> indexStrs = split(splitStrs[0], "_");
					if (indexStrs.size() != 2)
					{
						continue;
					}

					// Set a value
					C(atoi(indexStrs[1].c_str())) = atof(splitStrs[1].c_str());
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int KNN::openSampleInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("KNN_SAMPLE_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		int rows = 0;
		int cols = 0;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Vector_Rows")
			{
				rows = atoi(splitStrs[1].c_str());
				T = algmat(msize(rows), 0.0);
				continue;
			}
			if (splitStrs[0] == "Vector_Cols")
			{
				cols = atoi(splitStrs[1].c_str());
				M = algmat(msize(1, cols), 0.0);
				minVec = algmat(msize(1, cols), 0.0);
				maxVec = algmat(msize(1, cols), 0.0);
				X = algmat(msize(rows, cols), 0.0);
				while (!reader.eof())
				{
					// Check the end of the section
					getline(reader, lineStr);
					trimStr = trim(lineStr);
					if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
					{
						break;
					}

					// Check the string format
					splitStrs = split(trimStr, "=");
					if (splitStrs.size() == 1)
					{
						continue;
					}

					// Check the key format
					vector<string> indexStrs = split(splitStrs[0], "_");
					if (indexStrs.size() == 3)
					{
						if (indexStrs[0] == "Mean" && indexStrs[1] == "Vec")
						{
							M(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
						else if (indexStrs[0] == "Min" && indexStrs[1] == "Vec")
						{
							minVec(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
						else if (indexStrs[0] == "Max" && indexStrs[1] == "Vec")
						{
							maxVec(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
						else if (indexStrs[0] == "Label" && indexStrs[1] == "Vec")
						{
							T(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
					}
					else if (indexStrs.size() == 4)
					{
						if (indexStrs[0] == "Sample" && indexStrs[1] == "Vec")
						{
							X(atoi(indexStrs[2].c_str()), atoi(indexStrs[3].c_str())) = atof(splitStrs[1].c_str());
						}
					}
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int KNN::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save measurement information
		writer << getSectionName("KNN_MEASURE_INFO", prefix) << endl;
		if (meas->type != KNN_MEASURE_COSINE && meas->type != KNN_MEASURE_CORRELATION)
		{
			if (meas->nflag == true)
			{
				writer << "Normalization=1" << endl;
			}
			else
			{
				writer << "Normalization=0" << endl;
			}
			writer << "Slope=" << meas->slope << endl;
			writer << "Intercept=" << meas->intercept << endl;
			if (meas->type == KNN_MEASURE_MINKOWSKI)
			{
				writer << "p=" << meas->p << endl;
			}
		}
		writer << "Measure_Type=" << (int)meas->type << endl;
		writer << "K=" << K << endl;
		writer << endl;

		// Save label information
		writer << getSectionName("KNN_LABEL_INFO", prefix) << endl;
		writer << "Num_C=" << C.length() << endl;
		for (int i = 0; i < C.length(); i++)
		{
			writer << "C_" << i << "=" << C(i) << endl;
		}
		writer << endl;

		// Save sample information
		writer << getSectionName("KNN_SAMPLE_INFO", prefix) << endl;
		writer << "Vector_Rows=" << X.rows << endl;
		writer << "Vector_Cols=" << X.cols << endl;
		for (int i = 0; i < X.cols; i++)
		{
			writer << "Mean_Vec_" << i << "=" << M(i) << endl;
		}
		for (int i = 0; i < X.cols; i++)
		{
			writer << "Min_Vec_" << i << "=" << minVec(i) << endl;
		}
		for (int i = 0; i < X.cols; i++)
		{
			writer << "Max_Vec_" << i << "=" << maxVec(i) << endl;
		}
		for (int i = 0; i < X.rows; i++)
		{
			for (int j = 0; j < X.cols; j++)
			{
				writer << "Sample_Vec_" << i << "_" << j << "=" << X(i, j) << endl;
			}
		}
		for (int i = 0; i < X.rows; i++)
		{
			writer << "Label_Vec_" << i << "=" << T(i) << endl;
		}
		writer << endl;
		writer.close();

		return 0;
	}
}