#include "stdafx.h"
#include "LogisticRegression.h"

namespace mll
{
	logitmodel::logitmodel()
	{
		// Set an object
		setObject();
	}

	logitmodel::logitmodel(const int maxIter, const double E)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(maxIter, E);
	}

	logitmodel::logitmodel(const mlldataset& dataset, const int maxIter, const double E)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(maxIter, E);

		// Train the dataset
		train(dataset);
	}

	logitmodel::logitmodel(const logitmodel& obj)
	{
		// Set an object
		setObject();

		// Clone Object
		*this = obj;
	}

	logitmodel::~logitmodel()
	{
		// Clear the object
		clearObject();
	}

	logitmodel& logitmodel::operator=(const logitmodel& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	void logitmodel::setObject()
	{
		// Set the parameters
		setType(*this);
		maxIter = 0;
		E = 0.0;

		// Set the memories
		W.release();
	}

	void logitmodel::copyObject(const object& obj)
	{
		// Do down casting
		const logitmodel* _obj = static_cast<const logitmodel*>(&obj);

		// Copy the parameters
		maxIter = _obj->maxIter;
		E = _obj->E;

		// Copy the memories
		W = _obj->W;
	}

	void logitmodel::clearObject()
	{
		// Clear the memories
		W.release();
	}

	void logitmodel::condition(const int maxIter, const double E)
	{
		// Set the conditions
		this->maxIter = maxIter;
		this->E = E;
	}

	void logitmodel::train(const mlldataset& dataset)
	{
		// Check the number of the labels
		assert(dataset[2].length() == 2);

		// Check the label numbers
		assert(dataset[2](0) * dataset[2](1) == -1.0);

		// Backup the dataset and initialize a weight matrix
		algmat X = dataset[0];
		algmat T = dataset[1];
		W = algmat::ones(msize(X.cols, 1));

		// Optimize the weight matrix using SGA (Stochastic Gradient Ascent)
		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<int> dist(0, X.rows - 1);
		for (int i = 0; i < maxIter; i++)
		{
			vector<int> sel;
			for (int j = 0; j < X.rows; j++)
			{
				// Randomly select a sample vector
				int k = 0;
				while (true)
				{
					k = dist(gen);
					bool contains = false;
					for (int l = 0; l < (int)sel.size(); l++)
					{
						if (sel[l] == k)
						{
							contains = true;
							break;
						}
					}
					if (contains == false)
					{
						sel.push_back(k);
						break;
					}
				}

				// Update the weight matrix
				double alpha = 4.0 / (1.0 + j + i) + 0.01;
				double H = sigmoid(X.submat(k).dot(W)(0));
				double D = T(k) - H;
				W = W + alpha * D * X.submat(k).t();
			}
		}
	}

	void logitmodel::train(const mlldataset& dataset, const int maxIter, const double E)
	{
		// Set a train condition
		condition(maxIter, E);

		// Train the dataset
		train(dataset);
	}

	const double logitmodel::sigmoid(const double x)
	{
		// Activate a value by the sigmoid function
		return 2.0 / (1.0 + exp(-1.0 * x)) - 1.0;
	}

	const double logitmodel::predict(const algmat& x)
	{
		// predict a label
		return predict(x, nullptr);
	}

	const double logitmodel::predict(const algmat& x, double* score)
	{
		// Predict a result by the sigmoid function
		algmat y = x.dot(W);
		double z = sigmoid(y(0));

		// Save the response
		if (score != nullptr)
		{
			*score = z;
		}

		// Get a result label
		if (z < 0.5)
		{
			return -1.0;
		}
		else
		{
			return 1.0;
		}

		return z;
	}

	const int logitmodel::open(const string path, const string prefix)
	{
		// Open train information
		if (openTrainInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open weight information
		if (openWeightInfo(path, prefix) != 0)
		{
			return -1;
		}

		return 0;
	}

	const int logitmodel::openTrainInfo(const std::string path, const std::string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("LOGIT_MODEL_TRAIN_INFO", prefix)) == false)
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
			if (splitStrs[0] == "Dimension")
			{
				W = algmat::zeros(msize(atoi(splitStrs[1].c_str()), 1));
				continue;
			}
		}
		reader.close();

		return 0;
	}

	const int logitmodel::openWeightInfo(const std::string path, const std::string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("LOGIT_MODEL_WEIGHT_INFO", prefix)) == false)
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

			// Check the key format
			vector<string> indexStrs = split(splitStrs[0], "_");
			if (indexStrs.size() != 2)
			{
				continue;
			}

			// Set a value
			W(atoi(indexStrs[1].c_str())) = atof(splitStrs[1].c_str());
		}
		reader.close();

		return 0;
	}

	const int logitmodel::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save train information
		writer << getSectionName("LOGIT_MODEL_TRAIN_INFO", prefix) << endl;
		writer << "Error_Rate=" << E << endl;
		writer << "Max_Iter=" << maxIter << endl;
		writer << "Dimension=" << W.length() << endl;
		writer << endl;

		// Save weight information
		writer << getSectionName("LOGIT_MODEL_WEIGHT_INFO", prefix) << endl;
		for (int i = 0; i < W.length(); i++)
		{
			writer << "W_" << i << "=" << W(i) << endl;
		}
		writer << endl;
		writer.close();

		return 0;
	}
}