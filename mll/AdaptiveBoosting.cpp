#include "stdafx.h"
#include "AdaptiveBoosting.h"

namespace mll
{
	adaboost::adaboost()
	{
		// Set an object
		setObject();
	}

	adaboost::adaboost(const int nwc)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(nwc);
	}

	adaboost::adaboost(const mlldataset& dataset, const int nwc)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset, nwc);
	}

	adaboost::adaboost(const adaboost& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	adaboost::~adaboost()
	{
		// Clear the object
		clearObject();
	}

	adaboost& adaboost::operator=(const adaboost& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	void adaboost::setObject()
	{
		// Set Parameters
		setType(*this);
		nwc = 0;
		fdim = 0;
	}

	void adaboost::copyObject(const object& obj)
	{
		// Do down casting
		const adaboost* _obj = static_cast<const adaboost*>(&obj);

		// Copy the parameters
		nwc = _obj->nwc;
		fdim = _obj->fdim;

		// Copy the memories
		X = _obj->X;
		T = _obj->T;
		C = _obj->C;
		D = _obj->D;
		E = _obj->E;
		WC = _obj->WC;
	}

	void adaboost::clearObject()
	{
		// Clear the memories
		X.release();
		T.release();
		C.release();
		D.release();
		E.release();
		WC.clear();
	}

	void adaboost::condition(const int nwc)
	{
		// Set the conditions
		this->nwc = nwc;
	}

	void adaboost::train(const mlldataset& dataset)
	{
		// Backup the dataset
		X = dataset[0];
		T = dataset[1];
		C = dataset[2];
		D = algmat::ones(msize(X.rows)) / (double)X.rows;
		E = algmat::zeros(msize(X.rows));

		// Find weak classifiers for the strong classifier
		fdim = X.cols;
		WC.clear();
		for (int i = 0; i < nwc; i++)
		{
			// Find a best stump
			stump bestStump = buildStump();

			// Calculate an alpha
			bestStump.alpha = 0.5 * log((1.0 - bestStump.eps) / max(bestStump.eps, 1e-16));
			WC.push_back(bestStump);

			// Update the weight matrix
			D = D.mul(algmat::exp((-1.0 * bestStump.alpha * T) * bestStump.pred));
			D = D / algmat::sum(D);
			E += bestStump.alpha * bestStump.pred;

			// Check the stop condition
			double e = calculateError(bestStump);
			cout << "Weak Classifier Index : " << i + 1 << "/" << nwc << ", Error Value : " << e << endl;
			if (e == 0.0)
			{
				break;
			}
		}
	}

	void adaboost::train(const mlldataset& dataset, const int nwc)
	{
		// Set a train condition
		condition(nwc);

		// Train the dataset
		train(dataset);	
	}

	const stump adaboost::buildStump()
	{
		// Find minimum and maximum column vectors
		algmat min = algmat::min(X, 0);
		algmat max = algmat::max(X, 0);

		// Find a best stump
		stump bestStump;
		const double numSteps = 10.0;
		double minError = DBL_MAX;
		for (int i = 0; i < X.cols; i++)
		{
			double stepSize = (max(i) - min(i)) / numSteps;
			for (double j = -1.0; j < numSteps + 1.0; j += 1.0)
			{
				for (int k = INEQUAL_LT; k <= INEQUAL_GT; k++)
				{
					double thres = min(i) + j * stepSize;
					algmat pred = classifyStump(X.submat(i, 1), thres, k);
					algmat error = D.t().dot(compareResults(pred, T));
					if (error(0) < minError)
					{
						minError = error(0);
						bestStump.eps = minError;
						bestStump.pred = pred;
						bestStump.dim = i;
						bestStump.thres = thres;
						bestStump.type = k;
					}
				}
			}
		}

		return bestStump;
	}

	const algmat adaboost::classifyStump(const algmat& feature, const double thres, const int inequal) const
	{
		// Check the inequality method
		algmat result(msize(feature.rows, 1), 1.0);
		if (inequal == INEQUAL_LT)
		{
			// Classify a vector using the stump
			for (int i = 0; i < result.length(); i++)
			{
				// less than
				if (feature(i) <= thres)
				{
					result(i) = -1.0;
				}
			}
		}
		else
		{
			// Classify a vector using the stump
			for (int i = 0; i < result.length(); i++)
			{
				// greater than
				if (feature(i) > thres)
				{
					result(i) = -1.0;
				}
			}
		}

		return result;
	}

	const algmat adaboost::compareResults(const algmat& pred, const algmat& real) const
	{
		// Compare the results, pred vs. real
		algmat result(msize(pred.length()), 1.0);
		for (int i = 0; i < pred.length(); i++)
		{
			if (pred(i) == real(i))
			{
				result(i) = 0.0;
			}
		}

		return result;
	}

	const double adaboost::calculateError(const stump& bestStump) const
	{
		// Calculate an error rate
		algmat S = sign(E);
		double errorRate = 0.0;
		for (int i = 0; i < S.length(); i++)
		{
			if (S(i) != T(i))
			{
				errorRate += 1.0;
			}
		}

		return errorRate / S.length();
	}

	const algmat adaboost::sign(const algmat& vector) const
	{
		// Check the sign
		algmat result(msize(vector.rows, vector.cols), 1.0);
		for (int i = 0; i < vector.length(); i++)
		{
			if (vector(i) < 0.0)
			{
				result(i) = -1.0;
			}
		}

		return result;
	}

	const double adaboost::predict(const algmat& x)
	{
		// Predict a response value using the weak classifiers
		algmat error(msize(x.rows), 0.0);
		for (int i = 0; i < (int)WC.size(); i++)
		{
			algmat pred = classifyStump(x.submat(WC[i].dim, 1), WC[i].thres, WC[i].type);
			error += WC[i].alpha * pred;
		}

		return sign(error)(0);
	}

	const int adaboost::open(const string path, const string prefix)
	{
		// Open train condition information
		if (openTrainCondInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open weak classifier information
		if (openWeakClassifierInfo(path, prefix) != 0)
		{
			return -1;
		}

		return 0;
	}

	const int adaboost::openTrainCondInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("ADABOOST_LEARNING_INFO", prefix)) == false)
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
			if (splitStrs[0] == "Num_Feature_Dim")
			{
				fdim = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Target_Num_Weak_Classifiers")
			{
				nwc = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Trained_Num_Weak_Classifiers")
			{
				WC = vector<stump>(atoi(splitStrs[1].c_str()));
				continue;
			}
		}
		reader.close();

		return 0;
	}

	const int adaboost::openWeakClassifierInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("ADABOOST_WEAK_CLASSIFIER_INFO", prefix)) == false)
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

			// Check key format
			vector<string> indexStrs = split(splitStrs[0], "_");
			if (indexStrs.size() != 4)
			{
				continue;
			}

			// Check key name
			if (indexStrs[0] == "Weak" && indexStrs[1] == "Classifier" && indexStrs[2] == "Dim")
			{
				int index = atoi(splitStrs[1].c_str());
				if (index >= fdim)
				{
					return -1;			// check the training dimension
				}
				WC[atoi(indexStrs[3].c_str())].dim = index;
				continue;
			}
			if (indexStrs[0] == "Weak" && indexStrs[1] == "Classifier" && indexStrs[2] == "Thres")
			{
				WC[atoi(indexStrs[3].c_str())].thres = atof(splitStrs[1].c_str());
				continue;
			}
			if (indexStrs[0] == "Weak" && indexStrs[1] == "Classifier" && indexStrs[2] == "Type")
			{
				WC[atoi(indexStrs[3].c_str())].type = atoi(splitStrs[1].c_str());
				continue;
			}
			if (indexStrs[0] == "Weak" && indexStrs[1] == "Classifier" && indexStrs[2] == "Alpha")
			{
				WC[atoi(indexStrs[3].c_str())].alpha = atof(splitStrs[1].c_str());
				continue;
			}
		}
		reader.close();

		return 0;
	}

	const int adaboost::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save train condition information
		writer << getSectionName("ADABOOST_LEARNING_INFO", prefix) << endl;
		writer << "Num_Feature_Dim=" << fdim << endl;
		writer << "Target_Num_Weak_Classifiers=" << nwc << endl;
		writer << "Trained_Num_Weak_Classifiers=" << (int)WC.size() << endl;
		writer << endl;

		// Save weak classifier information
		writer << getSectionName("ADABOOST_WEAK_CLASSIFIER_INFO", prefix) << endl;
		for (int i = 0; i < (int)WC.size(); i++)
		{
			writer << "Weak_Classifier_Dim_" << i << "=" << WC[i].dim << endl;
		}
		for (int i = 0; i < (int)WC.size(); i++)
		{
			writer << "Weak_Classifier_Thres_" << i << "=" << WC[i].thres << endl;
		}
		for (int i = 0; i < (int)WC.size(); i++)
		{
			writer << "Weak_Classifier_Type_" << i << "=" << WC[i].type << endl;
		}
		for (int i = 0; i < (int)WC.size(); i++)
		{
			writer << "Weak_Classifier_Alpha_" << i << "=" << WC[i].alpha << endl;
		}
		writer << endl;
		writer.close();

		return 0;
	}
}