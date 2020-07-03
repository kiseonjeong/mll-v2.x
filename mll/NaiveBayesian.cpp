#include "stdafx.h"
#include "NaiveBayesian.h"

namespace mll
{
	naivebayes::naivebayes()
	{
		// Set an object
		setObject();
	}

	naivebayes::naivebayes(const mlldataset& dataset)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset);
	}

	naivebayes::naivebayes(const naivebayes& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	naivebayes::~naivebayes()
	{
		// Clear the object
		clearObject();
	}

	naivebayes& naivebayes::operator=(const naivebayes& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	inline void naivebayes::setObject()
	{
		// Set the parameters
		setType(*this);
		vrows = -1;
		vcols = -1;
	}

	inline void naivebayes::copyObject(const object& obj)
	{
		// Do down casting
		naivebayes* _obj = (naivebayes*)&obj;

		// Copy the parameters
		vrows = _obj->vrows;
		vcols = _obj->vcols;

		// Copy the memories
		prior = _obj->prior;
		cond = _obj->cond;
		denom = _obj->denom;
		C = _obj->C;
	}

	inline void naivebayes::clearObject()
	{
		// Clear the memories
		prior.release();
		cond.release();
		denom.release();
		C.release();
	}

	void naivebayes::train(const mlldataset& dataset)
	{
		// Set vectors and initialize the parameters
		algmat X = dataset[0];
		algmat T = dataset[1];
		C = dataset[2];
		vrows = X.rows;
		vcols = X.cols;

		// Calculate the prior probabilities
		prior = algmat(msize(1, C.length()), 0.0);
		for (int i = 0; i < vrows; i++)
		{
			for (int j = 0; j < C.length(); j++)
			{
				if (T(i) == C(j))
				{
					prior(j) += 1.0;
					break;
				}
			}
		}
		prior /= vrows;

		// Calculate the conditional probabilities
		denom = algmat(msize(1, C.length()), 0.0);
		cond = algmat(msize(C.length(), vcols), 0.0);
		for (int i = 0; i < vrows; i++)
		{
			// Get a sample vector
			algmat Xi = X.submat(i);

			// Check the label
			for (int j = 0; j < C.length(); j++)
			{
				if (T(i) == C(j))
				{
					for (int k = 0; k < vcols; k++)
					{
						cond(j, k) += Xi(k);
					}
					denom(j) += algmat::sum(Xi)(0);
					break;
				}
			}
		}
		for (int i = 0; i < C.length(); i++)
		{
			for (int j = 0; j < vcols; j++)
			{
				cond(i, j) = log(cond(i, j) / denom(i));
			}
		}
	}

	const double naivebayes::predict(const algmat& x)
	{
		// Calculate a posterior probability
		algmat post(msize(1, C.length()), 0.0);
		for (int i = 0; i < C.length(); i++)
		{
			post(i) = x.dot(cond.submat(i).t())(0) + log(prior(i));
		}

		// Find an argmax value
		int maxidx = 0;
		double maxval = post(0);
		for (int i = 1; i < C.length(); i++)
		{
			if (maxval < post(i))
			{
				maxval = post(i);
				maxidx = i;
			}
		}

		return C(maxidx);
	}

	const int naivebayes::open(const string path, const string prefix)
	{
		// Open label information
		if (openLabelInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open probability information
		if (openProbInfo(path, prefix) != 0)
		{
			return -1;
		}

		return 0;
	}

	const int naivebayes::openLabelInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("NAIVE_BAYESIAN_LABEL_INFO", prefix)) == false)
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
					C(atoi(indexStrs[1].c_str())) = atof(splitStrs[1].c_str());
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int naivebayes::openProbInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("NAIVE_BAYESIAN_PROB_INFO", prefix)) == false)
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
			if (splitStrs[0] == "Vector_Rows")
			{
				vrows = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Vector_Cols")
			{
				vcols = atoi(splitStrs[1].c_str());
				prior = algmat::zeros(msize(1, C.length()));
				cond = algmat::zeros(msize(C.length(), vcols));
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
						if (indexStrs[0] == "Prior" && indexStrs[1] == "Prob")
						{
							prior(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
						}
					}
					else if (indexStrs.size() == 5)
					{
						if (indexStrs[0] == "Log" && indexStrs[1] == "Conditional" && indexStrs[2] == "Prob")
						{
							cond(atoi(indexStrs[3].c_str()), atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
						}
					}
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int naivebayes::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save label information
		writer << getSectionName("NAIVE_BAYESIAN_LABEL_INFO", prefix) << endl;
		writer << "Num_C=" << C.length() << endl;
		for (int i = 0; i < C.length(); i++)
		{
			writer << "C_" << i << "=" << C(i) << endl;
		}
		writer << endl;

		// Save sample information
		writer << getSectionName("NAIVE_BAYESIAN_PROB_INFO", prefix) << endl;
		writer << "Vector_Rows=" << vrows << endl;
		writer << "Vector_Cols=" << vcols << endl;
		for (int i = 0; i < C.length(); i++)
		{
			writer << "Prior_Prob_" << i << "=" << prior(i) << endl;
		}
		for (int i = 0; i < C.length(); i++)
		{
			for (int j = 0; j < vcols; j++)
			{
				writer << "Log_Conditional_Prob_" << i << "_" << j << "=" << cond(i, j) << endl;
			}
		}
		writer << endl;
		writer.close();

		return 0;
	}
}