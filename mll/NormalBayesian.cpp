#include "stdafx.h"
#include "NormalBayesian.h"

namespace mll
{
	normalbayes::normalbayes()
	{
		// Set an object
		setObject();
	}

	normalbayes::normalbayes(const mlldataset& dataset)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset);
	}

	normalbayes::normalbayes(const normalbayes& obj)
	{
		// Set an object
		setObject();

		// Clone Object
		*this = obj;
	}

	normalbayes::~normalbayes()
	{
		// Clear the object
		clearObject();
	}

	normalbayes& normalbayes::operator=(const normalbayes& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void normalbayes::setObject()
	{
		// Set the parameters
		setType(*this);
		vrows = -1;
		vcols = -1;
	}

	void normalbayes::copyObject(const object& obj)
	{
		// Do down casting
		const normalbayes* _obj = static_cast<const normalbayes*>(&obj);

		// Copy the parameters
		vrows = _obj->vrows;
		vcols = _obj->vcols;

		// Copy the memories
		prior = _obj->prior;
		count = _obj->count;
		mean = _obj->mean;
		cov = _obj->cov;
		icov = _obj->icov;
		C = _obj->C;
	}

	void normalbayes::clearObject()
	{
		// Clear the memories
		prior.release();
		count.release();
		mean.release();
		cov.release();
		icov.release();
		C.release();
	}

	void normalbayes::train(const mlldataset& dataset)
	{
		// Set vectors and initialize the parameters
		algmat X = dataset[0];
		algmat T = dataset[1];
		C = dataset[2];
		vrows = X.rows;
		vcols = X.cols;

		// Calculate a mean matrix
		count.create(dim(1, C.length()), 0);
		mean.create(dim(3, C.length(), 1, vcols), 0);
		for (int i = 0; i < vrows; i++)
		{
			algmat Xi = X.submat(i);
			for (int j = 0; j < C.length(); j++)
			{
				if (T(i) == C(j))
				{
					count[j]++;
					mean[j] += Xi;
				}
			}
		}
		for (int i = 0; i < C.length(); i++)
		{
			mean[i] /= count[i];
		}

		// Calculate a covariance matrix
		count.set(0);
		cov.create(dim(3, C.length(), vcols, vcols), 0);
		for (int i = 0; i < vrows; i++)
		{
			algmat Xi = X.submat(i);
			for (int j = 0; j < C.length(); j++)
			{
				if (T(i) == C(j))
				{
					count[j]++;
					cov[j] += algmat(Xi - mean[j]).t().dot(Xi - mean[j]);
				}
			}
		}
		for (int i = 0; i < C.length(); i++)
		{
			cov[i] /= count[i];
		}

		// Calculate an inverse covariance matrix
		icov = ndmatrix<3>(dim(3, C.length(), vcols, vcols));
		icov.set(0);
		for (int i = 0; i < C.length(); i++)
		{
			icov[i] = SVD::pinv(cov[i]);
		}

		// Calculate the prior probabilities
		prior = algmat::zeros(msize(1, C.length()));
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
	}

	const double normalbayes::predict(const algmat& x)
	{
		// Calculate a posterior probability
		algmat post(msize(1, C.length()));
		for (int i = 0; i < C.length(); i++)
		{
			post(i) = -0.5 * (algmat(x - mean[i]).dot(icov[i]).dot(algmat(x - mean[i]).t()))(0) - 0.5 * log(algmat(cov[i]).det()) + log(prior(i));
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

	const int normalbayes::open(const string path, const string prefix)
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

	const int normalbayes::openLabelInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("NORMAL_BAYESIAN_LABEL_INFO", prefix)) == false)
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
				C = algmat::zeros(msize(1, atoi(splitStrs[1].c_str())));
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

	const int normalbayes::openProbInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("NORMAL_BAYESIAN_PROB_INFO", prefix)) == false)
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
				mean = ndmatrix<3>(dim(3, C.length(), 1, vcols), 0);
				cov = ndmatrix<3>(dim(3, C.length(), vcols, vcols), 0);
				icov = ndmatrix<3>(dim(3, C.length(), vcols, vcols), 0);
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
					else if (indexStrs.size() == 4)
					{
						if (indexStrs[0] == "Mean" && indexStrs[1] == "Matrix")
						{
							mean[atoi(indexStrs[2].c_str())][0][atoi(indexStrs[3].c_str())] = atof(splitStrs[1].c_str());
						}
					}
					else if (indexStrs.size() == 5)
					{
						if (indexStrs[0] == "Cov" && indexStrs[1] == "Matrix")
						{
							cov[atoi(indexStrs[2].c_str())][atoi(indexStrs[3].c_str())][atoi(indexStrs[4].c_str())] = atof(splitStrs[1].c_str());
						}
					}
					else if (indexStrs.size() == 6)
					{
						if (indexStrs[0] == "Inv" && indexStrs[1] == "Cov" && indexStrs[2] == "Matrix")
						{
							icov[atoi(indexStrs[3].c_str())][atoi(indexStrs[4].c_str())][atoi(indexStrs[5].c_str())] = atof(splitStrs[1].c_str());
						}
					}
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int normalbayes::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save label information
		writer << getSectionName("NORMAL_BAYESIAN_LABEL_INFO", prefix) << endl;
		writer << "Num_C=" << C.length() << endl;
		for (int i = 0; i < C.length(); i++)
		{
			writer << "C_" << i << "=" << C(i) << endl;
		}
		writer << endl;

		// Save sample information
		writer << getSectionName("NORMAL_BAYESIAN_PROB_INFO", prefix) << endl;
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
				writer << "Mean_Matrix_" << i << "_" << j << "=" << mean[i][0][j] << endl;
			}
		}
		for (int i = 0; i < C.length(); i++)
		{
			for (int j = 0; j < vcols; j++)
			{
				for (int k = 0; k < vcols; k++)
				{
					writer << "Cov_Matrix_" << i << "_" << j << "_" << k << "=" << cov[i][j][k] << endl;
				}
			}
		}
		for (int i = 0; i < C.length(); i++)
		{
			for (int j = 0; j < vcols; j++)
			{
				for (int k = 0; k < vcols; k++)
				{
					writer << "Inv_Cov_Matrix_" << i << "_" << j << "_" << k << "=" << icov[i][j][k] << endl;
				}
			}
		}
		writer << endl;
		writer.close();

		return 0;
	}
}