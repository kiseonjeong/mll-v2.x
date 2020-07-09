#include "stdafx.h"
#include "SupportVectorMachine.h"

namespace mll
{
	SVM::SVM()
	{
		// Set an object
		setObject();
	}

	SVM::SVM(const double C, const double toler, const int maxIter, const kernel& kn)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(C, toler, maxIter, kn);
	}

	SVM::SVM(const mlldataset& dataset, const double C, const double toler, const int maxIter, const kernel& kn)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset, C, toler, maxIter, kn);
	}

	SVM::SVM(const SVM& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	SVM::~SVM()
	{
		// Clear the object
		clearObject();
	}

	SVM& SVM::operator=(const SVM& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	void SVM::setObject()
	{
		// Set the parameters
		setType(*this);
		C = 1.0;
		toler = 0.001;
		maxIter = 1000;
		b = 0.0;
		svC = 0;

		// Set the memories
		kn = nullptr;
		A.release();
		W.release();
		X.release();
		T.release();
		E.release();
		K.release();
		svX.release();
		svT.release();
		svA.release();
	}

	void SVM::copyObject(const object& obj)
	{
		// Do down casting
		const SVM* _obj = static_cast<const SVM*>(&obj);

		// Copy the parameters
		C = _obj->C;
		toler = _obj->toler;
		maxIter = _obj->maxIter;
		b = _obj->b;
		svC = _obj->svC;

		// Copy the memories
		copyKernel(*_obj->kn);
		A = _obj->A;
		W = _obj->W;
		X = _obj->X;
		T = _obj->T;
		E = _obj->E;
		K = _obj->K;
		svX = _obj->svX;
		svT = _obj->svT;
		svA = _obj->svA;
	}

	void SVM::clearObject()
	{
		// Clear the memories
		if (kn != nullptr)
		{
			delete kn;
		}
		A.release();
		W.release();
		X.release();
		T.release();
		E.release();
		K.release();
		svX.release();
		svT.release();
		svA.release();
	}

	void SVM::condition(const double C, const double toler, const int maxIter, const kernel& kn)
	{
		// Set the conditions
		this->C = C;
		this->toler = toler;
		this->maxIter = maxIter;
		copyKernel(kn);
	}

	void SVM::createKernel(const int type, const double deg, const double cons, const double th1, const double th2, const double sig)
	{
		// Check the old memory
		if (kn != nullptr)
		{
			delete kn;
		}

		// Create a kernel
		switch (type)
		{
		case SVM_KERNEL_LINEAR: kn = new linear_kernel(); break;
		case SVM_KERNEL_POLYNOMIAL: kn = new polynomial_kernel(deg, cons); break;
		case SVM_KERNEL_TANH: kn = new tanh_kernel(th1, th2); break;
		case SVM_KERNEL_RBF: kn = new rbf_kernel(sig); break;
		default: kn = nullptr; break;
		}
	}

	void SVM::copyKernel(const kernel& kn)
	{
		// Create the kernel
		createKernel(kn.type, kn.deg, kn.cons, kn.th1, kn.th2, kn.sig);
	}

	void SVM::train(const mlldataset& dataset)
	{
		// Check the number of the labels
		assert(dataset[2].length() == 2);

		// Check the label numbers
		assert(dataset[2](0) * dataset[2](1) == -1.0);

		// Do remapping the dataset using the kernel function
		remapDataset(dataset[0]);

		// Find lagrange multipliers by SMO Algorithm
		int smoResult = SMO(dataset);
		assert(smoResult == 0);

		// Get the result parameters on the found support vectors
		getSupportVectorParams();

		// Console out the number of the found support vectors
		cout << "The Number of the Support Vectors : " << svC << endl;
	}

	void SVM::train(const mlldataset& dataset, const double C, const double toler, const int maxIter, const kernel& kn)
	{
		// Set a train condition
		condition(C, toler, maxIter, kn);

		// Train the dataset
		train(dataset);
	}

	void SVM::remapDataset(const nml::algmat& X)
	{
		// Create a trick matrix
		K = algmat::zeros(msize(X.rows, X.rows));

		// Generate the remapping dataset
		for (int i = 0; i < K.rows; i++)
		{
			algmat Ki = kn->trick(X, X.submat(i));
			for (int j = 0; j < K.cols; j++)
			{
				K(i, j) = Ki(j);
			}
		}
	}

	const int SVM::SMO(const mlldataset& dataset)
	{
		// Initialize the parameters for SMO
		X = dataset[0];
		T = dataset[1];
		E = algmat::zeros(msize(X.rows, 2));
		A = algmat::zeros(msize(X.rows, 1));
		b = 0.0;

		// Start the SMO algorithm
		int iter = 0;
		int changed = 0;
		bool entire = true;
		while (iter < maxIter && (changed > 0 || entire == true))
		{
			// Set the zero alpha changed flag
			changed = 0;

			// Check the entire dataset flag
			if (entire == true)
			{
				// Optimize the alpha pairs
				for (int i = 0; i < X.rows; i++)
				{
					changed += optimizeAlphaPair(i);
				}
			}
			else
			{
				// Find a non-bound list
				vector<int> nonBound = findNonBound();

				// Optimize the alpha pairs
				for (int i = 0; i < (int)nonBound.size(); i++)
				{
					changed += optimizeAlphaPair(nonBound[i]);
				}
			}

			// Check the entire flag
			if (entire == true)
			{
				entire = false;
			}
			else if (changed == 0)
			{
				entire = true;
			}

			// Count the iteration
			iter++;
		}

		return 0;
	}

	const vector<int> SVM::findNonBound()
	{
		// Check a alpha list
		vector<int> nonBound;
		for (int i = 0; i < A.length(); i++)
		{
			if (A(i) > 0.0 && A(i) < C)
			{
				nonBound.push_back(i);
			}
		}

		return nonBound;
	}

	const int SVM::optimizeAlphaPair(const int i)
	{
		// Calculate the error for the 'i'th alpha
		double Ei = calculateError(i);

		// Select and optimize an alpha pair
		if ((T(i) * Ei < -toler && A(i) < C) || (T(i) * Ei > toler && A(i) > 0.0))
		{
			// Select the 'j'th alpha and calculate the error
			int j = 0;
			double Ej = 0;
			selectAnotherAlpha(i, Ei, j, Ej);

			// Backup the old alpha values
			double Ai = A(i);
			double Aj = A(j);
			double L = 0.0;
			double H = 0.0;
			if (T(i) != T(j))
			{
				L = max(0.0, A(j) - A(i));
				H = min(C, C + A(j) - A(i));
			}
			else
			{
				L = max(0.0, A(j) + A(i) - C);
				H = min(C, A(j) + A(i));
			}

			// Check the boundary
			if (L == H)
			{
				return 0;
			}

			// Calculate the Eta value
			double eta = 0.0;
			if (kn->empty() == true)
			{
				eta = 2.0 * X.submat(i).dot(X.submat(j).t())(0) - X.submat(i).dot(X.submat(i).t())(0) - X.submat(j).dot(X.submat(j).t())(0);
			}
			else
			{
				eta = 2.0 * K(i, j) - K(i, i) - K(j, j);
			}

			// Check the Eta value
			if (eta >= 0.0)
			{
				return 0;
			}

			// Update the 'j'th alpha
			A(j) -= (T(j) * (Ei - Ej) / eta);
			A(j) = clipAlpha(A(j), H, L);
			updateError(j);
			if (abs(A(j) - Aj) < 0.00001)
			{
				return 0;
			}

			// Update the 'i'th alpha
			A(i) = A(i) + (T(j) * T(i) * (Aj - A(j)));
			updateError(i);

			// Update the threshold values
			double b1 = 0.0;
			double b2 = 0.0;
			if (kn->empty() == true)
			{
				b1 = b - Ei - T(i) * (A(i) - Ai) * X.submat(i).dot(X.submat(i).t())(0) - T(j) * (A(j) - Aj) * X.submat(i).dot(X.submat(j).t())(0);
				b2 = b - Ej - T(i) * (A(i) - Ai) * X.submat(i).dot(X.submat(j).t())(0) - T(j) * (A(j) - Aj) * X.submat(j).dot(X.submat(j).t())(0);
			}
			else
			{
				b1 = b - Ei - T(i) * (A(i) - Ai) * K(i, i) - T(j) * (A(j) - Aj) * K(i, j);
				b2 = b - Ej - T(i) * (A(i) - Ai) * K(i, j) - T(j) * (A(j) - Aj) * K(j, j);
			}
			if (A(i) > 0.0 && A(i) < C)
			{
				b = b1;
			}
			else if (A(j) > 0.0 && A(j) < C)
			{
				b = b2;
			}
			else
			{
				b = (b1 + b2) / 2.0;
			}

			return 1;
		}
		else
		{
			return 0;
		}
	}

	const double SVM::calculateError(const int idx)
	{
		// Check the kernel function
		if (kn->empty() == true)
		{
			return (A.mul(T).t().dot(X.dot(X.submat(idx).t())) + b)(0) - T(idx);
		}
		else
		{
			return (A.mul(T).t().dot(K.submat(idx, 1)) + b)(0) - T(idx);
		}
	}

	void SVM::updateError(const int idx)
	{
		// Calculate the error
		double Eidx = calculateError(idx);

		// Update the cache memory
		E(idx, 0) = 1.0;
		E(idx, 1) = Eidx;
	}

	void SVM::selectAnotherAlpha(const int i, const double Ei, int& j, double& Ej)
	{
		// Initialize the cache memory
		E(i, 0) = 1.0;
		E(i, 1) = Ei;

		// Find a non-zero list
		vector<int> nonZero;
		for (int k = 0; k < X.rows; k++)
		{
			if (E(k, 0) != 0.0)
			{
				nonZero.push_back(k);
			}
		}

		// Select another alpha
		if (nonZero.size() > 1)
		{
			// Find a max delta value and the index
			int maxIdx = X.rows - 1;
			double maxDeltaEk = 0.0;
			for (int l = 0; l < (int)nonZero.size(); l++)
			{
				int k = nonZero[l];
				if (k == i)
				{
					continue;
				}
				double Ek = calculateError(k);
				double deltaEk = abs(Ei - Ek);
				if (deltaEk > maxDeltaEk)
				{
					maxDeltaEk = deltaEk;
					maxIdx = k;
					Ej = Ek;
				}
				j = maxIdx;
			}
		}
		else
		{
			// Randomly select an index
			j = getRandomNumber(i, 0, X.rows - 1);
			Ej = calculateError(j);
		}
	}

	const int SVM::getRandomNumber(const int i, const int min, const int max)
	{
		// Get a random number for another alpha
		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<int> dist(min, max);
		int j = i;
		while (j == i)
		{
			j = dist(gen);
		}

		return j;
	}

	const double SVM::clipAlpha(const double alpha, const double H, const double L)
	{
		// Clip the alpha value
		double clip = alpha;
		if (alpha > H)
		{
			clip = H;
		}
		if (alpha < L)
		{
			clip = L;
		}

		return clip;
	}

	const algmat SVM::calculateW()
	{
		// Calculate a normal vector using multipliers for the hyperplane
		algmat W(msize(X.cols, 1), 0.0);
		for (int i = 0; i < X.rows; i++)
		{
			W += X.submat(i).t() * A(i) * T(i);
		}

		return W;
	}

	void SVM::getSupportVectorParams()
	{
		// Find the support vectors
		vector<int> svI;
		for (int i = 0; i < X.rows; i++)
		{
			if (A(i) > 0.0)
			{
				svI.push_back(i);
			}
		}

		// Check the found index
		assert(svI.size() != 0);

		// Get the result parameters
		svC = (int)svI.size();
		svX = algmat::zeros(msize(svC, X.cols));
		svA = algmat::zeros(msize(1, svC));
		svT = algmat::zeros(msize(1, svC));
		for (int i = 0; i < svC; i++)
		{
			for (int j = 0; j < X.cols; j++)
			{
				svX(i, j) = X(svI[i], j);
			}
			svA(i) = A(svI[i]);
			svT(i) = T(svI[i]);
		}
	}

	const double SVM::predict(const nml::algmat& x)
	{
		// Predict a label
		return predict(x, nullptr);
	}

	const double SVM::predict(const nml::algmat& x, double* distance)
	{
		// Check the kernel function
		double response = 0.0;
		if (kn->empty() == true)
		{
			// Check the normal vector
			if (W.empty() == true)
			{
				// Calculate the normal vector
				W = algmat::zeros(msize(svX.cols, 1));
				for (int i = 0; i < svX.rows; i++)
				{
					W += svX.submat(i).t() * svA(i) * svT(i);
				}
			}

			// Calculate a response
			response = W.dot(x)(0) + b;
		}
		else
		{
			// Calculate a response
			response = kn->trick(svX, x).t().dot(svT.mul(svA).t())(0) + b;
		}

		// Save the response
		if (distance != nullptr)
		{
			*distance = response;
		}

		// Get a result label
		if (response < 0.0)
		{
			return -1.0;
		}
		else
		{
			return 1.0;
		}
	}

	const int SVM::open(const string path, const string prefix)
	{
		// Open train condition information
		if (openTrainCondInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open kernel information
		if (openKernelInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open support Vector information
		if (openSupportVectorInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open label information
		if (openLabelInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open alpha information
		if (openAlphaInfo(path, prefix) != 0)
		{
			return -1;
		}

		return 0;
	}

	const int SVM::openTrainCondInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("SVM_LEARNING_INFO", prefix)) == false)
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
			if (splitStrs[0] == "C")
			{
				C = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Toler")
			{
				toler = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Max_Iter")
			{
				maxIter = atoi(splitStrs[1].c_str());
				continue;
			}
		}
		reader.close();

		return 0;
	}

	const int SVM::openKernelInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("SVM_KERNEL_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		int trick = 0;
		int type = SVM_KERNEL_UNKNOWN;
		double deg = 0.0;
		double cons = 0.0;
		double th1 = 0.0;
		double th2 = 0.0;
		double sig = 0.0;
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
			if (splitStrs[0] == "Kernel_Trick")
			{
				trick = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Kernel_Type")
			{
				type = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "C")
			{
				cons = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Degree")
			{
				deg = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Th1")
			{
				th1 = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Th2")
			{
				th2 = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Sigma")
			{
				sig = atof(splitStrs[1].c_str());
				continue;
			}
		}
		reader.close();

		// Check the kernel trick flag
		if (trick != 0)
		{
			// Create a kernel
			createKernel(type, deg, cons, th1, th2, sig);
		}

		return 0;
	}

	const int SVM::openSupportVectorInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("SVM_SUPPORT_VECTOR_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		int dim = 0;
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
				dim = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Num_SV")
			{
				svC = atoi(splitStrs[1].c_str());
				svX = algmat::zeros(msize(svC, dim));
				svT = algmat::zeros(msize(1, svC));
				svA = algmat::zeros(msize(1, svC));
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
					if (indexStrs.size() != 4)
					{
						continue;
					}

					// Set a value
					svX(atoi(indexStrs[2].c_str()), atoi(indexStrs[3].c_str())) = atof(splitStrs[1].c_str());
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int SVM::openLabelInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("SVM_LABEL_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		int dim = 0;
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
			if (indexStrs.size() != 3)
			{
				continue;
			}

			// Set a value
			svT(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
		}
		reader.close();

		return 0;
	}

	const int SVM::openAlphaInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("SVM_ALPHA_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		int dim = 0;
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
			if (splitStrs[0] == "b")
			{
				b = atof(splitStrs[1].c_str());
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
					if (indexStrs.size() != 3)
					{
						continue;
					}

					// Set a value
					svA(atoi(indexStrs[2].c_str())) = atof(splitStrs[1].c_str());
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int SVM::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save train condition information
		writer << getSectionName("SVM_LEARNING_INFO", prefix) << endl;
		writer << "C=" << C << endl;
		writer << "Toler=" << toler << endl;
		writer << "Max_Iter=" << maxIter << endl;
		writer << endl;

		// Save kernel information
		writer << getSectionName("SVM_KERNEL_INFO", prefix) << endl;
		switch (kn->type)
		{
		case SVM_KERNEL_LINEAR:
			writer << "Kernel_Trick=1" << endl;
			writer << "Kernel_Type=0" << endl;
			break;
		case SVM_KERNEL_POLYNOMIAL:
			writer << "Kernel_Trick=1" << endl;
			writer << "Kernel_Type=1" << endl;
			writer << "C=" << kn->cons << endl;
			writer << "Degree=" << kn->deg << endl;
			break;
		case SVM_KERNEL_TANH:
			writer << "Kernel_Trick=1" << endl;
			writer << "Kernel_Type=2" << endl;
			writer << "Th1=" << kn->th1 << endl;
			writer << "Th2=" << kn->th2 << endl;
			break;
		case SVM_KERNEL_RBF:
			writer << "Kernel_Trick=1" << endl;
			writer << "Kernel_Type=3" << endl;
			writer << "Sigma=" << kn->sig << endl;
			break;
		default:
			writer << "Kernel_Trick=0" << endl;
			break;
		}
		writer << endl;

		// Save support vector information
		writer << getSectionName("SVM_SUPPORT_VECTOR_INFO", prefix) << endl;
		writer << "Dimension=" << svX.cols << endl;
		writer << "Num_SV=" << svC << endl;
		for (int i = 0; i < svX.rows; i++)
		{
			for (int j = 0; j < svX.cols; j++)
			{
				writer << "SV_X_" << i << "_" << j << "=" << svX(i, j) << endl;
			}
		}
		writer << endl;

		// Save label information
		writer << getSectionName("SVM_LABEL_INFO", prefix) << endl;
		for (int i = 0; i < svT.length(); i++)
		{
			writer << "SV_C_" << i << "=" << svT(i) << endl;
		}
		writer << endl;

		// Save alpha information
		writer << getSectionName("SVM_ALPHA_INFO", prefix) << endl;
		writer << "b=" << b << endl;
		for (int i = 0; i < svA.length(); i++)
		{
			writer << "SV_Alpha_" << i << "=" << svA(i) << endl;
		}
		writer << endl;
		writer.close();

		return 0;
	}
}