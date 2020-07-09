#ifndef SUPPORT_VECTOR_MACHINE
#define SUPPORT_VECTOR_MACHINE

namespace mll
{
	// The support vector machine classifier
	class SVM : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		// C : SVM margin parameter (small = soft margin, big = hard margin)
		// toler : tolerance
		// maxIter : maximum iterations
		// kn : kernel for trick
		void condition(const double C, const double toler, const int maxIter, const kernel& kn);			// Set a train condition
		// dataset : training dataset
		// C : SVM margin parameter (small = soft margin, big = hard margin)
		// toler : tolerance
		// maxIter : maximum iterations
		// kn : kernel for trick
		void train(const mlldataset& dataset);			// Train the dataset
		void train(const mlldataset& dataset, const double C, const double toler, const int maxIter, const kernel& kn);			// Train the dataset
		const double predict(const nml::algmat& x);			// Predict a response
		const double predict(const nml::algmat& x, double* distance);			// Predict a response
		const int open(const std::string path, const std::string prefix = "");			// Open the trained parameters
		const int save(const std::string path, const std::string prefix = "");			// Save the trained parameters

		// Operators
	public:
		SVM& operator=(const SVM& obj);

		// Constructors & Destructor
	public:
		SVM();
		// C : SVM margin parameter (small = soft margin, big = hard margin)
		// toler : tolerance
		// maxIter : maximum iterations
		// kn : kernel for trick
		SVM(const double C, const double toler, const int maxIter, const kernel& kn);
		// dataset : training dataset
		// C : SVM margin parameter (small = soft margin, big = hard margin)
		// toler : tolerance
		// maxIter : maximum iterations
		// kn : kernel for trick
		SVM(const mlldataset& dataset, const double C, const double toler, const int maxIter, const kernel& kn);
		SVM(const SVM& obj);
		~SVM();

		// Variables
	private:
		double C;			// panelty term for the soft margin
		double toler;			// tolerence parameter
		int maxIter;			// maximum iterations
		kernel* kn;			// kernel function
		nml::algmat A;			// lagrange multipliers
		nml::algmat W;			// normal Vector W
		double b;			// bias Paramter b
		nml::algmat X;			// feature vector
		nml::algmat T;			// target vector
		nml::algmat E;			// error cache
		nml::algmat K;			// kernel transform matrix
		int svC;			// number of the support vectors
		nml::algmat svX;			// feature vector X for the support vectors
		nml::algmat svT;			// target vector T for the support vectors
		nml::algmat svA;			// alpha parameters for the support vectors

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const nml::object& obj);			// Copy the object
		void clearObject();			// Clear the object
		void createKernel(const int type, const double deg, const double cons, const double th1, const double th2, const double sig);			// Create a kernel function
		void copyKernel(const kernel& kn);			// Copy the kernel function
		void remapDataset(const nml::algmat& X);			// Remap the dataset using the kernel function
		const int SMO(const mlldataset& dataset);			// The Sequential Minimal Optimization Algorithm
		const std::vector<int> findNonBound();			// Find a non-bound alpha list
		const int optimizeAlphaPair(const int i);			// Optimize an alpha pair
		const double calculateError(const int idx);			// Calculate error
		void updateError(const int idx);			// Update error
		void selectAnotherAlpha(const int i, const double Ei, int& j, double& Ej);			// Select another alpha
		const int getRandomNumber(const int i, const int min, const int max);			// Get a random number
		const double clipAlpha(const double alpha, const double H, const double L);			// Clip the alpha value
		const nml::algmat calculateW();			// Calculate a normal vector
		void getSupportVectorParams();			// Get the result parameters on the support vectors
		const int openTrainCondInfo(const std::string path, const std::string prefix);			// Open train condition information
		const int openKernelInfo(const std::string path, const std::string prefix);			// Open kernel information
		const int openSupportVectorInfo(const std::string path, const std::string prefix);			// Open support vectors information
		const int openLabelInfo(const std::string path, const std::string prefix);			// Open label information
		const int openAlphaInfo(const std::string path, const std::string prefix);			// Open alpha information

	};
}

#endif