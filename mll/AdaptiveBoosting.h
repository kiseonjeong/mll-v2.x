#ifndef ADAPTIVE_BOOSTING_H
#define ADAPTIVE_BOOSTING_H

namespace mll
{
	// The stump data structure
	typedef struct _stump
	{
		double eps;
		nml::algmat pred;
		int dim;
		double thres;
		int type;
		double alpha;
	} stump;

	// The inequality type for adaboost
	typedef enum _ietype
	{
		INEQUAL_UNKNOWN = -1,
		INEQUAL_LT,
		INEQUAL_GT,
	} ietype;

	// The adaptive boosting classifier
	class adaboost : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		// nwc : target number of weak classifiers
		void condition(const int nwc);			// Set a train condition
		// dataset : training dataset
		void train(const mlldataset& dataset);			// Train the dataset
		// dataset : training dataset
		// nwc : target number of weak classifiers
		void train(const mlldataset& dataset, const int nwc);			// Train the dataset
		const double predict(const nml::algmat& x);			// Predict a response
		const int open(const std::string path, const std::string prefix = "");			// Open the trained parameters
		const int save(const std::string path, const std::string prefix = "");			// Save the trained parameters

		// Operators
	public:
		adaboost& operator=(const adaboost& obj);

		// Constructors & Destructor
	public:
		adaboost();
		// nwc : target number of weak classifiers
		adaboost(const int nwc);
		// dataset : training dataset
		// nwc : target number of weak classifiers
		adaboost(const mlldataset& dataset, const int nwc);
		adaboost(const adaboost& obj);
		~adaboost();

		// Variables
	private:
		int nwc;			// number of iterations
		int fdim;			// number of feature dimensions
		nml::algmat X;			// feature vector
		nml::algmat T;			// target vector
		nml::algmat C;			// class vector
		nml::algmat D;			// weight vector
		nml::algmat E;			// error vector
		std::vector<stump> WC;			// weak classifiers

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const nml::object& obj);			// Copy the object
		void clearObject();			// Clear the object
		const stump buildStump();			// Build a stump
		const nml::algmat classifyStump(const nml::algmat& feature, const double thres, const int inequal) const;			// Classify a stump
		const nml::algmat compareResults(const nml::algmat& pred, const nml::algmat& real) const;			// Compare classify results
		const double calculateError(const stump& bestStump) const;			// Calculate an error rate
		const nml::algmat sign(const nml::algmat& vector) const;			// Get a signed vector
		const int openTrainCondInfo(const std::string path, const std::string prefix);			// Open train condition information
		const int openWeakClassifierInfo(const std::string path, const std::string prefix);			// Open weak classifier information

	};
}

#endif