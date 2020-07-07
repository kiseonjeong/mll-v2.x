#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

namespace mll
{
	// The logistic regression classifier
	class logitmodel : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		void condition(const int maxIter, const double E);			// Set a train condition
		void train(const mlldataset& dataset);			// Train the dataset
		void train(const mlldataset& dataset, const int maxIter, const double E);			// Train the dataset
		const double predict(const nml::algmat& x);			// Predict a response
		const double predict(const nml::algmat& x, double* score);			// Predict a response
		const int open(const std::string path, const std::string prefix = "");			// Open the trained parameters
		const int save(const std::string path, const std::string prefix = "");			// Save the trained parameters

		// Operators
	public:
		logitmodel& operator=(const logitmodel& obj);

		// Constructors & Destructor
	public:
		logitmodel();
		logitmodel(const int maxIter, const double E);
		logitmodel(const mlldataset& dataset, const int maxIter, const double E);
		logitmodel(const logitmodel& obj);
		~logitmodel();

		// Variables
	private:
		int maxIter;			// max iteration
		double E;			// error rate
		nml::algmat W;			// weight matrix

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const nml::object& obj);			// Copy the object
		void clearObject();			// Clear the object
		const double sigmoid(const double x);			// Activate an input value
		const int openTrainInfo(const std::string path, const std::string prefix);			// Open train information
		const int openWeightInfo(const std::string path, const std::string prefix);			// Open weight information

	};
}

#endif