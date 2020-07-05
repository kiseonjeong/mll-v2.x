#ifndef NORMAL_BAYESIAN_H
#define NORMAL_BAYESIAN_H

namespace mll
{
	// The normal bayesian classifier
	class normalbayes : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		void train(const mlldataset& dataset);			// Train the dataset
		const double predict(const nml::algmat& x);			// Predict a response
		const int open(const std::string path, const std::string prefix = "");			// Open the trained parameters
		const int save(const std::string path, const std::string prefix = "");			// Save the trained parameters

		// Operators
	public:
		normalbayes& operator=(const normalbayes& obj);

		// Constructors & Destructor
	public:
		normalbayes();
		normalbayes(const mlldataset& dataset);
		normalbayes(const normalbayes& obj);
		~normalbayes();

		// Variables
	private:
		nml::algmat prior;			// prior probability
		nml::ndmatrix<1> count;			// count matrix
		nml::ndmatrix<3> mean;			// mean matrix
		nml::ndmatrix<3> cov;			// covariance matrix
		nml::ndmatrix<3> icov;			// inverse covariance matrix
		nml::algmat C;			// class vector
		int vrows;			// rows length
		int vcols;			// column length

		// Functions
	private:
		inline void setObject();			// Set an object
		inline void copyObject(const nml::object& obj);			// Copy the object
		inline void clearObject();			// Clear the object
		const int openLabelInfo(const std::string path, const std::string prefix);			// Open label information
		const int openProbInfo(const std::string path, const std::string prefix);			// Open probability information

	};
}

#endif