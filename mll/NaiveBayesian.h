#ifndef NAIVE_BAYESIAN_H
#define NAIVE_BAYESIAN_H

namespace mll
{
	// The naive bayesian classifier
	class naivebayes : public mllclassifier
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
		naivebayes& operator=(const naivebayes& obj);

		// Constructors & Destructor
	public:
		naivebayes();
		naivebayes(const mlldataset& dataset);
		naivebayes(const naivebayes& obj);
		~naivebayes();

		// Variables
	private:
		nml::algmat prior;			// prior probability
		nml::algmat cond;			// conditional probability
		nml::algmat denom;			// normalization matrix
		nml::algmat C;			// class vector
		int vrows;			// row length
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