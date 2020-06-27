#ifndef K_NEAREST_NEIGHBOR_H
#define K_NEAREST_NEIGHBOR_H

namespace mll
{
	// The K-nearest neighborhood classifier
	class KNN : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		void condition(const int K, const measure& meas);			// Set a train condition
		void train(const mlldataset& dataset);			// Train the dataset
		void train(const mlldataset& dataset, const int K, const measure& meas);			// Train the dataset
		const nml::algmat predict(const nml::algmat& x);			// Predict a response
		const int open(const std::string path, const std::string prefix = "");			// Open the trained parameters
		const int save(const std::string path, const std::string prefix = "");			// Save the trained parameters

		// Operators
	public:
		KNN& operator=(const KNN& obj);

		// Constructors & Destructor
	public:
		KNN();
		KNN(const int K, const measure& meas);
		KNN(const mlldataset& dataset, const int K, const measure& meas);
		KNN(const KNN& obj);
		~KNN();

		// Variables
	private:
		int K;			// number of neighbors
		measure* meas;			// distance measurement
		nml::algmat minVec;			// min vector
		nml::algmat maxVec;			// max vector
		nml::algmat M;			// mean vector
		nml::algmat X;			// feature vector
		nml::algmat T;			// target vector
		nml::algmat C;			// class vector

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const nml::object& obj);			// Copy the object
		void clearObject();			// Clear the object
		void createMeasure(const int type, const bool nflag, const double slope, const double intercept, const double p);			// Create a measurement for distance
		void copyMeasure(const measure& meas);			// Copy the measurement for distance
		void convertScale();			// Convert scale on the dataset
		void convertScale(nml::algmat& x) const;			// Convert scale on the dataset
		const int openMeasureInfo(const std::string path, const std::string prefix);			// Open measurement information
		const int openLabelInfo(const std::string path, const std::string prefix);			// Open label information
		const int openSampleInfo(const std::string path, const std::string prefix);			// Open sample information

	};
}

#endif