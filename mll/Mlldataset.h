#ifndef MLL_DATASET_H
#define MLL_DATASET_H

namespace mll
{
	// The dataset for mll
	class mlldataset : public mllutility
	{
		// Variables
	public:
		const nml::algmat& X;			// feature vector X
		const nml::algmat& T;			// target vector T
		nml::prop::get<int> dimension;			// dimension of feature vector
		nml::prop::get<int> nsample;			// number of samples
		const nml::algmat& C;			// class matrix
		nml::prop::get<int> nclass;			// number of classes
		const nml::algmat& slope;			// slope vector
		const nml::algmat& intercept;			// intercept vector

		// Functions
	public:
		const int open(const std::string path, const std::string separator = ",", const labelpos mode = LABEL_REAR);			// Open the dataset
		void set(const nml::algmat& X, const nml::algmat& T);			// Set the dataset
		void set(const nml::algmat& X);			// Set the dataset
		const bool empty() const;			// Check the dataset is empty or not
		void shuffle(const int iter = -1);			// Shuffle the dataset
		void scale();			// Scale the dataset
		void normalize();			// Normalize the dataset
		const std::vector<mlldataset> subdata(const int subsize, const bool shuffling = true) const;			// Generate the sub-datasets

		// Operators
	public:
		mlldataset& operator=(const mlldataset& obj);			// dataset copy operator
		const nml::algmat& operator[](const int idx) const;			// dataset access operator (read)

		// Constructors & Destructor
	public:
		mlldataset();
		mlldataset(const std::string path, const std::string separator = ",", const labelpos mode = LABEL_REAR);
		mlldataset(const nml::algmat& X, const nml::algmat& T);
		mlldataset(const nml::algmat& X);
		mlldataset(const mlldataset& obj);
		~mlldataset();

		// Variables
	private:
		nml::algmat _X;			// feature vector X
		nml::algmat _T;			// target vector T
		int _dimension;			// dimension of feature vector
		int _nsample;			// number of samples
		nml::algmat _C;			// class matrix
		int _nclass;			// number of classes
		nml::algmat _slope;			// slope vector
		nml::algmat _intercept;			// intercept vector

		// Functions
	private:
		inline void setObject();			// Set an object
		inline void copyObject(const nml::object& obj);			// Copy the object
		inline void clearObject();			// Clear the object

	};
}

#endif