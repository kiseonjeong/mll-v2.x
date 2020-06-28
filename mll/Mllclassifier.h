#ifndef MLL_CLASSIFIER_H
#define MLL_CLASSIFIER_H

namespace mll
{
	// The classifier for mll
	class mllclassifier : public mllutility
	{
		// Variables
	public:


		// Functions
	public:
		virtual void train(const mlldataset& dataset) = 0;			// Train the dataset
		virtual const double predict(const nml::algmat& x) = 0;			// Predict a response
		virtual const int open(const std::string path, const std::string prefix = "") = 0;			// Open the trained parameters
		virtual const int save(const std::string path, const std::string prefix = "") = 0;			// Save the trained parameters

		// Variables
	protected:


		// Functions
	protected:
		const bool findSectionName(std::ifstream& reader, const std::string sectionName);			// Find section name
		const std::string getSectionName(const std::string name, const std::string prefix);			// Get section name for parameter writing

	};
}

#endif