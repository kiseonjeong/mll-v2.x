#ifndef MLL_UTILITY_H
#define MLL_UTILITY_H

namespace mll
{
	// The label position type
	typedef enum _labelpos
	{
		LABEL_REAR = 0,
		LABEL_FRONT,
		LABEL_EMPTY,
	} labelpos;

	// The utility for mll
	class mllutility : public nml::object
	{
		// Constructors & Destructor
	public:
		mllutility();
		~mllutility();

		// Variables
	protected:


		// Functions
	protected:
		virtual void setObject();			// Set an object
		std::string ltrim(std::string str) const;			// Trim a string from the left side
		std::string rtrim(std::string str) const;			// Trim a string from the right side
		std::string trim(const std::string str) const;			// Trim a string from the both sides
		std::vector<std::string> split(const std::string str, const std::string separator) const;			// Split a string by the separator

	};
}

#endif