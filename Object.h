#ifndef OBJECT_H
#define OBJECT_H

namespace nml
{
	class object
	{
		// Variables
	public:


		// Functions
	public:
		const std::string getType() const;			// Get the object type
		template<typename T> bool isEqualType(const T& var) const;			// Compare the objects

		// Operators
	public:
		object& operator=(const object& obj);

		// Constructors & Destructor
	public:
		object();
		object(const object& obj);
		virtual ~object();

		// Variables
	protected:
		std::string objType;			// object type

		// Functions
	protected:
		virtual void setObject();			// Set an object
		virtual void copyObject(const object& obj);			// Copy the object
		virtual void clearObject();			// Clear the object
		template<typename T> void setType(const T& var);			// Set a object type

	};
}

#include "Object.hpp"

#endif