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
		inline const std::string getType() const;			// Get the object type
		template<typename T> inline bool isEqualType(const T& var) const;			// Compare the objects

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
		virtual inline void setObject();			// Set an object
		virtual inline void copyObject(const object& obj);			// Copy the object
		virtual inline void clearObject();			// Clear the object
		template<typename T> inline void setType(const T& var);			// Set a object type

	};
}

#include "Object.hpp"

#endif