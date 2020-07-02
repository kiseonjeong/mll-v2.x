#ifndef NUMEM_H
#define NUMEM_H

namespace nml
{
	// The Memory for Numerical Method
	template <typename var> class numem
	{
		// Variables
	public:
		prop::get<int> size;			// data size information

		// Functions
	public:
		var* ptr() const;			// Get the data pointer
		bool empty() const;			// Check the data is empty or not
		void release();			// Release the data
		void create(const int size);			// Create the data
		void create(const int size, var val);			// Create the data
		void copy(const numem& obj);			// Copy the data
		void copy(const int size, const var*& data);			// Copy the data

		// Operators
	public:
		numem& operator=(const numem& obj);			// data copy operator
		const var& operator[](const int idx) const;			// data access operator (read)
		var& operator[](const int idx);			// data access operator (write)

		// Constructors & Destructor
	public:
		numem();
		numem(const int size);
		numem(const int size, const var& val);
		numem(const numem& obj);
		~numem();

		// Variables
	private:
		int _size;			// data size information
		var* _ptr;			// data pointer

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const numem& obj);			// Copy the object
		void clearObject();			// Clear the object
		void overwrite(int size);			// Overwrite the data

	};
}

#include "Numem.hpp"

#endif