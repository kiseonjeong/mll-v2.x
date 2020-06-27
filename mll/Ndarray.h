#ifndef NDARRAY_H
#define NDARRAY_H

namespace nml
{
	// The N Dimensional Array
	template <typename var, const unsigned int N> class ndarray
	{
		// Variables
	public:
		const dim& dm;			// dimension information

		// Functions
	public:
		void create(const dim& di);			// Create an array
		void release();			// Release an array
		void set(const var val);			// Set a value
		var* ptr() const;			// Get a data pointer
		bool empty() const;			// Check an array
		void cout() const;			// Console out for an array
		void _cout(const int nspace = 0) const;			// Console out for an array
		void subdim(const dim& di, var* tdata);			// Set sub-dimensional array
		void subdata(var* ddata, const int tidx);			// Set sub-dimensional data

		// Operators
	public:
		ndarray& operator=(const ndarray& obj);
		void operator=(const var val);
		operator var&();
		ndarray<var, N - 1>& operator[](const int idx) const;

		// Constructors & Destructor
	public:
		ndarray();
		ndarray(const dim& di);
		ndarray(const ndarray& obj);
		virtual ~ndarray();

		// Variables
	protected:
		ndarray<var, N - 1> sub;			// sub-dimension array
		dim _dm;			// dimension information
		int didx;			// dimensional index
		int dlen;			// dimensional length
		var* ddata;			// dimensional data
		int step;			// width step
		int tidx;			// total index
		int tlen;			// total length
		var* tdata;			// total data

		// Functions
	protected:
		virtual void setObject();			// Set an object
		virtual void copyObject(const ndarray& obj);			// Copy the object
		virtual void clearObject();			// Clear the object
		void widthStep();			// Calculate a width step

	};

	// The 1 Dimensional Array
	template <typename var> class ndarray<var, 1>
	{
		// Variables
	public:
		const dim& dm;			// dimension information

		// Functions
	public:
		void create(const dim& di);			// Create an array
		void release();			// Release an array
		void set(const var val);			// Set a value
		var* ptr() const;			// Get a data pointer
		bool empty() const;			// Check the array
		void cout() const;			// Console out for an array
		void _cout(const int nspace = 0) const;			// Console out for an array
		void subdim(const dim& di, var* tdata);			// Set sub-dimensional array
		void subdata(var* ddata, const int tidx);			// Set sub-dimensional data

		// Operators
	public:
		ndarray& operator=(const ndarray& obj);
		void operator=(const var val);
		operator var&();
		var& operator[](const int idx) const;

		// Constructors & Destructor
	public:
		ndarray();
		ndarray(const dim& di);
		ndarray(const ndarray& obj);
		virtual ~ndarray();

		// Variables
	protected:
		dim _dm;			// dimension information
		int didx;			// dimensional index
		int dlen;			// dimensional length
		var* ddata;			// dimensional data
		int step;			// width step
		int tidx;			// total index
		int tlen;			// total length
		var* tdata;			// total data

		// Functions
	protected:
		virtual void setObject();			// Set an object
		virtual void copyObject(const ndarray& obj);			// Copy the object
		virtual void clearObject();			// Clear the object
		void widthStep();			// Calculate a width step

	};
}

#include "Ndarray.hpp"

#endif