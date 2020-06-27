#ifndef NDIMENSION_H
#define NDIMENSION_H

#include <iostream>

namespace nml
{
	// The dimension for array
	class dim
	{
		// Variables
	public:
		const std::vector<int>& d;			// dimension information
		prop::get<int> N;			// number of dimensions

		// Functions
	public:
		void create(const int N);			// Create N dimension information
		void release();			// Release N dimension information
		const int length() const;			// Get a total length
		const int get(const int idx) const;			// Get a dimension length
		void set(const int idx, const int len);			// Set a dimension length
		void reset(const int N, ...);			// Reset N dimension information
		void cout() const;			// Console out for N dimension information

		// Operators
	public:
		dim& operator=(const dim& obj);
		const int operator[](const int idx) const;

		// Constructors & Destructor
	public:
		dim();
		dim(const int N, ...);
		dim(const dim& obj);
		~dim();

		// Variables
	public:
		std::vector<int> _d;			// dimension information
		int _N;			// number of dimensions
		int len;			// length of dimensions

	};
}

#endif