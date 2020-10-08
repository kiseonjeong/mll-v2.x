#ifndef MSIZE_H
#define MSIZE_H

namespace nml
{
	// The Size for Matrix
	class msize
	{
		// Variables
	public:
		int d0;		// 1st Dimension
		int d1;		// 2nd Dimension

		// Functions
	public:
		void set(const int length);			// Set a vector
		void set(const int rows, const int cols);			// Set a matrix

		// Operators
	public:
		msize& operator=(const msize& obj);

		// Constructors & Destructor
	public:
		msize();
		explicit msize(const int length);
		explicit msize(const int rows, const int cols);
		~msize();

	};
}

#endif