#ifndef ALGMAT_H
#define ALGMAT_H

namespace nml
{
	// The Matrix for Linear Algebra
	class algmat : public ndmatrix<2>
	{
		// Variables
	public:
		prop::get<int> rows;			// row information
		prop::get<int> cols;			// column information

		// Functions
	public:
		static algmat to_algmat(const std::vector<double>& vdata);			// Convert a vector data to the matrix
		static algmat to_algmat(const std::vector<double>& vdata, const int nrows);			// Convert a vector data to the matrix
		static algmat append(const std::vector<algmat>& vmat, const int axis = 0);			// Append a matrix vector
		algmat append(const algmat& mat, const int axis = 0) const;						// Append a matrix
		algmat submat(const int idx, const int axis = 0) const;			// Get a selected sub-matrix
		algmat submat(const int bi, const int ei, const int ii, const int axis = 0) const;			// Get a selected sub-matrix (bi : begin index, ei : end index, ii : interval index)
		algmat minor(const int idx, const int axis = 0) const;			// Get a minor matrix except the selected index
		algmat reshape(const int nrows) const;			// Reshape a matrix
		algmat t() const;			// Transpose a matrix
		void swap(const int fi, const int ti, const int axis = 0);			// Swap the matrix (fi : from index, ti : to index)
		void sort(const bool desc = false, const int axis = 0);			// Sort the matrix
		algmat count(const double val, const int axis = -1) const;			// Find a value in the matrix
		static algmat min(const algmat& mat, const int axis = -1);			// Find min. values
		static algmat max(const algmat& mat, const int axis = -1);			// Find max. values
		static algmat argmin(const algmat& mat, const int axis = -1);			// Find min. arguments
		static algmat argmax(const algmat& mat, const int axis = -1);			// Calculate max. arguments
		static algmat sum(const algmat& mat, const int axis = -1);			// Calculate sum values
		static algmat mean(const algmat& mat, const int axis = -1);			// Calculate mean values
		static algmat var(const algmat& mat, const int axis = -1);			// Calculate var. values
		static algmat std(const algmat& mat, const int axis = -1);			// Calculate std. values
		static const bool almost(const double val, const double target, const int round = 30);			// Compare the values
		algmat diag() const;			// Get a diagonal matrix
		static algmat values(const msize& sz, const double val);			// Get a values matrix
		static algmat zeros(const msize& sz);			// Get a zeros matrix
		static algmat ones(const msize& sz);			// Get an ones matrix
		static algmat eyes(const int len);			// Get an eyes matrix
		static algmat permut(const int len, const int i0, const int i1);			// Get a permutation matrix
		algmat uniq() const;			// Get an unique matrix
		algmat dot(const algmat& mat) const;			// Get a dot product matrix
		algmat cof() const;			// Get a cofactor matrix
		algmat inv() const;			// Get an inverse matrix
		const double det() const;			// Get a determinant value
		algmat add(const algmat& mat) const { return ndarray<double, 2>::add(mat); }			// Calculate an addition matrix
		algmat add(const double val) const { return ndarray<double, 2>::add(val); }			// Calculate an addition matrix
		algmat sub(const algmat& mat) const { return ndarray<double, 2>::sub(mat); }			// Calculate a subtraction matrix
		algmat sub(const double val) const { return ndarray<double, 2>::sub(val); }			// Calculate a subtraction matrix
		algmat mul(const algmat& mat) const { return ndarray<double, 2>::mul(mat); }			// Calculate a multiplication matrix
		algmat mul(const double val) const { return ndarray<double, 2>::mul(val); }			// Calculate a multiplication matrix
		algmat div(const algmat& mat) const { return ndarray<double, 2>::div(mat); }			// Calculate a division matrix
		algmat div(const double val) const { return ndarray<double, 2>::div(val); }			// Calculate a division matrix
		static algmat log(const algmat& mat) { return ndmatrix<2>::log(mat); };			// Calculate a log matrix
		static algmat log(const algmat& mat, const double b) { return ndmatrix<2>::log(mat, b); };			// Calculate a log matrix
		static algmat exp(const algmat& mat) { return ndmatrix<2>::exp(mat); };			// Calculate an exponential matrix
		static algmat pow(const algmat& mat, const double p) { return ndmatrix<2>::pow(mat, p); };			// Calculate an exponential matrix
		static algmat sqrt(const algmat& mat) { return ndmatrix<2>::sqrt(mat); };			// Calculate an exponential matrix
		static algmat abs(const algmat& mat) { return ndmatrix<2>::abs(mat); };			// Calculate an exponential matrix

		// Operators
	public:
		algmat& operator=(const algmat& obj);
		void operator=(const double& val);			// data setting operator
		const double& operator()(const int idx) const;			// data access operator (read)
		double& operator()(const int idx);			// data access operator (write)
		const double& operator()(const int idx0, const int idx1) const;			// data access operator (read)
		double& operator()(const int idx0, const int idx1);			// data access operator (write)
		algmat operator+(const algmat& mat) const { return ndarray<double, 2>::add(mat); }
		algmat operator+(const double val) const { return ndarray<double, 2>::add(val); }
		friend algmat operator+(const double val, const algmat& mat) { return val + ndarray<double, 2>(mat); }
		algmat& operator+=(const algmat& mat) { *this = ndarray<double, 2>::add(mat); return *this; }
		algmat& operator+=(const double val) { *this = ndarray<double, 2>::add(val); return *this; }
		algmat operator-(const algmat& mat) const { return ndarray<double, 2>::sub(mat); }
		algmat operator-(const double val) const { return ndarray<double, 2>::sub(val); }
		friend algmat operator-(const double val, const algmat& mat) { return val - ndarray<double, 2>(mat); }
		algmat& operator-=(const algmat& mat) { *this = ndarray<double, 2>::sub(mat); return *this; }
		algmat& operator-=(const double val) { *this = ndarray<double, 2>::sub(val); return *this; }
		algmat operator*(const algmat& mat) const { return ndarray<double, 2>::mul(mat); }
		algmat operator*(const double val) const { return ndarray<double, 2>::mul(val); }
		friend algmat operator*(const double val, const algmat& mat) { return val * ndarray<double, 2>(mat); }
		algmat& operator*=(const algmat& mat) { *this = ndarray<double, 2>::mul(mat); return *this; }
		algmat& operator*=(const double val) { *this = ndarray<double, 2>::mul(val); return *this; }
		algmat operator/(const algmat& mat) const { return ndarray<double, 2>::div(mat); }
		algmat operator/(const double val) const { return ndarray<double, 2>::div(val); }
		friend algmat operator/(const double val, const algmat& mat) { return val / ndarray<double, 2>(mat); }
		algmat& operator/=(const algmat& mat) { *this = ndarray<double, 2>::div(mat); return *this; }
		algmat& operator/=(const double val) { *this = ndarray<double, 2>::div(val); return *this; }

		// Constructors & Destructor
	public:
		algmat();
		algmat(const msize& sz);
		algmat(const msize& sz, const double val);
		algmat(const ndarray<double, 1>& obj);
		algmat(const ndarray<double, 2>& obj);
		algmat(const ndmatrix<1>& obj);
		algmat(const ndmatrix<2>& obj);
		algmat(const algmat& obj);
		~algmat();

		// Variables
	private:
		msize sz;			// matrix size information

		// Functions
	private:
		virtual void copyObject(const ndarray& obj);			// Copy the object
		static const algmat vappend(const std::vector<algmat>& vmat);			// Vertically, append a matrix vector
		static const algmat happend(const std::vector<algmat>& vmat);			// Horizontally, append a matrix vector
		const algmat vsubmat(int bi, int ei, int ii) const;			// Vertically, get a selected sub-matrix
		const algmat hsubmat(int bi, int ei, int ii) const;			// Horizontally, get a selected sub-matrix
		const algmat vminor(const int idx) const;			// Vertically, get a minor matrix except the selected index
		const algmat hminor(const int idx) const;			// Horizontally, get a minor matrix except the selected index
		void vswap(const int fi, const int ti);			// Vertically, swap the matrix
		void hswap(const int fi, const int ti);			// Horizontally, swap the matrix
		void vsort(const bool desc);			// Vertically, sort the matrix
		void hsort(const bool desc);			// Horizontally, sort the matrix
		const algmat vcount(const double val) const;			// Vertically, find a value in the matrix
		const algmat hcount(const double val) const;			// Horizontally, find a value in the matrix
		static const algmat vmin(const algmat& mat, algmat& loc);			// Vertically, find min. values
		static const algmat hmin(const algmat& mat, algmat& loc);			// Horizontally, find min. values
		static const algmat vmax(const algmat& mat, algmat& loc);			// Vertically, find max. values
		static const algmat hmax(const algmat& mat, algmat& loc);			// Horizontally, find max. values
		static const algmat vsum(const algmat& mat);			// Vertically, calculate sum values
		static const algmat hsum(const algmat& mat);			// Horizontally, calculate sum values
		const bool isSquared() const;			// Check a squared matrix or not
		const double expansion(const algmat& mat) const;			// Expand a matrix for cofactor matrix

	};
}

#endif