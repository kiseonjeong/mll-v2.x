#ifndef NDMATRIX_H
#define NDMATRIX_H

namespace nml
{
	// The N Dimensional Matrix
	template <const unsigned int N> class ndmatrix : public ndarray<double, N>
	{
		// Variables
	public:


		// Functions
	public:
		inline const dim& shape() const;			// Get dimension information
		const ndmatrix clone() const;			// Clone a matrix
		void copy(const ndmatrix& mat) const;			// Copy a matrix
		void copy(const dim& di, const double* data) const;			// Copy a matrix
		static const ndmatrix<N + 1> append(const std::vector<ndmatrix>& vmat);			// Append a matrix vector to N + 1 dimensional matrix
		const ndmatrix<N + 1> append(const ndmatrix& mat) const;						// Append a matrix to N + 1 dimensional matrix
		static ndmatrix log(const ndmatrix& mat);			// Calculate a log matrix
		static ndmatrix log(const ndmatrix& mat, const double b);			// Calculate a log matrix
		static ndmatrix exp(const ndmatrix& mat);			// Calculate an exponential matrix
		static ndmatrix pow(const ndmatrix& mat, const double p);			// Calculate an exponential matrix
		static ndmatrix sqrt(const ndmatrix& mat);			// Calculate an exponential matrix
		static ndmatrix abs(const ndmatrix& mat);			// Calculate an exponential matrix
		ndmatrix add(const ndmatrix& mat) const { return ndarray<double, N>::add(mat); }			// Calculate an addition matrix
		ndmatrix add(const double val) const { return ndarray<double, N>::add(val); }			// Calculate an addition matrix
		ndmatrix sub(const ndmatrix& mat) const { return ndarray<double, N>::sub(mat); }			// Calculate a subtraction matrix
		ndmatrix sub(const double val) const { return ndarray<double, N>::sub(val); }			// Calculate a subtraction matrix
		ndmatrix mul(const ndmatrix& mat) const { return ndarray<double, N>::mul(mat); }			// Calculate a multiplication matrix
		ndmatrix mul(const double val) const { return ndarray<double, N>::mul(val); }			// Calculate a multiplication matrix
		ndmatrix div(const ndmatrix& mat) const { return ndarray<double, N>::div(mat); }			// Calculate a division matrix
		ndmatrix div(const double val) const { return ndarray<double, N>::div(val); }			// Calculate a division matrix

		// Operators
	public:
		ndmatrix& operator=(const ndmatrix& obj);			// data copy operator
		void operator=(const double& val);			// data setting operator
		ndmatrix operator+(const ndmatrix& mat) const { return ndarray<double, N>::add(mat); }
		ndmatrix operator+(const double val) const { return ndarray<double, N>::add(val); }
		friend ndmatrix operator+(const double val, const ndmatrix& mat) { return val + ndarray<double, N>(mat); }
		ndmatrix& operator+=(const ndmatrix& mat) { *this = ndarray<double, N>::add(mat); return *this; }
		ndmatrix& operator+=(const double val) { *this = ndarray<double, N>::add(val); return *this; }
		ndmatrix operator-(const ndmatrix& mat) const { return ndarray<double, N>::sub(mat); }
		ndmatrix operator-(const double val) const { return ndarray<double, N>::sub(val); }
		friend ndmatrix operator-(const double val, const ndmatrix& mat) { return val - ndarray<double, N>(mat); }
		ndmatrix& operator-=(const ndmatrix& mat) { *this = ndarray<double, N>::sub(mat); return *this; }
		ndmatrix& operator-=(const double val) { *this = ndarray<double, N>::sub(val); return *this; }
		ndmatrix operator*(const ndmatrix& mat) const { return ndarray<double, N>::mul(mat); }
		ndmatrix operator*(const double val) const { return ndarray<double, N>::mul(val); }
		friend ndmatrix operator*(const double val, const ndmatrix& mat) { return val * ndarray<double, N>(mat); }
		ndmatrix& operator*=(const ndmatrix& mat) { *this = ndarray<double, N>::mul(mat); return *this; }
		ndmatrix& operator*=(const double val) { *this = ndarray<double, N>::mul(val); return *this; }
		ndmatrix operator/(const ndmatrix& mat) const { return ndarray<double, N>::div(mat); }
		ndmatrix operator/(const double val) const { return ndarray<double, N>::div(val); }
		friend ndmatrix operator/(const double val, const ndmatrix& mat) { return val / ndarray<double, N>(mat); }
		ndmatrix& operator/=(const ndmatrix& mat) { *this = ndarray<double, N>::div(mat); return *this; }
		ndmatrix& operator/=(const double val) { *this = ndarray<double, N>::div(val); return *this; }

		// Constructors & Destructor
	public:
		ndmatrix();
		ndmatrix(const dim& di);
		ndmatrix(const dim& di, const double val);
		ndmatrix(const ndarray<double, N>& obj);
		ndmatrix(const ndmatrix& obj);
		virtual ~ndmatrix();

		// Variables
	private:


		// Functions
	private:


	};

	// A vector type
	typedef ndmatrix<1> algvec;
}

#include "ndmatrix.hpp"

#endif