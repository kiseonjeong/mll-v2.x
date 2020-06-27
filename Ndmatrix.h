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
		const dim& shape() const;			// Get dimension information
		ndmatrix clone() const;			// Clone a matrix
		void copy(const ndmatrix& mat) const;			// Copy a matrix
		void copy(const dim& di, const double* data) const;			// Copy a matrix
		static ndmatrix<N + 1> append(const std::vector<ndmatrix>& vmat);			// Append a matrix vector to N + 1 dimensional matrix
		ndmatrix<N + 1> append(const ndmatrix& mat) const;						// Append a matrix to N + 1 dimensional matrix
		template <const unsigned int M> ndmatrix<M> reshape(const dim& di) const;			// Reshape a matrix
		ndmatrix transpose(const dim& trd) const;			// Transpose a matrix
		ndmatrix add(const ndmatrix& mat) const;			// Calculate an addition matrix
		ndmatrix add(const double val) const;			// Calculate an addition matrix
		ndmatrix sub(const ndmatrix& mat) const;			// Calculate a subtraction matrix
		ndmatrix sub(const double val) const;			// Calculate a subtraction matrix
		ndmatrix mul(const ndmatrix& mat) const;			// Calculate a multiplication matrix
		ndmatrix mul(const double val) const;			// Calculate a multiplication matrix
		ndmatrix div(const ndmatrix& mat) const;			// Calculate a division matrix
		ndmatrix div(const double val) const;			// Calculate a division matrix
		static ndmatrix log(const ndmatrix& mat);			// Calculate a log matrix
		static ndmatrix log(const ndmatrix& mat, const double b);			// Calculate a log matrix
		static ndmatrix exp(const ndmatrix& mat);			// Calculate an exponential matrix
		static ndmatrix pow(const ndmatrix& mat, const double p);			// Calculate an exponential matrix
		static ndmatrix sqrt(const ndmatrix& mat);			// Calculate an exponential matrix
		static ndmatrix abs(const ndmatrix& mat);			// Calculate an exponential matrix

		// Operators
	public:
		ndmatrix& operator=(const ndmatrix& obj);
		ndmatrix operator+(const ndmatrix& mat) const;
		ndmatrix operator+(const double val) const;
		ndmatrix& operator+=(const ndmatrix& mat);
		ndmatrix& operator+=(const double val);
		friend ndmatrix operator+(const double val, const ndmatrix& mat)
		{
			// Check a status
			assert(mat.empty() == false);

			// Calculate an addition matrix
			ndmatrix result(mat);
			for (int i = 0; i < result.tlen; i++)
			{
				result.ddata[i] = val + result.ddata[i];
			}

			return result;
		}
		ndmatrix operator-(const ndmatrix& mat) const;
		ndmatrix operator-(const double val) const;
		ndmatrix& operator-=(const ndmatrix& mat);
		ndmatrix& operator-=(const double val);
		friend ndmatrix operator-(const double val, const ndmatrix& mat)
		{
			// Check a status
			assert(mat.empty() == false);

			// Calculate a subtraction matrix
			ndmatrix result(mat);
			for (int i = 0; i < result.tlen; i++)
			{
				result.ddata[i] = val - result.ddata[i];
			}

			return result;
		}
		ndmatrix operator*(const ndmatrix& mat) const;
		ndmatrix operator*(const double val) const;
		ndmatrix& operator*=(const ndmatrix& mat);
		ndmatrix& operator*=(const double val);
		friend ndmatrix operator*(const double val, const ndmatrix& mat)
		{
			// Check a status
			assert(mat.empty() == false);

			// Calculate a multiplication matrix
			ndmatrix result(mat);
			for (int i = 0; i < result.tlen; i++)
			{
				result.ddata[i] = val * result.ddata[i];
			}

			return result;
		}
		ndmatrix operator/(const ndmatrix& mat) const;
		ndmatrix operator/(const double val) const;
		ndmatrix& operator/=(const ndmatrix& mat);
		ndmatrix& operator/=(const double val);
		friend const ndmatrix operator/(const double val, const ndmatrix& mat)
		{
			// Check a status
			assert(mat.empty() == false);
			for (int i = 0; i < mat.tlen; i++)
			{
				assert(mat.ddata[i] != 0.0);
			}

			// Calculate a division matrix
			ndmatrix result(mat);
			for (int i = 0; i < result.tlen; i++)
			{
				result.ddata[i] = val / result.ddata[i];
			}

			return result;
		}
		friend ndmatrix operator-(const ndmatrix& mat)
		{
			// Check a status
			assert(mat.empty() == false);

			// Convert a sign on the matrix
			ndmatrix result(mat);
			for (int i = 0; i < result.tlen; i++)
			{
				result.ddata[i] = -result.ddata[i];
			}

			return result;
		}

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
		void calculateWidthstep(const ndarray<int, 1>& nds, const dim& dm, const int d, const int cstep) const;			// Calculate N dimensional width steps
		void getTransIndices(const ndarray<int, 1>& nds, const dim& dm, const dim& trd, const int d, const int cstep, int addr, std::vector<int>& indices) const;			// Get all indices for transpose
		void getBroadcastIndices(const ndarray<int, 1>& nds, const dim& dm, const int d, const int s, const int cstep, int addr, std::vector<int>& indices) const;			// Get all indices for broadcasting

	};

	// A vector type
	typedef ndmatrix<1> algvec;
}

#include "ndmatrix.hpp"

#endif