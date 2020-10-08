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
		inline void create(const dim& di);			// Create an array
		inline void create(const dim& di, const var& val);			// Create an array
		void release();			// Release an array
		void set(const var& val);			// Set a value
		inline const int length() const;			// Get a total length
		var* ptr() const;			// Get a data pointer
		bool empty() const;			// Check an array
		void cout() const;			// Console out for an array
		void _cout(const int nspace = 0) const;			// Console out for an array
		void subdim(const dim& di, var* tdata, const int didx);			// Set sub-dimensional array
		void subdata(var* ddata, const int tidx);			// Set sub-dimensional data
		template <const unsigned int M> ndarray<var, M> reshape(const dim& di) const;			// Reshape a matrix
		ndarray transpose(const dim& trd) const;			// Transpose a matrix

		// Operators
	public:
		ndarray& operator=(const ndarray& obj);			// data copy operator
		void operator=(const var& val);			// data setting operator
		const ndarray<var, N - 1>& operator[](const int idx) const;			// sub-dimensional array access operator (read)
		ndarray<var, N - 1>& operator[](const int idx);			// sub-dimensional array access operator (write)
		const var& operator()(const int idx) const;			// data access operator (read)
		var& operator()(const int idx);			// data access operator (write)
		ndarray<var, N> operator+(const ndarray<var, N>& mat) const;
		ndarray<var, N> operator+(const double val) const;
		friend ndarray<var, N> operator+(const double val, const ndarray<var, N>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.checkArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate an addition matrix
				ndarray<var, N> result(mat.dm);
				for (int i = 0; i < mat.tlen; i++)
				{
					result(i) = val + mat(i);
				}

				return result;
			}
			else
			{
				return mat;
			}
		}
		ndarray<var, N>& operator+=(const ndarray<var, N>& mat);
		ndarray<var, N>& operator+=(const double val);
		ndarray<var, N> operator-(const ndarray<var, N>& mat) const;
		ndarray<var, N> operator-(const double val) const;
		friend ndarray<var, N> operator-(const double val, const ndarray<var, N>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.checkArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate a subtraction matrix
				ndarray<var, N> result(mat.dm);
				for (int i = 0; i < mat.tlen; i++)
				{
					result(i) = val - mat(i);
				}

				return result;
			}
			else
			{
				return mat;
			}
		}
		ndarray<var, N>& operator-=(const ndarray<var, N>& mat);
		ndarray<var, N>& operator-=(const double val);
		ndarray<var, N> operator*(const ndarray<var, N>& mat) const;
		ndarray<var, N> operator*(const double val) const;
		friend ndarray<var, N> operator*(const double val, const ndarray<var, N>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.checkArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate a multiplication matrix
				ndarray<var, N> result(mat.dm);
				for (int i = 0; i < mat.tlen; i++)
				{
					result(i) = val * mat(i);
				}

				return result;
			}
			else
			{
				return mat;
			}
		}
		ndarray<var, N>& operator*=(const ndarray<var, N>& mat);
		ndarray<var, N>& operator*=(const double val);
		ndarray<var, N> operator/(const ndarray<var, N>& mat) const;
		ndarray<var, N> operator/(const double val) const;
		friend ndarray<var, N> operator/(const double val, const ndarray<var, N>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.checkArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);
				for (int i = 0; i < mat.tlen; i++)
				{
					assert(mat(i) != 0.0);
				}

				// Calculate a division matrix
				ndarray<var, N> result(mat.dm);
				for (int i = 0; i < mat.tlen; i++)
				{
					result(i) = val / mat(i);
				}

				return result;
			}
			else
			{
				return mat;
			}
		}
		ndarray<var, N>& operator/=(const ndarray<var, N>& mat);
		ndarray<var, N>& operator/=(const double val);

		// Constructors & Destructor
	public:
		ndarray();
		ndarray(const dim& di);
		ndarray(const dim& di, const var& val);
		ndarray(const ndarray& obj);
		virtual ~ndarray();

		// Variables
	protected:
		mutable ndarray<var, N - 1> subarr;			// sub-dimension array
		dim _dm;			// dimension information
		int didx;			// dimensional index
		int dlen;			// dimensional length
		var* ddata;			// dimensional data
		int step;			// width step
		int tidx;			// total index (now unused variable)
		int tlen;			// total length
		var* tdata;			// total data
		mutable bool nflag;			// number type flag
		mutable bool cflag;			// type check flag

		// Functions
	protected:
		virtual inline void setObject();			// Set an object
		virtual inline void copyObject(const ndarray& obj);			// Copy the object
		virtual inline void clearObject();			// Clear the object
		void widthStep();			// Calculate a width step
		void checkArrayType() const;			// Check the array type
		void calculateWidthstep(ndarray<int, 1>& nds, const dim& dm, const int d, const int cstep) const;			// Calculate N dimensional width steps
		void getTransIndices(const ndarray<int, 1>& nds, const dim& dm, const dim& trd, const int d, const int cstep, int addr, std::vector<int>& indices) const;			// Get all indices for transpose
		void getBroadcastIndices(const ndarray<int, 1>& nds, const dim& dm, const int d, const int s, const int cstep, int addr, std::vector<int>& indices) const;			// Get all indices for broadcasting
		inline ndarray add(const ndarray& mat) const;			// Calculate an addition matrix
		inline ndarray add(const double val) const;			// Calculate an addition matrix
		inline ndarray sub(const ndarray& mat) const;			// Calculate a subtraction matrix
		inline ndarray sub(const double val) const;			// Calculate a subtraction matrix
		inline ndarray mul(const ndarray& mat) const;			// Calculate a multiplication matrix
		inline ndarray mul(const double val) const;			// Calculate a multiplication matrix
		inline ndarray div(const ndarray& mat) const;			// Calculate a division matrix
		inline ndarray div(const double val) const;			// Calculate a division matrix

	};
}

#include "Ndarray.hpp"

#endif