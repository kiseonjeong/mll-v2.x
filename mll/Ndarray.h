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

		// Operators
	public:
		ndarray& operator=(const ndarray& obj);			// data copy operator
		void operator=(const var& val);			// data setting operator
		const ndarray<var, N - 1>& operator[](const int idx) const;			// sub-dimensional array access operator (read)
		ndarray<var, N - 1>& operator[](const int idx);			// sub-dimensional array access operator (write)
		const var& operator()(const int idx) const;			// data access operator (read)
		var& operator()(const int idx);			// data access operator (write)
		ndarray<var, N> operator+(const ndarray<var, N>& mat);
		ndarray<var, N> operator+(const double val);
		friend ndarray<var, N> operator+(const double val, ndarray<var, N>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
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
				return *this;
			}
		}
		friend ndarray<var, N>& operator+=(ndarray<var, N>& mat, const var val)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate an addition matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					mat(i) = mat(i) + val;
				}
			}

			return mat;
		}
		ndarray<var, N> operator-(const ndarray<var, N>& mat);
		ndarray<var, N> operator-(const double val);
		friend ndarray<var, N> operator-(const double val, ndarray<var, N>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
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
					result(i) = val - mat(i);
				}

				return result;
			}
			else
			{
				return *this;
			}
		}
		friend ndarray<var, N>& operator-=(ndarray<var, N>& mat, const double val)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate an addition matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					mat(i) = mat(i) - val;
				}
			}

			return mat;
		}
		ndarray<var, N> operator*(const ndarray<var, N>& mat);
		ndarray<var, N> operator*(const double val);
		friend ndarray<var, N> operator*(const double val, ndarray<var, N>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
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
					result(i) = val * mat(i);
				}

				return result;
			}
			else
			{
				return *this;
			}
		}
		friend ndarray<var, N>& operator*=(ndarray<var, N>& mat, const double val)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate an addition matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					mat(i) = mat(i) * val;
				}
			}

			return mat;
		}
		ndarray<var, N> operator/(const ndarray<var, N>& mat);
		ndarray<var, N> operator/(const double val);
		friend ndarray<var, N> operator/(const double val, ndarray<var, N>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
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

				// Calculate an addition matrix
				ndarray<var, N> result(mat.dm);
				for (int i = 0; i < mat.tlen; i++)
				{
					result(i) = val / mat(i);
				}

				return result;
			}
			else
			{
				return *this;
			}
		}
		friend ndarray<var, N>& operator/=(ndarray<var, N>& mat, const double val)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);
				assert(val != 0.0);

				// Calculate an addition matrix
				for (int i = 0; i < mat.length(); i++)
				{
					mat(i) = mat(i) / val;
				}
			}

			return mat;
		}

		// Constructors & Destructor
	public:
		ndarray();
		ndarray(const dim& di);
		ndarray(const dim& di, const var& val);
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
		int tidx;			// total index (now unused variable)
		int tlen;			// total length
		var* tdata;			// total data
		bool nflag;			// number type flag
		bool cflag;			// type check flag

		// Functions
	protected:
		virtual inline void setObject();			// Set an object
		virtual inline void copyObject(const ndarray& obj);			// Copy the object
		virtual inline void clearObject();			// Clear the object
		void widthStep();			// Calculate a width step

		// Functions
	private:
		void setArrayType();

	};

	// The 1 Dimensional Array
	template <typename var> class ndarray<var, 1>
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
		bool empty() const;			// Check the array
		void cout() const;			// Console out for an array
		void _cout(const int nspace = 0) const;			// Console out for an array
		void subdim(const dim& di, var* tdata, const int didx);			// Set sub-dimensional array
		void subdata(var* ddata, const int tidx);			// Set sub-dimensional data

		// Operators
	public:
		ndarray& operator=(const ndarray& obj);			// data copy operator
		void operator=(const var& val);			// data setting operator
		const var& operator[](const int idx) const;			// data access operator (read)
		var& operator[](const int idx);			// data access operator (write)
		const var& operator()(const int idx) const;			// data access operator (read)
		var& operator()(const int idx);			// data access operator (write)
		ndarray<var, 1> operator+(const ndarray<var, 1>& mat);
		ndarray<var, 1> operator+(const double val);
		friend ndarray<var, 1> operator+(const double val, ndarray<var, 1>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
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
				return *this;
			}
		}
		friend ndarray<var, 1>& operator+=(ndarray<var, 1>& mat, const var val)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate an addition matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					mat(i) = mat(i) + val;
				}
			}

			return mat;
		}
		ndarray<var, 1> operator-(const ndarray<var, 1>& mat);
		ndarray<var, 1> operator-(const double val);
		friend ndarray<var, 1> operator-(const double val, ndarray<var, 1>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
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
				return *this;
			}
		}
		friend ndarray<var, 1>& operator-=(ndarray<var, 1>& mat, const double val)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate a subtraction matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					mat(i) = mat(i) - val;
				}
			}

			return mat;
		}
		ndarray<var, 1> operator*(const ndarray<var, 1>& mat);
		ndarray<var, 1> operator*(const double val);
		friend ndarray<var, 1> operator*(const double val, ndarray<var, 1>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
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
				return *this;
			}
		}
		friend ndarray<var, 1>& operator*=(ndarray<var, 1>& mat, const double val)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);

				// Calculate a multiplication matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					mat(i) = mat(i) * val;
				}
			}

			return mat;
		}
		ndarray<var, 1> operator/(const ndarray<var, 1>& mat);
		ndarray<var, 1> operator/(const double val);
		friend ndarray<var, 1> operator/(const double val, ndarray<var, 1>& mat)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
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
				return *this;
			}
		}
		friend ndarray<var, 1>& operator/=(ndarray<var, 1>& mat, const double val)
		{
			// Check the check flag
			if (mat.cflag == true)
			{
				mat.setArrayType();
				mat.cflag = false;
			}

			// Check the type flag
			if (mat.nflag == true)
			{
				// Check a status
				assert(mat.empty() == false);
				assert(val != 0.0);

				// Calculate a division matrix
				for (int i = 0; i < mat.length(); i++)
				{
					mat(i) = mat(i) / val;
				}
			}

			return mat;
		}

		// Constructors & Destructor
	public:
		ndarray();
		ndarray(const dim& di);
		ndarray(const dim& di, const var& val);
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
		bool nflag;			// number type flag
		bool cflag;			// type check flag

		// Functions
	protected:
		virtual inline void setObject();			// Set an object
		virtual inline void copyObject(const ndarray& obj);			// Copy the object
		virtual inline void clearObject();			// Clear the object
		void widthStep();			// Calculate a width step

		// Functions
	private:
		void setArrayType();

	};
}

#include "Ndarray.hpp"

#endif