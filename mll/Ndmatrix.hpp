namespace nml
{
	template<const unsigned int N> ndmatrix<N>::ndmatrix()
	{
		// Set the number type flag
		nflag = true;
		cflag = false;
	}

	template<const unsigned int N> ndmatrix<N>::ndmatrix(const dim& di)
	{
		// Set the number type flag
		nflag = true;
		cflag = false;

		// Check a dimension value
		assert(di.N == N);

		// Create an array
		create(di);
	}

	template<const unsigned int N> ndmatrix<N>::ndmatrix(const dim& di, const double val)
	{
		// Set the number type flag
		nflag = true;
		cflag = false;

		// Check a dimension value
		assert(di.N == N);

		// Create an array
		create(di);

		// Set the array
		set(val);
	}

	template<const unsigned int N> ndmatrix<N>::ndmatrix(const ndarray<double, N>& obj)
	{
		// Set the number type flag
		nflag = true;
		cflag = false;

		// Copy the object
		copyObject(obj);
	}

	template<const unsigned int N> ndmatrix<N>::ndmatrix(const ndmatrix<N>& obj)
	{
		// Set the number type flag
		nflag = true;
		cflag = false;

		// Clone the object
		*this = obj;
	}

	template<const unsigned int N> ndmatrix<N>::~ndmatrix()
	{
		// Do nothing
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator=(const ndmatrix<N>& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	template<const unsigned int N> void ndmatrix<N>::operator=(const double& val)
	{
		// Set the array
		set(val);
	}

	template<const unsigned int N> inline const dim& ndmatrix<N>::shape() const
	{
		// Get dimension information
		return _dm;
	}

	template<const unsigned int N> const ndmatrix<N> ndmatrix<N>::clone() const
	{
		// Check and clone the matrix
		if (empty() == true)
		{
			return ndmatrix<N>();
		}
		else
		{
			return ndmatrix<N>(*this);
		}
	}

	template<const unsigned int N> void ndmatrix<N>::copy(const ndmatrix<N>& mat) const
	{
		// Check a status
		assert(empty() == false && mat.empty() == false);
		assert(_dm.N == mat._dm.N);
		for (int i = 0; i < _dm.N; i++)
		{
			assert(_dm[i] == mat._dm[i]);
		}

		// Copy a matrix
		for (int i = 0; i < tlen; i++)
		{
			ddata[i] = mat.ddata[i];
		}
	}

	template<const unsigned int N> void ndmatrix<N>::copy(const dim& di, const double* data) const
	{
		// Check a status
		assert(empty() == false);
		assert(_dm.N == di.N);
		for (int i = 0; i < di.N; i++)
		{
			assert(_dm[i] == di[i]);
		}

		// Copy a matrix
		for (int i = 0; i < tlen; i++)
		{
			this->ddata[i] = data[i];
		}
	}

	template<const unsigned int N> const ndmatrix<N + 1> ndmatrix<N>::append(const std::vector<ndmatrix>& vmat)
	{
		// Check dimension information
		assert(vmat.empty() == false);
		dim dm = vmat[0]._dm;
		const int count = (int)vmat.size();
		for (int i = 1; i < count; i++)
		{
			assert(dm.N == vmat[i]._dm.N);
			for (int j = 0; j < dm.N; j++)
			{
				assert(dm[j] == vmat[i]._dm[j]);
			}
		}

		// Set dimension information
		dim ndm;
		ndm.create(N + 1);
		ndm.set(0, count);
		for (int i = 1; i < ndm.N; i++)
		{
			ndm.set(i, dm[i - 1]);
		}

		// Copy the data
		ndmatrix<N + 1> result(ndm);
		const int step = vmat[0].tlen;
		for (int i = 0; i < count; i++)
		{
			for (int j = 0; j < step; j++)
			{
				result[i * step + j] = vmat[i].ddata[j];
			}
		}

		return result;
	}

	template<const unsigned int N> const ndmatrix<N + 1> ndmatrix<N>::append(const ndmatrix<N>& mat) const
	{
		// Set a matrix vector
		vector<ndmatrix<N>> vmat;
		vmat.push_back(*this);
		vmat.push_back(mat);

		// Append a matrix
		return append(vmat);
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::log(const ndmatrix<N>& mat)
	{
		// Check a status
		assert(mat.empty() == false);

		// Calculate a logarithm matrix
		ndmatrix result(mat.dm);
		for (int i = 0; i < mat.tlen; i++)
		{
			result.ddata[i] = std::log(mat.ddata[i]);
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::log(const ndmatrix<N>& mat, const double b)
	{
		// Check a status
		assert(mat.empty() == false);

		// Calculate a logarithm matrix
		ndmatrix result(mat.dm);
		const double temp = std::log(b);
		for (int i = 0; i < mat.tlen; i++)
		{
			result.ddata[i] = std::log(mat.ddata[i]) / temp;
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::exp(const ndmatrix<N>& mat)
	{
		// Check a status
		assert(mat.empty() == false);

		// Calculate an exponential matrix
		ndmatrix result(mat.dm);
		for (int i = 0; i < mat.tlen; i++)
		{
			result.ddata[i] = std::exp(mat.ddata[i]);
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::pow(const ndmatrix<N>& mat, const double p)
	{
		// Check a status
		assert(mat.empty() == false);

		// Calculate a power matrix
		ndmatrix result(mat.dm);
		for (int i = 0; i < mat.tlen; i++)
		{
			result.ddata[i] = std::pow(mat.ddata[i], p);
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::sqrt(const ndmatrix<N>& mat)
	{
		// Check a status
		assert(mat.empty() == false);

		// Calculate a square root matrix
		ndmatrix result(mat.dm);
		for (int i = 0; i < mat.tlen; i++)
		{
			result.ddata[i] = std::sqrt(mat.ddata[i]);
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::abs(const ndmatrix<N>& mat)
	{
		// Check a status
		assert(mat.empty() == false);

		// Calculate an absolute matrix
		ndmatrix result(mat.dm);
		for (int i = 0; i < mat.tlen; i++)
		{
			result.ddata[i] = std::abs(mat.ddata[i]);
		}

		return result;
	}
}
