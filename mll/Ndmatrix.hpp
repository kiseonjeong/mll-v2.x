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

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::operator+(const ndmatrix<N>& mat) const
	{
		// Calculate an addition matrix
		return add(mat);
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::operator+(const double val) const
	{
		// Calculate an addition matrix
		return add(val);
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator+=(const ndmatrix<N>& mat)
	{
		// Calculate an addition matrix
		*this = add(mat);

		return *this;
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator+=(const double val)
	{
		// Calculate an addition matrix
		*this = add(val);

		return *this;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::operator-(const ndmatrix<N>& mat) const
	{
		// Calculate a subtraction matrix
		return sub(mat);
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::operator-(const double val) const
	{
		// Calculate a subtraction matrix
		return sub(val);
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator-=(const ndmatrix<N>& mat)
	{
		// Calculate a subtraction matrix
		*this = sub(mat);

		return *this;
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator-=(const double val)
	{
		// Calculate a subtraction matrix
		*this = sub(val);

		return *this;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::operator*(const ndmatrix<N>& mat) const
	{
		// Calculate a multiplication matrix
		return mul(mat);
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::operator*(const double val) const
	{
		// Calculate a multiplication matrix
		return mul(val);
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator*=(const ndmatrix<N>& mat)
	{
		// Calculate a multiplication matrix
		*this = mul(mat);

		return *this;
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator*=(const double val)
	{
		// Calculate a multiplication matrix
		*this = mul(val);

		return *this;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::operator/(const ndmatrix<N>& mat) const
	{
		// Calculate a division matrix
		return div(mat);
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::operator/(const double val) const
	{
		// Calculate a division matrix
		return div(val);
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator/=(const ndmatrix<N>& mat)
	{
		// Calculate a division matrix
		*this = div(mat);

		return *this;
	}

	template<const unsigned int N> ndmatrix<N>& ndmatrix<N>::operator/=(const double val)
	{
		// Calculate a division matrix
		*this = div(val);

		return *this;
	}

	template<const unsigned int N> inline const dim& ndmatrix<N>::shape() const
	{
		// Get dimension information
		return _dm;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::clone() const
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

	template<const unsigned int N> ndmatrix<N + 1> ndmatrix<N>::append(const std::vector<ndmatrix>& vmat)
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

	template<const unsigned int N> ndmatrix<N + 1> ndmatrix<N>::append(const ndmatrix<N>& mat) const
	{
		// Set a matrix vector
		vector<ndmatrix<N>> vmat;
		vmat.push_back(*this);
		vmat.push_back(mat);

		// Append a matrix
		return append(vmat);
	}

	template<const unsigned int N> template <const unsigned int M> ndmatrix<M> ndmatrix<N>::reshape(const dim& di) const
	{
		// Check dimension information
		assert(di.N == M);

		// Calculate a target length
		int acount = 0;
		int totlen = 1;
		for (int i = 0; i < di.N; i++)
		{
			if (di[i] > 0)
			{
				totlen *= di[i];
			}
			else
			{
				acount++;
			}
		}

		// Check an auto dimension count
		dim rdm = di;
		if (acount > 1)
		{
			// Copy the original dimension information
			assert(acount < 2 && "reshape : Unknown dimension information");
			rdm = _dm;
		}
		else
		{
			// Calculate reshape dimension information
			for (int i = 0; i < rdm.N; i++)
			{
				if (rdm[i] < 1)
				{
					rdm.set(i, tlen / totlen);
				}
			}
		}

		// Calculate a reshaped length
		int reslen = 1;
		for (int i = 0; i < rdm.N; i++)
		{
			reslen *= rdm[i];
		}

		// Check the reshaped length
		if (tlen != reslen)
		{
			// Copy the original dimension information
			assert(tlen == reslen);
			rdm = _dm;
		}

		// Copy the data
		ndmatrix<M> result(rdm);
		for (int i = 0; i < tlen; i++)
		{
			result.ptr()[i] = ddata[i];
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::transpose(const dim& trd) const
	{
		// Calculate steps for N dimensions
		ndarray<int, 1> nds(dim(1, trd.N));
		dim dm = _dm;
		const int d = 0;
		int cstep = 1;
		for (int i = 1; i < _dm.N; i++)
		{
			cstep *= _dm[i];
		}
		calculateWidthstep(nds, dm, d, cstep);

		// Get transpose indices
		vector<int> tri;
		for (int i = 0; i < dm.N; i++)
		{
			dm.set(i, _dm[trd[i]]);
		}
		getTransIndices(nds, dm, trd, 0, nds[trd[0]], 0, tri);

		// Transpose a matrix
		ndmatrix<N> result(dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[tri[i]];
		}

		return result;
	}

	template<const unsigned int N> void ndmatrix<N>::calculateWidthstep(ndarray<int, 1>& nds, const dim& dm, const int d, const int cstep) const
	{
		// Calculate a next step value
		int nstep = 1;
		for (int i = d + 2; i < dm.N; i++)
		{
			nstep *= dm[i];
		}
		nds[d] = cstep;

		// Check a dimension index
		if (d < N - 1)
		{
			calculateWidthstep(nds, dm, d + 1, nstep);
		}
	}

	template<const unsigned int N> void ndmatrix<N>::getTransIndices(const ndarray<int, 1>& nds, const dim& dm, const dim& trd, const int d, const int cstep, int addr, std::vector<int>& indices) const
	{
		// Calculate an index
		int idx = addr;
		for (int i = 0; i < dm[d]; i++)
		{
			idx = addr + i * cstep;
			if (d < N - 1)
			{
				getTransIndices(nds, dm, trd, d + 1, nds[trd[d + 1]], idx, indices);
			}
			else
			{
				indices.push_back(idx);
			}
		}
	}

	template<const unsigned int N> void ndmatrix<N>::getBroadcastIndices(const ndarray<int, 1>& nds, const dim& dm, const int d, const int s, const int cstep, int addr, std::vector<int>& indices) const
	{
		// Calculate an index
		int idx = addr;
		for (int i = 0; i < dm[d]; i++)
		{
			if (d == s)
			{
				idx = addr;
			}
			else
			{
				idx = addr + i * cstep;
			}
			if (d < N - 1)
			{
				getBroadcastIndices(nds, dm, d + 1, s, nds[d + 1], idx, indices);
			}
			else
			{
				indices.push_back(idx);
			}
		}
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::add(const ndmatrix<N>& mat) const
	{
		// Check a status
		assert(empty() == false);
		assert(mat.empty() == false);
		assert(_dm.N == mat._dm.N);
		if (mat.length() == 1)
		{
			return add(mat.ddata[0]);
		}
		else if (length() == 1)
		{
			return ddata[0] + mat;
		}
		int didx = 0;
		int dcount = 0;
		for (int i = 0; i < _dm.N; i++)
		{
			if (_dm[i] != mat._dm[i])
			{
				didx = i;
				dcount++;
			}
		}
		if (dcount > 1)
		{
			assert(_dm[didx] == mat._dm[didx]);
		}

		// Check a broadcasting flag
		ndmatrix result;
		if (dcount == 1)
		{
			// Check a total length
			ndarray<int, 1> nds(dim(1, mat.dm.N));
			vector<int> bci;
			const int d = 0;
			int cstep = 1;
			if (length() > mat.length())
			{
				// Calculate steps for N dimensions
				for (int i = 1; i < mat.dm.N; i++)
				{
					cstep *= mat.dm[i];
				}
				calculateWidthstep(nds, mat.dm, d, cstep);

				// Get broadcasting indices
				getBroadcastIndices(nds, _dm, 0, didx, nds[0], 0, bci);

				// Create a result memory
				result.create(_dm);

				// Calculate an addition matrix
				for (int i = 0; i < tlen; i++)
				{
					result.ddata[i] = ddata[i] + mat.ddata[bci[i]];
				}
			}
			else
			{
				// Calculate steps for N dimensions
				for (int i = 1; i < _dm.N; i++)
				{
					cstep *= _dm[i];
				}
				calculateWidthstep(nds, _dm, d, cstep);

				// Get broadcasting indices
				getBroadcastIndices(nds, mat.dm, 0, didx, nds[0], 0, bci);

				// Create a result memory
				result.create(mat.dm);

				// Calculate an addition matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					result.ddata[i] = ddata[bci[i]] + mat.ddata[i];
				}
			}
		}
		else
		{
			// Create a result memory
			result.create(_dm);

			// Calculate an addition matrix
			for (int i = 0; i < tlen; i++)
			{
				result.ddata[i] = ddata[i] + mat.ddata[i];
			}
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::add(const double val) const
	{
		// Check a status
		assert(empty() == false);

		// Calculate an addition matrix
		ndmatrix result(_dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[i] + val;
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::sub(const ndmatrix<N>& mat) const
	{
		// Check a status
		assert(empty() == false);
		assert(mat.empty() == false);
		assert(_dm.N == mat._dm.N);
		if (mat.length() == 1)
		{
			return sub(mat.ddata[0]);
		}
		else if (length() == 1)
		{
			return ddata[0] - mat;
		}
		int didx = 0;
		int dcount = 0;
		for (int i = 0; i < _dm.N; i++)
		{
			if (_dm[i] != mat._dm[i])
			{
				didx = i;
				dcount++;
			}
		}
		if (dcount > 1)
		{
			assert(_dm[didx] == mat._dm[didx]);
		}

		// Check a broadcasting flag
		ndmatrix result;
		if (dcount == 1)
		{
			// Check a total length
			ndarray<int, 1> nds(dim(1, mat.dm.N));
			vector<int> bci;
			const int d = 0;
			int cstep = 1;
			if (length() > mat.length())
			{
				// Calculate steps for N dimensions
				for (int i = 1; i < mat.dm.N; i++)
				{
					cstep *= mat.dm[i];
				}
				calculateWidthstep(nds, mat.dm, d, cstep);

				// Get broadcasting indices
				getBroadcastIndices(nds, _dm, 0, didx, nds[0], 0, bci);

				// Create a result memory
				result.create(_dm);

				// Calculate a subtraction matrix
				for (int i = 0; i < tlen; i++)
				{
					result.ddata[i] = ddata[i] - mat.ddata[bci[i]];
				}
			}
			else
			{
				// Calculate steps for N dimensions
				for (int i = 1; i < _dm.N; i++)
				{
					cstep *= _dm[i];
				}
				calculateWidthstep(nds, _dm, d, cstep);

				// Get broadcasting indices
				getBroadcastIndices(nds, mat.dm, 0, didx, nds[0], 0, bci);

				// Create a result memory
				result.create(mat.dm);

				// Calculate a subtraction matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					result.ddata[i] = ddata[bci[i]] - mat.ddata[i];
				}
			}
		}
		else
		{
			// Create a result memory
			result.create(_dm);

			// Calculate a subtraction matrix
			for (int i = 0; i < tlen; i++)
			{
				result.ddata[i] = ddata[i] - mat.ddata[i];
			}
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::sub(const double val) const
	{
		// Check a status
		assert(empty() == false);

		// Calculate a subtraction matrix
		ndmatrix result(_dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[i] - val;
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::mul(const ndmatrix<N>& mat) const
	{
		// Check a status
		assert(empty() == false);
		assert(mat.empty() == false);
		assert(_dm.N == mat._dm.N);
		if (mat.length() == 1)
		{
			return mul(mat.ddata[0]);
		}
		else if (length() == 1)
		{
			return ddata[0] * mat;
		}
		int didx = 0;
		int dcount = 0;
		for (int i = 0; i < _dm.N; i++)
		{
			if (_dm[i] != mat._dm[i])
			{
				didx = i;
				dcount++;
			}
		}
		if (dcount > 1)
		{
			assert(_dm[didx] == mat._dm[didx]);
		}

		// Check a broadcasting flag
		ndmatrix result;
		if (dcount == 1)
		{
			// Check a total length
			ndarray<int, 1> nds(dim(1, mat.dm.N));
			vector<int> bci;
			const int d = 0;
			int cstep = 1;
			if (length() > mat.length())
			{
				// Calculate steps for N dimensions
				for (int i = 1; i < mat.dm.N; i++)
				{
					cstep *= mat.dm[i];
				}
				calculateWidthstep(nds, mat.dm, d, cstep);

				// Get broadcasting indices
				getBroadcastIndices(nds, _dm, 0, didx, nds[0], 0, bci);

				// Create a result memory
				result.create(_dm);

				// Calculate a multiplication matrix
				for (int i = 0; i < tlen; i++)
				{
					result.ddata[i] = ddata[i] * mat.ddata[bci[i]];
				}
			}
			else
			{
				// Calculate steps for N dimensions
				for (int i = 1; i < _dm.N; i++)
				{
					cstep *= _dm[i];
				}
				calculateWidthstep(nds, _dm, d, cstep);

				// Get broadcasting indices
				getBroadcastIndices(nds, mat.dm, 0, didx, nds[0], 0, bci);

				// Create a result memory
				result.create(mat.dm);

				// Calculate a multiplication matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					result.ddata[i] = ddata[bci[i]] * mat.ddata[i];
				}
			}
		}
		else
		{
			// Create a result memory
			result.create(_dm);

			// Calculate a multiplication matrix
			for (int i = 0; i < tlen; i++)
			{
				result.ddata[i] = ddata[i] * mat.ddata[i];
			}
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::mul(const double val) const
	{
		// Check a status
		assert(empty() == false);

		// Calculate a multiplication matrix
		ndmatrix result(_dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[i] * val;
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::div(const ndmatrix<N>& mat) const
	{
		// Check a status
		assert(empty() == false);
		assert(mat.empty() == false);
		assert(_dm.N == mat._dm.N);
		if (mat.length() == 1)
		{
			return div(mat.ddata[0]);
		}
		else if (length() == 1)
		{
			return ddata[0] / mat;
		}
		int didx = 0;
		int dcount = 0;
		for (int i = 0; i < _dm.N; i++)
		{
			if (_dm[i] != mat._dm[i])
			{
				didx = i;
				dcount++;
			}
		}
		if (dcount > 1)
		{
			assert(_dm[didx] == mat._dm[didx]);
		}
		for (int i = 0; i < mat.tlen; i++)
		{
			assert(mat.ddata[i] != 0.0);
		}

		// Check a broadcasting flag
		ndmatrix result(_dm);
		if (dcount == 1)
		{
			// Check a total length
			ndarray<int, 1> nds(dim(1, mat.dm.N));
			vector<int> bci;
			const int d = 0;
			int cstep = 1;
			if (length() > mat.length())
			{
				// Calculate steps for N dimensions
				for (int i = 1; i < mat.dm.N; i++)
				{
					cstep *= mat.dm[i];
				}
				calculateWidthstep(nds, mat.dm, d, cstep);

				// Get broadcasting indices
				getBroadcastIndices(nds, _dm, 0, didx, nds[0], 0, bci);

				// Create a result memory
				result.create(_dm);

				// Calculate a division matrix
				for (int i = 0; i < tlen; i++)
				{
					result.ddata[i] = ddata[i] / mat.ddata[bci[i]];
				}
			}
			else
			{
				// Calculate steps for N dimensions
				for (int i = 1; i < _dm.N; i++)
				{
					cstep *= _dm[i];
				}
				calculateWidthstep(nds, _dm, d, cstep);

				// Get broadcasting indices
				getBroadcastIndices(nds, mat.dm, 0, didx, nds[0], 0, bci);

				// Create a result memory
				result.create(mat.dm);

				// Calculate a division matrix
				for (int i = 0; i < mat.tlen; i++)
				{
					result.ddata[i] = ddata[bci[i]] / mat.ddata[i];
				}
			}
		}
		else
		{
			// Create a result memory
			result.create(_dm);

			// Calculate a division matrix
			for (int i = 0; i < tlen; i++)
			{
				result.ddata[i] = ddata[i] / mat.ddata[i];
			}
		}

		return result;
	}

	template<const unsigned int N> ndmatrix<N> ndmatrix<N>::div(const double val) const
	{
		// Check a status
		assert(empty() == false);
		assert(val != 0.0);

		// Calculate a division matrix
		ndmatrix result(_dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[i] / val;
		}

		return result;
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
