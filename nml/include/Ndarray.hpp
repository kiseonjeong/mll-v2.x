namespace nml
{
	template<typename var, const unsigned int N> ndarray<var, N>::ndarray() : dm(_dm)
	{
		// Set an object
		setObject();
	}

	template<typename var, const unsigned int N> ndarray<var, N>::ndarray(const dim& di) : dm(_dm)
	{
		// Set an object
		setObject();

		// Check a dimension value
		assert(di.N == N);

		// Create an array
		create(di);
	}

	template<typename var, const unsigned int N> ndarray<var, N>::ndarray(const dim& di, const var& val) : dm(_dm)
	{
		// Set an object
		setObject();

		// Check a dimension value
		assert(di.N == N);

		// Create an array
		create(di);

		// Set the array
		set(val);
	}

	template<typename var, const unsigned int N> ndarray<var, N>::ndarray(const ndarray<var, N>& obj) : dm(_dm)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	template<typename var, const unsigned int N> ndarray<var, N>::~ndarray()
	{
		// Clear the object
		clearObject();
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator=(const ndarray<var, N>& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::operator=(const var& val)
	{
		// Set the array
		set(val);
	}

	template<typename var, const unsigned int N> const ndarray<var, N - 1>& ndarray<var, N>::operator[](const int idx) const
	{
		// Check an address
		int addr = idx * step;
		assert(idx >= 0 && idx < dlen);

		// Set a 1 dimensional index
		subarr.subdata(&ddata[addr], idx);

		// Get a sub-dimensional array
		return subarr;
	}

	template<typename var, const unsigned int N> ndarray<var, N - 1>& ndarray<var, N>::operator[](const int idx)
	{
		// Check an address
		int addr = idx * step;
		assert(idx >= 0 && idx < dlen);

		// Set a 1 dimensional index
		subarr.subdata(&ddata[addr], idx);

		// Get a sub-dimensional array
		return subarr;
	}

	template<typename var, const unsigned int N> const var& ndarray<var, N>::operator()(const int idx) const
	{
		// Check an index
		assert(idx >= 0 && idx < tlen);

		// Get a value
		return ddata[idx];
	}

	template<typename var, const unsigned int N> var& ndarray<var, N>::operator()(const int idx)
	{
		// Check an index
		assert(idx >= 0 && idx < tlen);

		// Get a value
		return ddata[idx];
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator+(const ndarray<var, N>& mat) const
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate an addition matrix
			return add(mat);
		}
		else
		{
			return ndarray<var, N>();
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator+(const double val) const
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate an addition matrix
			return add(val);
		}
		else
		{
			return ndarray<var, N>();
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator+=(const ndarray<var, N>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate an addition matrix
			*this = add(mat);

			return *this;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator+=(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate an addition matrix
			*this = add(val);

			return *this;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator-(const ndarray<var, N>& mat) const
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a subtraction matrix
			return sub(mat);
		}
		else
		{
			return ndarray<var, N>();
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator-(const double val) const
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a subtraction matrix
			return sub(val);
		}
		else
		{
			return ndarray<var, N>();
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator-=(const ndarray<var, N>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a subtraction matrix
			*this = sub(mat);

			return *this;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator-=(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a subtraction matrix
			*this = sub(val);

			return *this;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator*(const ndarray<var, N>& mat) const
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a multiplication matrix
			return mul(mat);
		}
		else
		{
			return ndarray<var, N>();
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator*(const double val) const
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a multiplication matrix
			return mul(val);
		}
		else
		{
			return ndarray<var, N>();
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator*=(const ndarray<var, N>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a multiplication matrix
			*this = mul(mat);

			return *this;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator*=(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a multiplication matrix
			*this = mul(val);

			return *this;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator/(const ndarray<var, N>& mat) const
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a division matrix
			return div(mat);
		}
		else
		{
			return ndarray<var, N>();
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator/(const double val) const
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a division matrix
			return div(val);
		}
		else
		{
			return ndarray<var, N>();
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator/=(const ndarray<var, N>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a division matrix
			*this = div(mat);

			return *this;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N>& ndarray<var, N>::operator/=(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			checkArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Calculate a division matrix
			*this = div(val);

			return *this;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> inline void ndarray<var, N>::setObject()
	{
		// Initialize the parameters
		didx = 0;
		dlen = 0;
		step = 1;
		tidx = 0;
		tlen = 0;
		nflag = false;
		cflag = true;

		// Initialize the memories
		ddata = nullptr;
		tdata = nullptr;
	}

	template<typename var, const unsigned int N> inline void ndarray<var, N>::copyObject(const ndarray<var, N>& obj)
	{
		// Check a creation flag
		bool creation = false;
		if (_dm.N == obj._dm.N)
		{
			for (int i = 0; i < _dm.N; i++)
			{
				if (_dm[i] != obj._dm[i])
				{
					creation = true;
					break;
				}
			}
		}
		else
		{
			creation = true;
		}

		// Check the data memory
		if (obj.ddata != nullptr)
		{
			// Check a creation flag
			if (creation == true)
			{
				// Create a new data memory
				create(obj._dm);
			}

			// Copy the data memory
			for (int i = 0; i < tlen; i++)
			{
				ddata[i] = obj.ddata[i];
			}
		}
	}

	template<typename var, const unsigned int N> inline void ndarray<var, N>::clearObject()
	{
		// Check a dimension status
		if (didx == 0)
		{
			// Clear the data memory
			if (ddata != nullptr)
			{
				delete[] ddata;
			}

			// Initialize the parameters
			didx = 0;
			dlen = 0;
			step = 1;
			tidx = 0;
			tlen = 0;

			// Initialize the memories
			ddata = nullptr;
			tdata = nullptr;
			_dm.release();
		}
	}

	template<typename var, const unsigned int N> inline void ndarray<var, N>::create(const dim& di)
	{
		// Check the dimension information
		assert(di.d.size > 0);

		// Copy the dimension information
		_dm = di;

		// Calculate a data length and a width step
		if (ddata != nullptr)
		{
			delete[] ddata;
			ddata = nullptr;
		}
		ddata = new var[di.length()];
		didx = _dm.N - N;
		dlen = _dm[didx];
		step = 1;
		for (int i = didx + 1; i < _dm.N; i++)
		{
			step *= _dm[i];
		}
		tdata = ddata;
		tidx = 0;
		tlen = dlen * step;

		// Set sub-dimension information
		dim sdi;
		sdi.create(N - 1);
		for (int i = didx + 1, j = 0; i < _dm.N; i++, j++)
		{
			sdi.set(j, _dm[i]);
		}

		// Set sub dimensional data
		subarr.subdim(sdi, tdata, didx);
	}

	template<typename var, const unsigned int N> inline void ndarray<var, N>::create(const dim& di, const var& val)
	{
		// Create an array
		create(di);

		// Set the array
		set(val);
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::release()
	{
		// Clear the object
		clearObject();
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::set(const var& val)
	{
		// Check the dimension information
		assert(_dm.d.size > 0);

		// Set a value on the data
		for (int i = 0; i < tlen; i++)
		{
			ddata[i] = val;
		}
	}

	template<typename var, const unsigned int N> inline const int ndarray<var, N>::length() const
	{
		// Get a total length
		return tlen;
	}

	template<typename var, const unsigned int N> var* ndarray<var, N>::ptr() const
	{
		// Check a dimension information
		assert(_dm.d.size > 0);

		// Get a data pointer
		return ddata;
	}

	template<typename var, const unsigned int N> bool ndarray<var, N>::empty() const
	{
		// Check the data
		if (ddata == nullptr)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::cout() const
	{
		// Console out begin of array
		for (int i = 0; i < didx; i++)
		{
			std::cout << " ";
		}
		std::cout << "(";

		// Check a dimension information
		if (empty() == true)
		{
			// Console out the empty array
			std::cout << "[]";
		}
		else
		{
			// Console out the array
			_cout(1);
		}

		// Console out end of array
		std::cout << ")" << endl << endl;
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::_cout(const int nspace) const
	{
		// Console out the array
		for (int i = 0; i < _dm[0]; i++)
		{
			if (i == 0)
			{
				std::cout << "[";
			}
			(*this)[i]._cout(nspace);
			if (i == _dm[0] - 1)
			{
				std::cout << "]";
			}
			else
			{
				std::cout << ", ";
				for (int j = 0; j < N - 1; j++)
				{
					std::cout << endl;
				}
				for (int j = 0; j <= didx + nspace; j++)
				{
					std::cout << " ";
				}
			}
		}
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::subdim(const dim& di, var* tdata, const int didx)
	{
		// Copy the dimension information
		_dm = di;

		// Calculate a data length and a width step
		this->didx = didx + 1;
		dlen = _dm[0];
		step = 1;
		for (int i = 1; i < _dm.N; i++)
		{
			step *= _dm[i];
		}
		this->tdata = tdata;
		tidx = 0;
		tlen = dlen * step;

		// Set sub-dimension information
		dim sdi;
		sdi.create(N - 1);
		for (int i = 1, j = 0; i < _dm.N; i++, j++)
		{
			sdi.set(j, _dm[i]);
		}

		// Set sub dimensional data
		subarr.subdim(sdi, tdata, this->didx);
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::subdata(var* ddata, const int tidx)
	{
		// Set the dimensional data
		this->ddata = ddata;

		// Set a total index
		this->tidx = tidx;
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::checkArrayType() const
	{
		// Check an array type
		string varName = typeid(tdata[0]).name();
		if (varName == "unsigned char" || varName == "char" ||
			varName == "unsigned short" || varName == "short" ||
			varName == "int" || varName == "unsigned int" ||
			varName == "float" || varName == "double")
		{
			nflag = true;
		}
		else
		{
			nflag = false;
		}
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::calculateWidthstep(ndarray<int, 1>& nds, const dim& dm, const int d, const int cstep) const
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

	template<typename var, const unsigned int N> void ndarray<var, N>::getTransIndices(const ndarray<int, 1>& nds, const dim& dm, const dim& trd, const int d, const int cstep, int addr, std::vector<int>& indices) const
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

	template<typename var, const unsigned int N> void ndarray<var, N>::getBroadcastIndices(const ndarray<int, 1>& nds, const dim& dm, const int d, const int s, const int cstep, int addr, std::vector<int>& indices) const
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

	template<typename var, const unsigned int N> template <const unsigned int M> ndarray<var, M> ndarray<var, N>::reshape(const dim& di) const
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
		ndarray<var, M> result(rdm);
		for (int i = 0; i < tlen; i++)
		{
			result.ptr()[i] = ddata[i];
		}

		return result;
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::transpose(const dim& trd) const
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
		ndarray<var, N> result(dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[tri[i]];
		}

		return result;
	}

	template<typename var, const unsigned int N> inline ndarray<var, N> ndarray<var, N>::add(const ndarray<var, N>& mat) const
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
		ndarray<double, N> result;
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

	template<typename var, const unsigned int N> inline ndarray<var, N> ndarray<var, N>::add(const double val) const
	{
		// Check a status
		assert(empty() == false);

		// Calculate an addition matrix
		ndarray<double, N> result(_dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[i] + val;
		}

		return result;
	}

	template<typename var, const unsigned int N> inline ndarray<var, N> ndarray<var, N>::sub(const ndarray<var, N>& mat) const
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
		ndarray<double, N> result;
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

	template<typename var, const unsigned int N> inline ndarray<var, N> ndarray<var, N>::sub(const double val) const
	{
		// Check a status
		assert(empty() == false);

		// Calculate a subtraction matrix
		ndarray<double, N> result(_dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[i] - val;
		}

		return result;
	}

	template<typename var, const unsigned int N> inline ndarray<var, N> ndarray<var, N>::mul(const ndarray<var, N>& mat) const
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
		ndarray<double, N> result;
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

	template<typename var, const unsigned int N> inline ndarray<var, N> ndarray<var, N>::mul(const double val) const
	{
		// Check a status
		assert(empty() == false);

		// Calculate a multiplication matrix
		ndarray<double, N> result(_dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[i] * val;
		}

		return result;
	}

	template<typename var, const unsigned int N> inline ndarray<var, N> ndarray<var, N>::div(const ndarray<var, N>& mat) const
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
		ndarray<double, N> result(_dm);
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

	template<typename var, const unsigned int N> inline ndarray<var, N> ndarray<var, N>::div(const double val) const
	{
		// Check a status
		assert(empty() == false);
		assert(val != 0.0);

		// Calculate a division matrix
		ndarray<double, N> result(_dm);
		for (int i = 0; i < tlen; i++)
		{
			result.ddata[i] = ddata[i] / val;
		}

		return result;
	}
}