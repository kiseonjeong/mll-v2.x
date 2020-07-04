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
		ndarray<var, N - 1>& _sub = const_cast<ndarray<var, N - 1>&>(sub);
		_sub.subdata(&ddata[addr], idx);

		// Get a sub-dimensional array
		return sub;
	}

	template<typename var, const unsigned int N> ndarray<var, N - 1>& ndarray<var, N>::operator[](const int idx)
	{
		// Check an address
		int addr = idx * step;
		assert(idx >= 0 && idx < dlen);

		// Set a 1 dimensional index
		sub.subdata(&ddata[addr], idx);

		// Get a sub-dimensional array
		return sub;
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

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator+(const ndarray<var, N>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false && mat.empty() == false);
			for (int i = 0; i < _dm.N; i++)
			{
				assert(_dm[i] == mat.dm[i]);
			}

			// Calculate an addition matrix
			ndarray<double, N> result(_dm);
			for (int i = 0; i < mat.tlen; i++)
			{
				result.ddata[i] = ddata[i] + mat.ddata[i];
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator+(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
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
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator-(const ndarray<var, N>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false && mat.empty() == false);
			for (int i = 0; i < _dm.N; i++)
			{
				assert(_dm[i] == mat.dm[i]);
			}

			// Calculate a subtraction matrix
			ndarray<double, N> result(_dm);
			for (int i = 0; i < mat.tlen; i++)
			{
				result.ddata[i] = ddata[i] - mat.ddata[i];
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator-(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
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
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator*(const ndarray<var, N>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false && mat.empty() == false);
			for (int i = 0; i < _dm.N; i++)
			{
				assert(_dm[i] == mat.dm[i]);
			}

			// Calculate a multiplication matrix
			ndarray<double, N> result(_dm);
			for (int i = 0; i < mat.tlen; i++)
			{
				result.ddata[i] = ddata[i] * mat.ddata[i];
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator*(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
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
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator/(const ndarray<var, N>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false && mat.empty() == false);
			for (int i = 0; i < _dm.N; i++)
			{
				assert(_dm[i] == mat.dm[i]);
			}
			for (int i = 0; i < mat.tlen; i++)
			{
				assert(mat(i) != 0.0);
			}

			// Calculate a division matrix
			ndarray<double, N> result(_dm);
			for (int i = 0; i < mat.tlen; i++)
			{
				result.ddata[i] = ddata[i] / mat.ddata[i];
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var, const unsigned int N> ndarray<var, N> ndarray<var, N>::operator/(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
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
			nflag = false;

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
		sub.subdim(sdi, tdata, didx);
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
		sub.subdim(sdi, tdata, this->didx);
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::subdata(var* ddata, const int tidx)
	{
		// Set the dimensional data
		this->ddata = ddata;

		// Set a total index
		this->tidx = tidx;
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::setArrayType()
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

	template<typename var> ndarray<var, 1>::ndarray() : dm(_dm)
	{
		// Set an object
		setObject();
	}

	template<typename var> ndarray<var, 1>::ndarray(const dim& di) : dm(_dm)
	{
		// Set an object
		setObject();

		// Check a dimension value
		assert(di.N == 1);

		// Create an array
		create(di);
	}

	template<typename var> ndarray<var, 1>::ndarray(const dim& di, const var& val) : dm(_dm)
	{
		// Set an object
		setObject();

		// Check a dimension value
		assert(di.N == 1);

		// Create an array
		create(di);

		// Set the array
		set(val);
	}

	template<typename var> ndarray<var, 1>::ndarray(const ndarray<var, 1>& obj) : dm(_dm)
	{
		// Copy the object
		copyObject(obj);
	}

	template<typename var> ndarray<var, 1>::~ndarray()
	{
		// Clear the object
		clearObject();
	}

	template<typename var> ndarray<var, 1>& ndarray<var, 1>::operator=(const ndarray<var, 1>& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	template<typename var> void ndarray<var, 1>::operator=(const var& val)
	{
		// Set the array
		set(val);
	}

	template<typename var> const var& ndarray<var, 1>::operator[](const int idx) const
	{
		// Check the index
		assert(idx >= 0 && idx < tlen);

		// Get a value
		return ddata[idx];
	}

	template<typename var> var& ndarray<var, 1>::operator[](const int idx)
	{
		// Check the index
		assert(idx >= 0 && idx < tlen);

		// Get a value
		return ddata[idx];
	}

	template<typename var> const var& ndarray<var, 1>::operator()(const int idx) const
	{
		// Check an index
		assert(idx >= 0 && idx < tlen);

		// Get a value
		return ddata[idx];
	}

	template<typename var> var& ndarray<var, 1>::operator()(const int idx)
	{
		// Check an index
		assert(idx >= 0 && idx < tlen);

		// Get a value
		return ddata[idx];
	}

	template<typename var> ndarray<var, 1> ndarray<var, 1>::operator+(const ndarray<var, 1>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false && mat.empty() == false);
			for (int i = 0; i < _dm.N; i++)
			{
				assert(_dm[i] == mat.dm[i]);
			}

			// Calculate an addition matrix
			ndarray<double, 1> result(_dm);
			for (int i = 0; i < mat.tlen; i++)
			{
				result.ddata[i] = ddata[i] + mat.ddata[i];
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var> ndarray<var, 1> ndarray<var, 1>::operator+(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false);

			// Calculate an addition matrix
			ndarray<double, 1> result(_dm);
			for (int i = 0; i < tlen; i++)
			{
				result.ddata[i] = ddata[i] + val;
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var> ndarray<var, 1> ndarray<var, 1>::operator-(const ndarray<var, 1>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false && mat.empty() == false);
			for (int i = 0; i < _dm.N; i++)
			{
				assert(_dm[i] == mat.dm[i]);
			}

			// Calculate a subtraction matrix
			ndarray<double, 1> result(_dm);
			for (int i = 0; i < mat.tlen; i++)
			{
				result.ddata[i] = ddata[i] - mat.ddata[i];
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var> ndarray<var, 1> ndarray<var, 1>::operator-(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false);

			// Calculate a subtraction matrix
			ndarray<double, 1> result(_dm);
			for (int i = 0; i < tlen; i++)
			{
				result.ddata[i] = ddata[i] - val;
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var> ndarray<var, 1> ndarray<var, 1>::operator*(const ndarray<var, 1>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false && mat.empty() == false);
			for (int i = 0; i < _dm.N; i++)
			{
				assert(_dm[i] == mat.dm[i]);
			}

			// Calculate a multiplication matrix
			ndarray<double, 1> result(_dm);
			for (int i = 0; i < mat.tlen; i++)
			{
				result.ddata[i] = ddata[i] * mat.ddata[i];
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var> ndarray<var, 1> ndarray<var, 1>::operator*(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false);

			// Calculate a multiplication matrix
			ndarray<double, 1> result(_dm);
			for (int i = 0; i < tlen; i++)
			{
				result.ddata[i] = ddata[i] * val;
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var> ndarray<var, 1> ndarray<var, 1>::operator/(const ndarray<var, 1>& mat)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false && mat.empty() == false);
			for (int i = 0; i < _dm.N; i++)
			{
				assert(_dm[i] == mat.dm[i]);
			}
			for (int i = 0; i < mat.tlen; i++)
			{
				assert(mat(i) != 0.0);
			}

			// Calculate a division matrix
			ndarray<double, 1> result(_dm);
			for (int i = 0; i < mat.tlen; i++)
			{
				result.ddata[i] = ddata[i] / mat.ddata[i];
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var> ndarray<var, 1> ndarray<var, 1>::operator/(const double val)
	{
		// Check the check flag
		if (cflag == true)
		{
			setArrayType();
			cflag = false;
		}

		// Check the type flag
		if (nflag == true)
		{
			// Check a status
			assert(empty() == false);
			assert(val != 0.0);

			// Calculate an addition matrix
			ndarray<var, 1> result(_dm);
			for (int i = 0; i < tlen; i++)
			{
				result.ddata[i] = ddata[i] / val;
			}

			return result;
		}
		else
		{
			return *this;
		}
	}

	template<typename var> inline void ndarray<var, 1>::setObject()
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

	template<typename var> inline void ndarray<var, 1>::copyObject(const ndarray<var, 1>& obj)
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

	template<typename var> inline void ndarray<var, 1>::clearObject()
	{
		// Clear the data memory
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
			nflag = false;

			// Initialize the memories
			ddata = nullptr;
			tdata = nullptr;
			_dm.release();
		}
	}

	template<typename var> inline void ndarray<var, 1>::create(const dim& di)
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
		didx = 0;
		dlen = _dm[didx];
		step = 1;
		tdata = ddata;
		tidx = 0;
		tlen = dlen * step;
	}

	template<typename var> inline void ndarray<var, 1>::create(const dim& di, const var& val)
	{
		// Create an array
		create(di);

		// Set the array
		set(val);
	}

	template<typename var> void ndarray<var, 1>::release()
	{
		// Clear the object
		clearObject();
	}

	template<typename var> void ndarray<var, 1>::set(const var& val)
	{
		// Check the dimension information
		assert(_dm.d.size > 0);

		// Set a value on the data
		for (int i = 0; i < tlen; i++)
		{
			ddata[i] = val;
		}
	}

	template<typename var> inline const int ndarray<var, 1>::length() const
	{
		// Get a total length
		return tlen;
	}

	template<typename var> var* ndarray<var, 1>::ptr() const
	{
		// Check the dimension information
		assert(_dm.d.size() > 0);

		// Get a data pointer
		return ddata;
	}

	template<typename var> bool ndarray<var, 1>::empty() const
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

	template<typename var> void ndarray<var, 1>::cout() const
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

	template<typename var> void ndarray<var, 1>::_cout(const int nspace) const
	{
		// Console out the array
		for (int i = 0; i < _dm[0]; i++)
		{
			if (i == 0)
			{
				std::cout << "[";
			}
			std::cout << ddata[i];
			if (i == _dm[0] - 1)
			{
				std::cout << "]";
			}
			else
			{
				std::cout << ", ";
			}
		}
	}

	template<typename var> void ndarray<var, 1>::subdim(const dim& di, var* tdata, const int didx)
	{
		// Copy the dimension information
		_dm = di;

		// Calculate a data length and a width step
		this->didx = didx + 1;
		dlen = _dm[0];
		step = 1;
		this->tdata = tdata;
		tidx = 0;
		tlen = dlen * step;
	}

	template<typename var> void ndarray<var, 1>::subdata(var* ddata, const int tidx)
	{
		// Set the dimensional data
		this->ddata = ddata;

		// Set a total index
		this->tidx = tidx;
	}

	template<typename var> void ndarray<var, 1>::setArrayType()
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
}