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

	template<typename var, const unsigned int N> void ndarray<var, N>::operator=(const var val)
	{
		// Set a value
		tdata[tidx] = val;

		// Initialize the temporary index
		tidx = 0;
	}

	template<typename var, const unsigned int N> ndarray<var, N>::operator var&() 
	{
		// Get a value
		var& val = tdata[tidx];

		// Initialize the temporary index
		tidx = 0;

		return val;
	}

	template<typename var, const unsigned int N> ndarray<var, N - 1>& ndarray<var, N>::operator[](const int idx) const
	{
		// Check the index
		int addr = idx * step;
		assert(idx >= 0 && idx < _dm.length());

		// Set a 1 dimensional index
		ndarray<var, N - 1>& _sub = (ndarray<var, N - 1>&)sub;
		_sub.subdata(&ddata[addr], idx);

		return _sub;
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::setObject()
	{
		// Initialize the parameters
		didx = 0;
		dlen = 0;
		step = 1;
		tidx = 0;
		tlen = 0;

		// Initialize the memories
		ddata = nullptr;
		tdata = nullptr;
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::copyObject(const ndarray<var, N>& obj)
	{
		// Check a dimension status
		if (obj._dm.N == N)
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
		else
		{
			// Copy the value
			tdata[tidx] = obj.tdata[obj.tidx];
		}
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::clearObject()
	{
		// Check a dimension status
		if (_dm.N == N)
		{
			// Clear the data memory
			if (ddata != nullptr)
			{
				free(ddata);
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

	template<typename var, const unsigned int N> void ndarray<var, N>::create(const dim& di)
	{
		// Check a dimension status
		if (di.N == N)
		{
			// Check the dimension information
			assert(di.d.size() > 0);

			// Copy the dimension information
			_dm = di;

			// Calculate a data length and a width step
			if (ddata != nullptr)
			{
				free(ddata);
				ddata = nullptr;
			}
			ddata = (var*)malloc(sizeof(var) * di.length());
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

			// Set sub dimensional data
			sub.subdim(di, tdata);
		}
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::release()
	{
		// Clear the object
		clearObject();
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::set(const var val)
	{
		// Check the dimension information
		assert(_dm.d.size() > 0);

		// Set a value on the data
		for (int i = 0; i < tlen; i++)
		{
			ddata[i] = val;
		}
	}

	template<typename var, const unsigned int N> var* ndarray<var, N>::ptr() const
	{
		// Check a dimension information
		assert(_dm.d.size() > 0);

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
		for (int i = 0; i < _dm[didx]; i++)
		{
			if (i == 0)
			{
				std::cout << "[";
			}
			(*this)[i]._cout(nspace);
			if (i == _dm[didx] - 1)
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

	template<typename var, const unsigned int N> void ndarray<var, N>::subdim(const dim& di, var* tdata)
	{
		// Copy the dimension information
		_dm = di;

		// Calculate a data length and a width step
		didx = _dm.N - N;
		dlen = _dm[didx];
		step = 1;
		for (int i = didx + 1; i < _dm.N; i++)
		{
			step *= _dm[i];
		}
		this->tdata = tdata;
		tidx = 0;
		tlen = dlen * step;

		// Set sub dimensional data
		sub.subdim(di, tdata);
	}

	template<typename var, const unsigned int N> void ndarray<var, N>::subdata(var* ddata, const int tidx)
	{
		// Set the dimensional data
		this->ddata = ddata;

		// Set a total index
		this->tidx = tidx;
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

	template<typename var> void ndarray<var, 1>::operator=(const var val)
	{
		// Set a value
		tdata[tidx] = val;

		// Initialize the temporary index
		tidx = 0;
	}

	template<typename var> ndarray<var, 1>::operator var&()
	{
		// Get a value
		var& val = tdata[tidx];

		// Initialize the temporary index
		tidx = 0;

		return val;
	}

	template<typename var> var& ndarray<var, 1>::operator[](const int idx) const
	{
		// Check the index
		assert(idx >= 0 && idx < _dm.length());

		// Get a value
		return ddata[idx];
	}

	template<typename var> void ndarray<var, 1>::setObject()
	{
		// Initialize the parameters
		didx = 0;
		dlen = 0;
		step = 1;
		tidx = 0;
		tlen = 0;

		// Initialize the memories
		ddata = nullptr;
		tdata = nullptr;
	}

	template<typename var> void ndarray<var, 1>::copyObject(const ndarray<var, 1>& obj)
	{
		// Check a dimension status
		if (obj._dm.N == 1)
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
		else
		{
			// Copy the value
			tdata[tidx] = obj.tdata[obj.tidx];
		}
	}

	template<typename var> void ndarray<var, 1>::clearObject()
	{
		// Clear the data memory
		if (_dm.N == 1)
		{
			// Clear the data memory
			if (ddata != nullptr)
			{
				free(ddata);
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

	template<typename var> void ndarray<var, 1>::create(const dim& di)
	{
		// Check the dimension information
		if (di.N == 1)
		{
			// Check the dimension information
			assert(di.d.size() > 0);

			// Copy the dimension information
			_dm = di;

			// Calculate a data length and a width step
			if (ddata != nullptr)
			{
				free(ddata);
				ddata = nullptr;
			}
			ddata = (var*)malloc(sizeof(var) * di.length());
			didx = 0;
			dlen = _dm[didx];
			step = 1;
			tdata = ddata;
			tidx = 0;
			tlen = dlen * step;
		}
	}

	template<typename var> void ndarray<var, 1>::release()
	{
		// Clear the object
		clearObject();
	}

	template<typename var> void ndarray<var, 1>::set(const var val)
	{
		// Check the dimension information
		assert(_dm.d.size() > 0);

		// Set a value on the data
		for (int i = 0; i < tlen; i++)
		{
			ddata[i] = val;
		}
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
		for (int i = 0; i < _dm[didx]; i++)
		{
			if (i == 0)
			{
				std::cout << "[";
			}
			std::cout << ddata[i];
			if (i == _dm[didx] - 1)
			{
				std::cout << "]";
			}
			else
			{
				std::cout << ", ";
			}
		}
	}

	template<typename var> void ndarray<var, 1>::subdim(const dim& di, var* tdata)
	{
		// Copy the dimension information
		_dm = di;

		// Calculate a data length and a width step
		didx = _dm.N - 1;
		dlen = _dm[didx];
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
}