namespace nml
{
	template<typename var> numem<var>::numem() : size(_size)
	{
		// Set an object
		setObject();
	}

	template<typename var> numem<var>::numem(const int size) : size(_size)
	{
		// Set an object
		setObject();

		// Create the data
		create(size);
	}

	template<typename var> numem<var>::numem(const int size, const var& val) : size(_size)
	{
		// Set an object
		setObject();

		// Create the data
		create(size, val);
	}

	template<typename var> numem<var>::numem(const numem& obj) : size(_size)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	template<typename var> numem<var>::~numem()
	{
		// Clear the object
		clearObject();
	}

	template<typename var> numem<var>& numem<var>::operator=(const numem& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	template<typename var> const var& numem<var>::operator[](const int idx) const
	{
		// Check the index
		assert(idx >= 0 && idx < _size);

		return _ptr[idx];
	}

	template<typename var> var& numem<var>::operator[](const int idx)
	{
		// Check the index
		assert(idx >= 0 && idx < _size);

		return _ptr[idx];
	}

	template<typename var> void numem<var>::setObject()
	{
		// Set the parameters
		_size = -1;

		// Set the memories
		_ptr = nullptr;
	}

	template<typename var> void numem<var>::copyObject(const numem<var>& obj)
	{
		// Check the object status
		if (obj.empty() == false)
		{
			// Copy the parameters
			overwrite(obj._size);

			// Copy the memories
			for (int i = 0; i < obj._size; i++)
			{
				_ptr[i] = obj._ptr[i];
			}
		}
	}

	template<typename var> void numem<var>::clearObject()
	{
		// Clear the parameters
		_size = -1;

		// Clear the memories
		if (empty() == false)
		{
			delete[] _ptr;
			_ptr = nullptr;
		}
	}

	template<typename var> void numem<var>::overwrite(int size)
	{
		// Check the size information
		assert(size >= 0);

		// Release the old data
		release();

		// Create an object
		_size = size;
		_ptr = new var[size];
	}

	template<typename var> var* numem<var>::ptr() const
	{
		// Get the data pointer
		return _ptr;
	}

	template<typename var> bool numem<var>::empty() const
	{
		// Check the data pointer
		if (_ptr == nullptr)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	template<typename var> void numem<var>::release()
	{
		// Clear the object
		clearObject();
	}

	template<typename var> void numem<var>::create(const int size)
	{
		// Overwrite the data
		overwrite(size);
	}

	template<typename var> void numem<var>::create(const int size, var val)
	{
		// Overwrite the data
		overwrite(size);

		// Initialize the data
		for (int i = 0; i < size; i++)
		{
			_ptr[i] = val;
		}
	}

	template<typename var> void numem<var>::copy(const numem& obj)
	{
		// Copy the data
		copyObject(obj);
	}

	template<typename var> void numem<var>::copy(const int size, const var*& data)
	{
		// Overwrite the data
		overwrite(size);

		// Initialize the data
		for (int i = 0; i < size; i++)
		{
			_ptr[i] = data[i];
		}
	}
}