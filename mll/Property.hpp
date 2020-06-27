namespace nml
{
	namespace prop
	{
		// The Property for reading
		template<typename var> get<var>::get(var& value) : _value(value)
		{

		}

		template<typename var> get<var>::operator const var&() const 
		{
			return _value; 
		}

		template<typename var> const var& get<var>::val() const
		{
			return _value;
		}

		// The Property for writing
		template<typename var> set<var>::set(var& value) : _value(value) 
		{

		}

		template<typename var> var& set<var>::operator=(const var& value) 
		{ 
			_value = value; 
			
			return _value;
		}

		template<typename var> void set<var>::val(const var& value)
		{
			_value = value;
		}

		// The Property for reading and writing
		template<typename var> all<var>::all(var& value) : _value(value)
		{

		}

		template<typename var> all<var>::operator const var&() const 
		{
			return _value; 
		}

		template<typename var> var& all<var>::operator=(const var& value) 
		{ 
			_value = value; 

			return _value;
		}

		template<typename var> const var& all<var>::val() const
		{
			return _value;
		}

		template<typename var> void all<var>::val(const var& value)
		{
			_value = value;
		}
	}
}