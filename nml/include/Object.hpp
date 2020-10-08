namespace nml
{
	template<typename T> inline bool object::isEqualType(const T& var) const
	{
		// Compare the objects
		return typeid(*this) == typeid(T);
	}

	template<typename T> inline void object::setType(const T& var)
	{
		// Set a object type
		objType = typeid(var).name();
	}
}
