namespace nml
{
	template<typename T> bool object::isEqualType(const T& var) const
	{
		// Compare the objects
		return typeid(*this) == typeid(T);
	}

	template<typename T> void object::setType(const T& var)
	{
		// Set a object type
		objType = typeid(var).name();
	}
}
