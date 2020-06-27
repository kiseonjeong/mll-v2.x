#ifndef PROPERTY_H
#define PROPERTY_H

namespace nml
{
	namespace prop
	{
		// The Property for reading
		template<typename var> class get
		{
		public:
			const var& val() const;

		public:
			operator const var&() const;

		public:
			get(var& value);

		private:
			var& _value;

		};

		// The Property for writing
		template<typename var> class set
		{
		public:
			void val(const var& value);

		public:
			var& operator=(const var& value);

		public:
			set(var& value);

		private:
			var& _value;

		};

		// The Property for reading and writing
		template<typename var> class all
		{
		public:
			const var& val() const;
			void val(const var& value);

		public:
			operator const var&() const;
			var& operator=(const var& value);

		public:
			all(var& value);

		private:
			var& _value;

		};
	}
}

#include "Property.hpp"

#endif