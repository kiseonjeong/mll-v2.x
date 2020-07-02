#ifndef INTERPOLATION_H
#define INTERPOLATION_H

namespace nml
{
	// The Interpolation for Numerical Method
	class interpolation : public object
	{
		// Variables
	public:


		// Functions
	public:
		virtual void estimate(const algmat& X, const algmat& Y) = 0;			// Estimate the coefficients
		virtual const double interpolate(const double x) const = 0;			// Interpolate the data
		virtual const algmat coeff() const;			// Get the estimated coefficients

		// Variables
	protected:
		algmat C;			// estimated coefficients
		int deg;			// degree of interpolation
		algmat X;			// backup vector X
		algmat Y;			// backup vector Y

		// Functions
	protected:
		void checkEmpty(const algmat& X, const algmat& Y) const;			// Check the matrices are empty or not
		void checkLength(const algmat& X, const algmat& Y) const;			// Check the length of the matrices
		void compareLength(const algmat& X, const algmat& Y) const;			// Compare the length between the matrices

	};

	// The Vandermonde Interpolation
	class vandermonde : public interpolation
	{
		// Variables
	public:


		// Functions
	public:
		void estimate(const algmat& X, const algmat& Y);			// Estimate the coefficients
		const double interpolate(const double x) const;			// Interpolate the data

		// Operators
	public:
		vandermonde& operator=(const vandermonde& obj);

		// Constructors & Destructor
	public:
		vandermonde();
		vandermonde(const algmat& X, const algmat& Y);
		vandermonde(const vandermonde& obj);
		~vandermonde();

		// Variables
	private:


		// Functions
	private:
		inline void setObject();			// Set an object
		inline void copyObject(const object& obj);			// Copy the object
		inline void clearObject();			// Clear the object
		void calculate();			// Calculate the coefficients

	};

	// The Newton Interpolation
	class newton : public interpolation
	{
		// Variables
	public:


		// Functions
	public:
		void estimate(const algmat& X, const algmat& Y);			// Estimate the coefficients
		const double interpolate(const double x) const;			// Interpolate the data

		// Operators
	public:
		newton& operator=(const newton& obj);

		// Constructors & Destructor
	public:
		newton();
		newton(const algmat& X, const algmat& Y);
		newton(const newton& obj);
		~newton();

		// Variables
	private:
		int idx;			// index of coefficients

		// Functions
	private:
		inline void setObject();			// Set an object
		inline void copyObject(const object& obj);			// Copy the object
		inline void clearObject();			// Clear the object
		void calculate(const algmat& X, const algmat& Y);			// Calculate the coefficients
		const double slope(const double x0, const double y0, const double x1, const double y1) const;			// Calculate the slope

	};

	// Lagrange Interpolation
	class lagrange : public interpolation
	{
		// Variables
	public:


		// Functions
	public:
		void estimate(const algmat& X, const algmat& Y);			// Estimate Coefficients
		const double interpolate(const double x) const;			// Interpolate Data

		// Operators
	public:
		lagrange& operator=(const lagrange& obj);

		// Constructors & Destructor
	public:
		lagrange();
		lagrange(const algmat& X, const algmat& Y);
		lagrange(const lagrange& obj);
		~lagrange();

		// Variables
	private:


		// Functions
	private:
		inline void setObject();			// Set an object
		inline void copyObject(const object& obj);			// Copy the object
		inline void clearObject();			// Clear the object
		const double calculate(const double x, const int idx) const;			// Calculate the coefficients
		const algmat coeff() const;			// Get the estimated coefficients (Prohibit this access)

	};
}

#endif