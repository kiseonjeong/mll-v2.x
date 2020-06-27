#ifndef REGRESSION_H
#define REGRESSION_H

namespace nml
{
	// The regression for numerical method
	class regression : public object
	{
		// Variables
	public:


		// Functions
	public:
		virtual void estimate(const algmat& X, const algmat& Y, const int deg) = 0;			// Estimate the coefficients
		virtual const double fit(const double x) const = 0;			// Interpolate the data
		virtual const algmat coeff() const;			// Get the estimated coefficients

		// Variables
	protected:
		algmat C;			// estimated coefficients
		int deg;			// degree of regression
		algmat X;			// backup vector X
		algmat Y;			// backup vector Y

		// Functions
	protected:
		void checkEmpty(const algmat& X, const algmat& Y) const;			// Check the matrices are empty or not
		void checkLength(const algmat& X, const algmat& Y) const;			// Check the length of the matrices
		void compareLength(const algmat& X, const algmat& Y) const;			// Compare the length between the matrices

	};

	// The polynomial fitting using ordinary least square
	class polyfit : public regression
	{
		// Variables
	public:


		// Functions
	public:
		void estimate(const algmat& X, const algmat& Y, const int deg);			// Estimate the coefficients
		const double fit(const double x) const;			// Interpolate the data

		// Operators
	public:
		polyfit& operator=(const polyfit& obj);

		// Constructors & Destructor
	public:
		polyfit();
		polyfit(const algmat& X, const algmat& Y, const int deg);
		polyfit(const polyfit& obj);
		~polyfit();

		// Variables
	private:
		double epsilon;			// epsilon parameter
		double meanX;			// mean of vector X
		double stdX;			// standard deviation of vector X
		double meanY;			// mean of vector Y
		double stdY;			// standard deviation of vector X

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const object& obj);			// Copy the object
		void clearObject();			// Clear the object
		void standardization();			// Do standardization on the vectors
		void calculate();			// Calculate the coefficients

	};
}

#endif