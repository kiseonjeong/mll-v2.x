#ifndef SPLINE_H
#define SPLINE_H

namespace nml
{
	// The spline for numerical method
	class spline : public object
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
		algmat I;			// interval matrix
		algmat X;			// backup vector X
		algmat Y;			// backup vector Y
		int numpt;			// number of points
		int numit;			// number of intervals

		// Functions
	protected:
		void checkEmpty(const algmat& X, const algmat& Y) const;			// Check the matrices are empty or not
		void checkLength(const algmat& X, const algmat& Y) const;			// Check the length of the matrices
		void compareLength(const algmat& X, const algmat& Y) const;			// Compare the length between the matrices
		const int getInterval(const double x) const;			// Get an interval index

	};

	// The linear spline
	class spline1d : public spline
	{
		// Variables
	public:


		// Functions
	public:
		virtual void estimate(const algmat& X, const algmat& Y);			// Estimate the coefficients
		virtual const double interpolate(const double x) const;			// Interpolate the data

		// Operators
	public:
		spline1d& operator=(const spline1d& obj);

		// Constructors & Destructor
	public:
		spline1d();
		spline1d(const algmat& X, const algmat& Y);
		spline1d(const spline1d& obj);
		virtual ~spline1d();

		// Variables
	protected:


		// Functions
	protected:
		virtual void setObject();			// Set an object
		virtual void copyObject(const object& obj);			// Copy the object
		virtual void clearObject();			// Clear the object
		virtual void calculate();			// Calculate the coefficients

		// Variables
	private:


		// Functions
	private:
		const double slope(const double x0, const double y0, const double x1, const double y1) const;			// Calculate the slope

	};

	// The Quadratic Spline
	class spline2d : public spline1d
	{
		// Variables
	public:


		// Functions
	public:
		void estimate(const algmat& X, const algmat& Y);			// Estimate the coefficients
		const double interpolate(const double x) const;			// Interpolate the data

		// Operators
	public:
		spline2d& operator=(const spline2d& obj);

		// Constructors & Destructor
	public:
		spline2d();
		spline2d(const algmat& X, const algmat& Y);
		spline2d(const spline2d& obj);
		~spline2d();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object
		void calculate();			// Calculate the coefficients

	};

	// The end condition type of the cubic spline
	typedef enum _cubic_type
	{
		CUBIC_UNKNOWN = 0,
		CUBIC_NATURAL,
		CUBIC_CLAMPED,
		CUBIC_NOTAKNOT,
	} cubic_type;

	// The cubic spline parameters
	class cubic_params : public object
	{
		// Variables
	public:		
		prop::get<cubic_type> type;			// end condition type

		// Functions
	public:


		// Constructors & Destructor
	public:
		cubic_params();
		~cubic_params();

		// Variables
	protected:
		cubic_type _type;			// end condition type

		// Functions
	protected:
		virtual void setObject();			// Set an object
		virtual void copyObject(const object& obj);			// Copy the object
		virtual void clearObject();			// Clear the object

	};

	// The natural type
	class natural : public cubic_params
	{
		// Variables
	public:


		// Functions
	public:


		// Operators
	public:
		natural& operator=(const natural& obj);

		// Constructors & Destructor
	public:
		natural();
		natural(const natural& obj);
		~natural();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object

	};

	// The clamped type
	class clamped : public cubic_params
	{
		// Variables
	public:


		// Functions
	public:


		// Operators
	public:
		clamped& operator=(const clamped& obj);

		// Constructors & Destructor
	public:
		clamped();
		clamped(const clamped& obj);
		~clamped();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object

	};

	// The not a knot type
	class notaknot : public cubic_params
	{
		// Variables
	public:
		prop::get<double> alpha;			// condition parameter 'alpha'
		prop::get<double> beta;			// condition parameter 'beta'

		// Functions
	public:
		
		void set(const double alpha, const double beta);			// Set the parameters

		// Operators
	public:
		notaknot& operator=(const notaknot& obj);

		// Constructors & Destructor
	public:
		notaknot();
		notaknot(const double alpha, const double beta);
		notaknot(const notaknot& obj);
		~notaknot();

		// Variables
	private:
		double _alpha;			// condition parameter 'alpha'
		double _beta;			// condition parameter 'beta'

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const object& obj);			// Copy the object
		void clearObject();			// Clear the object

	};

	// The cubic spline
	class spline3d : public spline1d
	{
		// Variables
	public:


		// Functions
	public:
		void condition(const cubic_params& params);			// Set an end condition for the cubic spline
		void estimate(const algmat& X, const algmat& Y);			// Estimate the coefficients
		const double interpolate(const double x) const;			// Interpolate the data

		// Operators
	public:
		spline3d& operator=(const spline3d& obj);

		// Constructors & Destructor
	public:
		spline3d();
		spline3d(const cubic_params& params);
		spline3d(const algmat& X, const algmat& Y, const cubic_params& params = natural());
		spline3d(const spline3d& obj);
		~spline3d();

		// Variables
	private:
		cubic_params* params;			// cubic spline parameters

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const object& obj);			// Copy the object
		void clearObject();			// Clear the object
		void createParams(const cubic_params& params);			// Create the parameters
		void calculate();			// Calculate the coefficients
		const algmat condmat(const algmat& W) const;			// Get a condition matrix
		const algmat resmat(const algmat& S) const;			// Get a result matrix

	};

	// rename the spliners
	typedef spline1d linear_spline;
	typedef spline2d quadratic_spline;
	typedef spline3d cubic_spline;

	// The hermite spline type
	typedef enum _hermite_type
	{
		HERMITE_UNKNOWN = 0,
		HERMITE_FINITE_DIFFERENCE,
		HERMITE_CARDINAL,
		HERMITE_CATMULL_ROM,
	} hermite_type;

	// The hermite spline parameters
	class hermite_params : public object
	{
		// Variables
	public:
		prop::get<hermite_type> type;			// hermite spline type

		// Functions
	public:
		virtual const algmat calculate(const algmat& X, const algmat& Y, const int numpt) const = 0;			// Calculate the coefficients

		// Constructors & Destructor
	public:
		hermite_params();
		~hermite_params();

		// Variables
	protected:
		
		hermite_type _type;			// hermite spline type

		// Functions
	protected:
		virtual void setObject();			// Set an object
		virtual void copyObject(const object& obj);			// Copy the object
		virtual void clearObject();			// Clear the object

	};

	// The finite difference type
	class finitediff : public hermite_params
	{
		// Variables
	public:


		// Functions
	public:
		virtual const algmat calculate(const algmat& X, const algmat& Y, const int numpt) const;			// Calculate the coefficents

		// Operators
	public:
		finitediff& operator=(const finitediff& obj);

		// Constructors & Destructor
	public:
		finitediff();
		finitediff(const finitediff& obj);
		virtual ~finitediff();

		// Variables
	protected:


		// Functions
	protected:
		virtual void setObject();			// Set an object
		const double slope(const double x0, const double y0, const double x1, const double y1) const;			// Calculate the slope

	};

	// The cardinal type
	class cardinal : public finitediff
	{
		// Variables
	public:
		prop::get<double> c;			// tension parameter

		// Functions
	public:
		void set(double c);			// Set the parameters
		virtual const algmat calculate(const algmat& X, const algmat& Y, const int numpt) const;			// Calculate the coefficents

		// Operators
	public:
		cardinal& operator=(const cardinal& obj);

		// Constructors & Destructor
	public:
		cardinal();
		cardinal(double c);
		cardinal(const cardinal& obj);
		virtual ~cardinal();

		// Variables
	protected:
		double _c;			// tension parameter

		// Functions
	protected:
		virtual void setObject();			// Set an object
		void copyObject(const object& obj);			// Copy the object

	};

	// The catmull-rom type
	class catmullrom : public cardinal
	{
		// Variables
	public:


		// Functions
	public:
		const algmat calculate(const algmat& X, const algmat& Y, const int numpt) const;			// Calculate the coefficents

		// Operators
	public:
		catmullrom& operator=(const catmullrom& obj);

		// Constructors & Destructor
	public:
		catmullrom();
		catmullrom(const catmullrom& obj);
		~catmullrom();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object

	};

	// The Hermite Spline
	class hermite : public spline
	{
		// Variables
	public:


		// Functions
	public:
		void condition(const hermite_params& params);			// Set a spline condition for the hermite spline
		void estimate(const algmat& X, const algmat& Y);			// Estimate the coefficients
		const double interpolate(const double x) const;			// Interpolate the data

		// Operators
	public:
		hermite& operator=(const hermite& obj);

		// Constructors & Destructor
	public:
		hermite();
		hermite(const hermite_params& params);
		hermite(const algmat& X, const algmat& Y, const hermite_params& params = finitediff());
		hermite(const hermite& obj);
		~hermite();

		// Variables
	private:
		hermite_params* params;			// hermite spline parameters

		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const object& obj);			// Copy the object
		void clearObject();			// Clear the object
		void createParams(const hermite_params& params);			// Create the hermite parameters
		const double basis(const double x, const int idx, const int mode) const;			// Calculate a basis
		const double tangent(const double x, const int idx) const;			// Calculate a tangent

	};

	// The akima spline
	class akima : public spline
	{
		// Variables
	public:


		// Functions
	public:
		void estimate(const algmat& X, const algmat& Y);			// Estimate the coefficients
		const double interpolate(const double x) const;			// Interpolate the data

		// Operators
	public:
		akima& operator=(const akima& obj);

		// Constructors & Destructor
	public:
		akima();
		akima(const algmat& X, const algmat& Y);
		akima(const akima& obj);
		~akima();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const object& obj);			// Copy the object
		void clearObject();			// Clear the object
		void calculate();			// Calculate the coefficients
		const algmat augment(const algmat& S) const;			// Augment the slopes on the end points
		const double derivative(const algmat& S, const int idx) const;			// Calculate a derivative value

	};
}

#endif