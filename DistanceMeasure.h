#ifndef DISTANCE_MEASURE_H
#define DISTANCE_MEASURE_H

namespace mll
{
	// The measurement type
	typedef enum _meastype
	{
		KNN_MEASURE_UNKNOWN = -1,
		KNN_MEASURE_EUCLIDEAN,
		KNN_MEASURE_MANHATTAN,
		KNN_MEASURE_CHEBYCHEV,
		KNN_MEASURE_MINKOWSKI,
		KNN_MEASURE_COSINE,
		KNN_MEASURE_CORRELATION,
	} meastype;

	// The measurement for distance
	class measure : public nml::object
	{
		// Variables
	public:
		nml::prop::get<meastype> type;			// measure type
		nml::prop::get<bool> nflag;			// normalization flag
		nml::prop::get<double> slope;			// normalization slope
		nml::prop::get<double> intercept;			// normalization intercept
		nml::prop::get<double> p;			// power parameter for distance

		// Functions
	public:
		virtual void set(const bool nflag, const double slope, const double intercept, const double p) = 0;			// Set measurement parameters
		virtual const double calculate(const nml::algmat& xi, const nml::algmat& xj) const = 0;			// Calculate distance

		// Constructors & Destructor
	public:
		measure();
		virtual ~measure();

		// Variables
	protected:
		meastype _type;			// measure type
		bool _nflag;			// normalization flag
		double _slope;			// normalization slope
		double _intercept;			// normalization intercept
		double _p;			// power parameter for a distance

		// Functions
	protected:
		virtual void setObject();			// Set an object
		void copyObject(const nml::object& obj);			// Copy the object

	};

	// The euclidean measurement
	class euclidean : public measure
	{
		// Variables
	public:


		// Functions
	public:
		void set(const bool nflag, const double slope, const double intercept);			// Set measurement parameters
		virtual const double calculate(const nml::algmat& xi, const nml::algmat& xj) const;			// Calculate distance

		// Operators
	public:
		euclidean& operator=(const euclidean& obj);

		// Constructors & Destructor
	public:
		euclidean();
		euclidean(const bool nflag, const double slope, const double intercept);
		euclidean(const euclidean& obj);
		virtual ~euclidean();

		// Variables
	protected:


		// Functions
	protected:
		virtual void setObject();			// Set an object
		virtual void set(const bool nflag, const double slope, const double intercept, const double p);			// Set measurement parameters

	};

	// The manhattan measurement
	class manhattan : public euclidean
	{
		// Operators
	public:
		manhattan& operator=(const manhattan& obj);

		// Constructors & Destructor
	public:
		manhattan();
		manhattan(const bool nflag, const double slope, const double intercept);
		manhattan(const manhattan& obj);
		~manhattan();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object

	};

	// The minkowski measurement
	class minkowski : public euclidean
	{
		// Variables
	public:


		// Functions
	public:
		void set(const bool nflag, const double slope, const double intercept, const double p);			// Set measurement parameters

		// Operators
	public:
		minkowski& operator=(const minkowski& obj);

		// Constructors & Destructor
	public:
		minkowski();
		minkowski(const bool nflag, const double slope, const double intercept, const double p);
		minkowski(const minkowski& obj);
		~minkowski();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object

	};

	// The chebychev measurement
	class chebychev : public euclidean
	{
		// Variables
	public:


		// Functions
	public:
		const double calculate(const nml::algmat& xi, const nml::algmat& xj) const;			// Calculate distance

		// Operators
	public:
		chebychev& operator=(const chebychev& obj);

		// Constructors & Destructor
	public:
		chebychev();
		chebychev(const bool nflag, const double slope, const double intercept);
		chebychev(const chebychev& obj);
		~chebychev();

		// Variables
	private:


		// Functions
	private:		
		void setObject();			// Set an object

	};

	// The cosine measurement
	class cosine : public measure
	{
		// Variables
	public:


		// Functions
	public:
		const double calculate(const nml::algmat& xi, const nml::algmat& xj) const;			// Calculate distance

		// Operators
	public:
		cosine& operator=(const cosine& obj);

		// Constructors & Destructor
	public:
		cosine();
		cosine(const cosine& obj);
		virtual ~cosine();

		// Variables
	protected:


		// Functions
	protected:
		virtual void setObject();			// Set an object
		void set(const bool norm, const double gain, const double offset, const double p);			// Set measurement parameters

	};

	// The correlation measurement
	class correlation : public cosine
	{
		// Operators
	public:
		correlation& operator=(const correlation& obj);

		// Constructors & Destructor
	public:
		correlation();
		correlation(const correlation& obj);
		~correlation();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};
}

#endif