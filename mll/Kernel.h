#ifndef KERNEL_H
#define KERNEL_H

namespace mll
{
	// The kernel type for SVM
	typedef enum _kntype
	{
		SVM_KERNEL_UNKNOWN = -1,
		SVM_KERNEL_LINEAR,
		SVM_KERNEL_POLYNOMIAL,
		SVM_KERNEL_TANH,
		SVM_KERNEL_RBF,
	} kntype;

	// The kernel for the data trick
	class kernel : public nml::object
	{
		// Variables
	public:
		nml::prop::get<kntype> type;			// kernel type
		nml::prop::get<double> deg;			// degree for the polynomial kernel
		nml::prop::get<double> cons;			// constant Value for the polynomial kernel
		nml::prop::get<double> th1;			// threshold 1 for the hyperbolic tangent kernel
		nml::prop::get<double> th2;			// threshold 2 for the hyperbolic tangent kernel
		nml::prop::get<double> sig;			// sigma parameter for the RBF kernel

		// Functions
	public:
		const bool empty() const;			// Check the kernel is empty or not
		virtual void set(const double deg, const double cons, const double th1, const double th2, const double sigma) = 0;			// Set a kernel for the data trick
		virtual const nml::algmat trick(const nml::algmat& X, const nml::algmat& Xi) const = 0;			// Remap the feature vectors using the kernel data trick

		// Constructors & Destructor
	public:
		kernel();
		virtual ~kernel();

		// Variables
	protected:
		kntype _type;			// kernel type
		double _deg;			// degree for the polynomial kernel
		double _cons;			// constant Value for the polynomial kernel
		double _th1;			// threshold 1 for the hyperbolic tangent kernel
		double _th2;			// threshold 2 for the hyperbolic tangent kernel
		double _sig;			// sigma parameter for the RBF kernel

		// Functions
	protected:
		virtual void setObject();			// Set an object
		virtual void copyObject(const nml::object& obj);			// Copy the object
		void clearObject();			// Clear the object

	};

	// The linear kernel for SVM
	class linear_kernel : public kernel
	{
		// Variables
	public:


		// Functions
	public:		
		const nml::algmat trick(const nml::algmat& X, const nml::algmat& Xi) const;			// Remap the feature vectors using the kernel trick

		// Operators
	public:
		linear_kernel& operator=(const linear_kernel& obj);

		// Constructors & Destructor
	public:
		linear_kernel();
		linear_kernel(const linear_kernel& obj);
		~linear_kernel();

		// Variables
	private:


		// Functions
	private:		
		void setObject();			// Set an object		
		void set(const double deg, const double cons, const double th1, const double th2, const double sig);			// Set a kernel for the trick

	};

	// The polynomial kernel
	class polynomial_kernel : public kernel
	{
		// Variables
	public:


		// Functions
	public:
		// cons > 0
		void set(const double deg, const double cons = 1.0);			// Set a kernel for the trick
		const nml::algmat trick(const nml::algmat& X, const nml::algmat& Xi) const;			// Remap the feature vectors using the kernel trick

		// Operators
	public:
		polynomial_kernel& operator=(const polynomial_kernel& obj);

		// Constructors & Destructor
	public:
		polynomial_kernel();
		// cons > 0
		polynomial_kernel(const double deg, const double cons);
		polynomial_kernel(const polynomial_kernel& obj);
		~polynomial_kernel();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const nml::object& obj);			// Copy the object
		void set(const double deg, const double cons, const double th1, const double th2, const double sig);			// Set a kernel for the trick

	};

	// The hyperbolic tangent kernel
	class tanh_kernel : public kernel
	{
		// Variables
	public:


		// Functions
	public:
		// th1, th2 >= 0
		void set(const double th1, const double th2 = 0.0);			// Set a kernel for the trick
		const nml::algmat trick(const nml::algmat& X, const nml::algmat& Xi) const;			// Remap the feature vectors using the kernel trick

		// Operators
	public:
		tanh_kernel& operator=(const tanh_kernel& obj);

		// Constructors & Destructor
	public:
		tanh_kernel();
		// th1, th2 >= 0
		tanh_kernel(const double th1, const double th2 = 0.0);
		tanh_kernel(const tanh_kernel& obj);
		~tanh_kernel();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const nml::object& obj);			// Copy the object
		void set(const double deg, const double cons, const double th1, const double th2, const double sig);			// Set a kernel for the trick

	};

	// The RBF kernel
	class rbf_kernel : public kernel
	{
		// Variables
	public:


		// Functions
	public:
		// Sigma != 0
		void set(const double sig);			// Set a kernel for the trick
		const nml::algmat trick(const nml::algmat& X, const nml::algmat& Xi) const;			// Remap the feature vectors using the kernel trick

		// Operators
	public:
		rbf_kernel& operator=(const rbf_kernel& obj);

		// Constructors & Destructor
	public:
		rbf_kernel();
		// Sigma != 0
		rbf_kernel(const double sig);
		rbf_kernel(const rbf_kernel& obj);
		~rbf_kernel();

		// Variables
	private:


		// Functions
	private:
		void setObject();			// Set an object
		void copyObject(const nml::object& obj);			// Copy the object
		void set(const double deg, const double cons, const double th1, const double th2, const double sig);			// Set a kernel for the trick

	};
}

#endif