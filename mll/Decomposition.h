#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

namespace nml
{
	// The decomposition for algebra
	class decomposition : public object
	{
		// Variables
	public:


		// Functions
	public:
		virtual const std::vector<algmat> decomp(const algmat& mat) = 0;			// Decompose the matrix
		const std::vector<algmat> result() const;			// Get the decomposition result

		// Operators
	public:
		virtual const algmat& operator[](const int idx) const;			// Get the decomposition result

		// Variables
	protected:
		std::vector<algmat> D;			// decomposition result

		// Functions
	protected:
		void checkEmpty(const algmat& mat) const;			// Check the matrix is empty or not
		void checkSquareness(const algmat& mat) const;			// Check the matrix is square or not

	};

	// The pivot mode for the LUD
	typedef enum _pivotMode
	{
		LU_PARTIAL_PIVOT = 0,
		LU_FULL_PIVOT,
	} pivotMode;

	// The LU decomposition
	class LUD : public decomposition
	{
		// Variables
	public:


		// Functions
	public:
		void pivot(const pivotMode mode);			// Set a pivot mode
		const std::vector<algmat> decomp(const algmat& mat);			// Decompose the matrix

		// Operators
	public:
		LUD& operator=(const LUD& obj);

		// Constructors & Destructor
	public:
		LUD();
		LUD(const algmat& mat);
		LUD(const algmat& mat, const pivotMode mode);
		LUD(const LUD& obj);
		~LUD();

		// Variables
	private:
		pivotMode mode;			// pivot mode flag
		std::vector<algmat> P;			// factorization matrix P
		std::vector<algmat> M;			// factorization matrix M

		// Functions
	private:
		inline void setObject();			// Set an object
		inline void copyObject(const object& obj);			// Copy the object
		inline void clearObject();			// Clear the object
		void factorize(const algmat& mat);			// Factorize the matrix

	};

	// The QR decomposition
	class QRD : public decomposition
	{
		// Variables
	public:


		// Functions
	public:
		const std::vector<algmat> decomp(const algmat& mat);			// Decompose the matrix

		// Operators
	public:
		QRD& operator=(const QRD& obj);

		// Constructors & Destructor
	public:
		QRD();
		QRD(const algmat& mat);
		QRD(const QRD& obj);
		~QRD();

		// Variables
	private:
		std::vector<algmat> T;			// transformation matrix

		// Functions
	private:
		inline void setObject();			// Set an object
		inline void copyObject(const object& obj);			// Copy the object
		inline void clearObject();			// Clear the object
		const algmat reduction(const algmat& mat, const int flag) const;			// Reduce the matrix
		const algmat houseHolder(const algmat& mat) const;			// Convert the matrix using House Holder Transform
		const algmat reflection(const algmat& mat, const int length) const;			// Convert the matrix using House Holder Transform
		const double vecsize(const algmat& mat) const;			// Calculate a vector length
		const algmat extension(const algmat& mat, const int offset, const int length) const;			// Extend a matrix

	};

	// The EV decomposition
	class EVD : public decomposition
	{
		// Variables
	public:


		// Functions
	public:
		virtual const std::vector<algmat> decomp(const algmat& mat);			// Decompose the matrix

		// Operators
	public:
		EVD& operator=(const EVD& obj);

		// Constructors & Destructor
	public:
		EVD();
		EVD(const algmat& mat);
		EVD(const EVD& obj);
		virtual ~EVD();

		// Variables
	protected:


		// Functions
	protected:
		virtual inline void setObject();			// Set an object
		inline void copyObject(const object& obj);			// Copy the object
		inline void clearObject();			// Clear the object
		const bool isConvergence(const algmat& curr, const algmat& prev) const;			// Check the matrix is convergence or not

		// Variables
	private:


		// Functions
	private:

	};

	// The SV decomposition
	class SVD : public EVD
	{
		// Variables
	public:


		// Functions
	public:
		const std::vector<algmat> decomp(const algmat& mat);			// Decompose the matrix
		static algmat pinv(const algmat& mat);			// Get a pseudo inverse matrix

		// Operators
	public:
		SVD& operator=(const SVD& obj);
		const algmat& operator[](const int idx) const;

		// Constructors & Destructor
	public:
		SVD();
		SVD(const algmat& mat);
		SVD(const SVD& obj);
		~SVD();

		// Variables
	private:


		// Functions
	private:
		inline void setObject();			// Set an object

	};
}

#endif