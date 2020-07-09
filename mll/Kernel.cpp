#include "stdafx.h"
#include "Kernel.h"

namespace mll
{
	kernel::kernel() : type(_type), deg(_deg), cons(_cons), th1(_th1), th2(_th2), sig(_sig)
	{
		// Set an object
		setObject();
	}

	kernel::~kernel()
	{
		// Do nothing
	}

	void kernel::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = SVM_KERNEL_UNKNOWN;
	}

	void kernel::copyObject(const object& obj)
	{
		// Do down casting
		const kernel* _obj = static_cast<const kernel*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
	}

	void kernel::clearObject()
	{
		// Do nothing
	}

	const bool kernel::empty() const
	{
		// Check the kernel type
		if (_type == SVM_KERNEL_UNKNOWN)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	linear_kernel::linear_kernel()
	{
		// Set an object
		setObject();
	}

	linear_kernel::linear_kernel(const linear_kernel& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	linear_kernel::~linear_kernel()
	{
		// Clear the object
		clearObject();
	}

	linear_kernel& linear_kernel::operator=(const linear_kernel& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	void linear_kernel::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = SVM_KERNEL_LINEAR;
	}

	void linear_kernel::set(const double deg, const double cons, const double th1, const double th2, const double sig)
	{
		// Set the kernel parameters
		// In case of the linear kernel, do nothing
	}

	const algmat linear_kernel::trick(const algmat& X, const algmat& Xi) const
	{
		// Do the trick using the linear kernel
		algmat Ki(msize(X.rows, 1));
		for (int i = 0; i < X.rows; i++)
		{
			Ki(i) = X.submat(i).dot(Xi.t())(0);
		}

		return Ki;
	}

	polynomial_kernel::polynomial_kernel()
	{
		// Set an object
		setObject();
	}

	polynomial_kernel::polynomial_kernel(const double deg, const double cons)
	{
		// Set an object
		setObject();

		// Set a kernel
		set(deg, cons);
	}

	polynomial_kernel::polynomial_kernel(const polynomial_kernel& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	polynomial_kernel::~polynomial_kernel()
	{
		// Clear the object
		clearObject();
	}

	polynomial_kernel& polynomial_kernel::operator=(const polynomial_kernel& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	void polynomial_kernel::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = SVM_KERNEL_POLYNOMIAL;
		_deg = 0.0;
		_cons = 1.0;
	}

	void polynomial_kernel::copyObject(const object& obj)
	{
		// Do down casting
		const polynomial_kernel* _obj = static_cast<const polynomial_kernel*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_deg = _obj->deg;
		_cons = _obj->cons;
	}

	void polynomial_kernel::set(const double deg, const double cons)
	{
		// Set the kernel parameters
		_deg = deg;
		_cons = cons;
	}

	void polynomial_kernel::set(const double deg, const double cons, const double th1, const double th2, const double sig)
	{
		// Set the kernel parameters
		set(deg, cons);
	}

	const algmat polynomial_kernel::trick(const algmat& X, const algmat& Xi) const
	{
		// Do the trick using the polynomial kernel
		algmat Ki(msize(X.rows, 1));
		for (int i = 0; i < X.rows; i++)
		{
			Ki(i) = pow(X.submat(i).dot(Xi.t())(0) + cons, deg);
		}

		return Ki;
	}

	tanh_kernel::tanh_kernel()
	{
		// Set an object
		setObject();
	}

	tanh_kernel::tanh_kernel(const double th1, const double th2)
	{
		// Set an object
		setObject();

		// Set a kernel
		set(th1, th2);
	}

	tanh_kernel::tanh_kernel(const tanh_kernel& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	tanh_kernel::~tanh_kernel()
	{
		// Clear the object
		clearObject();
	}

	tanh_kernel& tanh_kernel::operator=(const tanh_kernel& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	void tanh_kernel::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = SVM_KERNEL_TANH;
		_th1 = 0.0;
		_th2 = 1.0;
	}

	void tanh_kernel::copyObject(const object& obj)
	{
		// Do down casting
		const tanh_kernel* _obj = static_cast<const tanh_kernel*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_th1 = _obj->_th1;
		_th2 = _obj->_th2;
	}

	void tanh_kernel::set(const double th1, const double th2)
	{
		// Set the kernel parameters
		_th1 = th1;
		_th2 = th2;
	}

	void tanh_kernel::set(const double deg, const double cons, const double th1, const double th2, const double sig)
	{
		// Set the kernel parameters
		set(th1, th2);
	}

	const algmat tanh_kernel::trick(const algmat& X, const algmat& Xi) const
	{
		// Do the trick using the hyperbolic tangent kernel
		algmat Ki(msize(X.rows, 1));
		for (int i = 0; i < X.rows; i++)
		{
			Ki(i) = std::tanh(th1 * X.submat(i).dot(Xi.t())(0) + th2);
		}

		return Ki;
	}

	rbf_kernel::rbf_kernel()
	{
		// Set an object
		setObject();
	}

	rbf_kernel::rbf_kernel(const double sig)
	{
		// Set an object
		setObject();

		// Set a kernel
		set(sig);
	}

	rbf_kernel::rbf_kernel(const rbf_kernel& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	rbf_kernel::~rbf_kernel()
	{
		// Clear the object
		clearObject();
	}

	rbf_kernel& rbf_kernel::operator=(const rbf_kernel& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	void rbf_kernel::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = SVM_KERNEL_RBF;
		_sig = 0.0;
	}

	void rbf_kernel::copyObject(const object& obj)
	{
		// Do down casting
		const rbf_kernel* _obj = static_cast<const rbf_kernel*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_sig = _obj->_sig;
	}

	void rbf_kernel::set(const double sig)
	{
		// Set the kernel parameters
		_sig = sig;
	}

	void rbf_kernel::set(const double deg, const double cons, const double th1, const double th2, const double sig)
	{
		// Set the kernel parameters
		set(sig);
	}

	const algmat rbf_kernel::trick(const algmat& X, const algmat& Xi) const
	{
		// Do the trick using the radial basis kernel
		algmat Ki(msize(X.rows, 1));
		for (int i = 0; i < X.rows; i++)
		{
			algmat delta = X.submat(i) - Xi;
			Ki(i) = delta.dot(delta.t())(0);
			Ki(i) = exp(-Ki(i) / (2.0 * pow(sig, 2.0)));
		}

		return Ki;
	}
}