#include "stdafx.h"
#include "DistanceMeasure.h"

namespace mll
{
	measure::measure() : type(_type), nflag(_nflag), slope(_slope), intercept(_intercept), p(_p)
	{
		// Set an object
		setObject();
	}

	measure::~measure()
	{
		// Do nothing
	}

	void measure::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_UNKNOWN;
		_nflag = false;
		_slope = 1.0;
		_intercept = 0.0;
		_p = 1.0;
	}

	void measure::copyObject(const object& obj)
	{
		// Do down casting
		measure* _obj = (measure*)&obj;

		// Copy the parameters
		_type = _obj->_type;
		_nflag = _obj->_nflag;
		_slope = _obj->_slope;
		_intercept = _obj->_intercept;
		_p = _obj->_p;
	}

	euclidean::euclidean()
	{
		// Set an object
		setObject();
	}

	euclidean::euclidean(const bool nflag, const double slope, const double intercept)
	{
		// Set an object
		setObject();

		// Set measurement parameters
		set(nflag, slope, intercept);
	}

	euclidean::euclidean(const euclidean& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	euclidean::~euclidean()
	{
		// Do nothing
	}

	euclidean& euclidean::operator=(const euclidean& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	inline void euclidean::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_EUCLIDEAN;
		_nflag = false;
		_slope = 1.0;
		_intercept = 0.0;
		_p = 2.0;
	}

	void euclidean::set(const bool nflag, const double slope, const double intercept)
	{
		// Set the parameters
		_nflag = nflag;
		_slope = slope;
		_intercept = intercept;
	}

	void euclidean::set(const bool nflag, const double slope, const double intercept, const double p)
	{
		// Set the parameters
		set(nflag, slope, intercept);
	}

	const double euclidean::calculate(const algmat& xi, const algmat& xj) const
	{
		// Calculate distance between the input dataset
		return pow(algmat::sum(algmat::pow(algmat::abs(xi - xj), p))(0), 1.0 / p);
	}

	manhattan::manhattan()
	{
		// Set an object
		setObject();
	}

	manhattan::manhattan(const bool nflag, const double slope, const double intercept)
	{
		// Set an object
		setObject();

		// Set measurement parameters
		set(nflag, slope, intercept);
	}

	manhattan::manhattan(const manhattan& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	manhattan::~manhattan()
	{
		// Do nothing
	}

	manhattan& manhattan::operator=(const manhattan& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	inline void manhattan::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_MANHATTAN;
		_nflag = false;
		_slope = 1.0;
		_intercept = 0.0;
		_p = 1.0;
	}

	minkowski::minkowski()
	{
		// Set an object
		setObject();
	}

	minkowski::minkowski(const bool nflag, const double slope, const double intercept, const double p)
	{
		// Set an object
		setObject();

		// Set measurement parameters
		set(nflag, slope, intercept, p);
	}

	minkowski::minkowski(const minkowski& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	minkowski::~minkowski()
	{
		// Do nothing
	}

	minkowski& minkowski::operator=(const minkowski& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	inline void minkowski::setObject()
	{
		// Set measurement parameters
		setType(*this);
		_type = KNN_MEASURE_MINKOWSKI;
		_nflag = false;
		_slope = 1.0;
		_intercept = 0.0;
		_p = 1.0;
	}

	void minkowski::set(const bool nflag, const double slope, const double intercept, const double p)
	{
		// Set measurement parameters
		_nflag = nflag;
		_slope = slope;
		_intercept = intercept;
		_p = p;
	}

	chebychev::chebychev()
	{
		// Set an object
		setObject();
	}

	chebychev::chebychev(const bool norm, const double gain, const double offset)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(norm, gain, offset);
	}

	chebychev::chebychev(const chebychev& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	chebychev::~chebychev()
	{
		// Do nothing
	}

	chebychev& chebychev::operator=(const chebychev& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	inline void chebychev::setObject()
	{
		// Set measurement parameters
		setType(*this);
		_type = KNN_MEASURE_CHEBYCHEV;
		_nflag = false;
		_slope = 1.0;
		_intercept = 0.0;
	}

	const double chebychev::calculate(const algmat& xi, const algmat& xj) const
	{
		// Calculate a distance between the input dataset
		return algmat::max(algmat::abs(xi - xj))(0);
	}

	cosine::cosine()
	{
		// Set an object
		setObject();
	}

	cosine::cosine(const cosine& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	cosine::~cosine()
	{
		// Do nothing
	}

	cosine& cosine::operator=(const cosine& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	inline void cosine::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = KNN_MEASURE_COSINE;
		_nflag = false;
		_slope = 1.0;
		_intercept = 0.0;
	}

	void cosine::set(const bool norm, const double gain, const double offset, const double p)
	{
		// Do nothing
	}

	const double cosine::calculate(const algmat& xi, const algmat& xj) const
	{
		// Calculate a distance between the input dataset
		return 1.0 - (xi.dot(xj.t())(0) / sqrt(xi.dot(xi.t())(0) * xj.dot(xj.t())(0)));
	}

	correlation::correlation()
	{
		// Set an object
		setObject();
	}

	correlation::correlation(const correlation& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	correlation::~correlation()
	{
		// Do nothing
	}

	correlation& correlation::operator=(const correlation& obj)
	{
		// Copy the object
		copyObject(obj);

		return *this;
	}

	inline void correlation::setObject()
	{
		// Set measurement parameters
		setType(*this);
		_type = KNN_MEASURE_CORRELATION;
		_nflag = false;
		_slope = 1.0;
		_intercept = 0.0;
	}
}
