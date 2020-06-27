#include "stdafx.h"
#include "KNearestNeighbor.h"

namespace mll
{
	KNN::KNN()
	{
		// Set an object
		setObject();
	}

	KNN::KNN(const int K, const measure& meas)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(K, meas);
	}

	KNN::KNN(const mlldataset& dataset, const int K, const measure& meas)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset, K, meas);
	}

	KNN::KNN(const KNN& obj)
	{
		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);
	}

	KNN::~KNN()
	{
		// Clear the object
		clearObject();
	}

	KNN& KNN::operator=(const KNN& obj)
	{
		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void KNN::setObject()
	{
		// Set the parameters
		setType(*this);
		K = 1;

		// Set the memories
		meas = nullptr;
		minVec.release();
		maxVec.release();
		M.release();
		X.release();
		T.release();
		C.release();
	}

	void KNN::copyObject(const object& obj)
	{
		// Do down casting
		KNN* _obj = (KNN*)&obj;

		// Copy the parameters
		K = _obj->K;

		// Copy the memories
		copyMeasure(*_obj->meas);
		minVec = _obj->minVec;
		maxVec = _obj->maxVec;
		M = _obj->M;
		X = _obj->X;
		T = _obj->T;
		C = _obj->C;
	}

	void KNN::clearObject()
	{
		// Clear the memories
		if (meas != nullptr)
		{
			delete meas;
		}
		minVec.release();
		maxVec.release();
		M.release();
		X.release();
		T.release();
		C.release();
	}

}