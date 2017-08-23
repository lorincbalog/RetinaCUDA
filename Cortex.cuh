#ifndef CORTEX__CUH
#define CORTEX__CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "SamplingPoint.cuh"

typedef unsigned short ushort;
typedef unsigned int  uint;
typedef unsigned char uchar;
typedef int error;

class Cortex {
	enum ERRORS {
		invalidArguments = -1,
		uninitialized = 1,
		cortexSizeDidNotMatch,
		imageParametersDidNotMatch
	};

public:
	Cortex() : _rgb(false), _channels(1),
				_leftCortexSize(0), _rightCortexSize(0), _alpha(nanf("")),
				d_leftLoc(nullptr), d_rightLoc(nullptr), d_leftFields(nullptr), d_rightFields(nullptr),
				_shrink(nanf("")), _cortImgSize(make_uint2(0,0)),
				_gaussKernelWidth(0), _gaussSigma(nanf("")), d_gauss(nullptr) {}
	Cortex(const Cortex&) = delete;
	~Cortex();

	__host__ error locationsFromCortexFields(SamplingPoint *h_leftFields, size_t leftSize,
										SamplingPoint *h_rightFields, size_t rightSize);

	__host__ error cortImageLeft(double *h_imageVector, size_t vecLen, uchar *h_result,
					size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = nullptr) const;
	__host__ error cortImageRight(double *h_imageVector, size_t vecLen, uchar *h_result,
					size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = nullptr) const;

	__host__ float getAlpha() const { return _alpha; }
	__host__ void setAlpha(float alpha);

	__host__ float getShrink() const { return _shrink; }
	__host__ void setShrink(float shrink);

	__host__ bool getRGB() const { return _rgb; }
	__host__ void setRGB(bool rgb);

	__host__ uint2 getCortImageSize() const { return _cortImgSize; }
	__host__ void setCortImageSize(uint2 cortImgSize);

	__host__ size_t getLeftSize() { return _leftCortexSize; }
	__host__ error getLeftCortexFields(SamplingPoint *h_leftFields, size_t leftSize) const;
	__host__ error setLeftCortexFields(const SamplingPoint *h_leftFields, size_t leftSize);

	__host__ size_t getRightSize() { return _rightCortexSize; }
	__host__ error getRightCortexFields(SamplingPoint *h_rightFields, size_t rightSize) const;
	__host__ error setRightCortexFields(const SamplingPoint *h_rightFields, size_t rightSize);

	__host__ error getLeftCortexLocations(double2 *h_leftLoc, size_t leftSize) const;
	__host__ error setLeftCortexLocations(const double2 *h_leftLoc, size_t leftSize);

	__host__ error getRightCortexLocations(double2 *h_leftLoc, size_t leftSize) const;
	__host__ error setRightCortexLocations(const double2 *h_leftLoc, size_t rightSize);

	__host__ size_t getGaussKernelWidth() const { return _gaussKernelWidth; }
	__host__ float getGaussSigma() { return _gaussSigma; }

	__host__ error getGauss100(double *h_gauss, size_t kernelWidth, float sigma) const;
	__host__ error setGauss100(const size_t kernelWidth, const float sigma, double *h_gauss = nullptr);

private:
	__host__ bool isReady() const;
	__host__ void gauss100();
	__host__ error cortImage(double *h_imageVector, size_t vecLen, uchar *h_result,
			size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector,
			SamplingPoint *d_fields, double2 *d_loc, size_t size) const;
	template <class T>
	__host__ error getFromDevice(T *h_fields, const size_t h_size, const T *d_fields, const size_t d_size) const;
	template <class T>
	__host__ error setOnDevice(const T *h_fields, const size_t h_size, T **d_fields, size_t &d_size);

	bool _rgb;
	ushort _channels;

	size_t _leftCortexSize;
	size_t _rightCortexSize;
	SamplingPoint *d_leftFields;
	SamplingPoint *d_rightFields;
	double2 *d_leftLoc;
	double2 *d_rightLoc;

	float _alpha;
	float _shrink;
	uint2 _cortImgSize;

	size_t _gaussKernelWidth;
	float _gaussSigma;
	double *d_gauss;
};

#endif //CORTEX__CUH
