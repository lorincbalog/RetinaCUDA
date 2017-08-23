#ifndef RETINA__CUH
#define RETINA__CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "SamplingPoint.cuh"

typedef unsigned short  ushort;
typedef unsigned int  uint;
typedef unsigned char uchar;

class Retina {
	enum ERRORS {
		invalidArguments = -1,
		uninitialized = 1,
		retinaSizeDidNotMatch,
		imageParametersDidNotMatch
	};

public:
	Retina() : _rgb(false), _channels(1), _imageW(0), _imageH(0),
				_centerX(0), _centerY(0), d_gauss(nullptr),
				_retinaSize(0), d_points(nullptr), _d_imageVector(nullptr) {}
	Retina(const Retina&) = delete;
	~Retina();

	__host__ int getSamplingFields(SamplingPoint *h_points, size_t retinaSize);
	__host__ int setSamplingFields(SamplingPoint *h_points, size_t retinaSize);

	__host__ int getGaussNormImage(double *h_gauss, size_t gaussH, size_t gaussW, size_t gaussC) const;
	__host__ int setGaussNormImage(const double *h_gauss = nullptr, size_t gaussH = 0,
								   size_t gaussW = 0, size_t gaussC = 0);

	__host__ int sample(const uchar *h_imageIn, size_t imageH, size_t imageW, size_t imageC,
						double *h_imageVector, size_t vectorLength,
						bool keepImageVectorOnDevice = false);
	__host__ int inverse(const double *h_imageVector,  size_t vectorLength,
						 uchar *h_imageInverse, size_t imageH, size_t imageW, size_t imageC,
						 bool useImageVectorOnDevice = false) const;

	__host__ int getRetinaSize() const { return _retinaSize; }

	__host__ int getImageHeight() const { return _imageH; }
	__host__ void setImageHeight(const int imageH);

	__host__ int getImageWidth() const { return _imageW; }
	__host__ void setImageWidth(const int imageW);

	__host__ bool getRGB() const { return _rgb; }
	__host__ void setRGB(const bool rgb);

	__host__ int getCenterX() const { return _centerX; }
	__host__ void setCenterX(const int centerX);

	__host__ int getCenterY() const { return _centerY; }
	__host__ void setCenterY(const int centerY);

	__host__ double* imageVectorOnDevice(size_t &vectorLength);

private:
	__host__ bool isReady() const;
	__host__ bool validateImageSize(size_t imageH, size_t imageW, size_t imageC) const;
	__host__ int removeSamplingPointsFromDevice();

	bool _rgb;
	ushort _channels;
	int _imageH;
	int _imageW;
	int _centerX;
	int _centerY;

	double *d_gauss;

	int _retinaSize;
	SamplingPoint *d_points;

	double *_d_imageVector;
};
#endif //RETINA__CUH
