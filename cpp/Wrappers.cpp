#include "Retina.cuh"
#include "Cortex.cuh"
#include <algorithm>
/**
 * These wrapper functions are C wrappers for the underlying CUDA C++ implementation.
 * Python ctypes can only communicate with C functions, so here we go.
 */

std::vector<SamplingPoint> spVecFromArrays(float *h_loc, size_t numOfLocs, double *h_coeff = NULL) {
	std::vector<SamplingPoint> v;
	int coeffOffset = 0;
	for (int i = 0; i != numOfLocs; ++i) {
		float *locStart = h_loc + i * 7;
		size_t kernelSize = *(locStart + 6);
		double *coeffStart = h_coeff + coeffOffset;
		SamplingPoint sp(*locStart, *(locStart + 1), *(locStart + 2),
			*(locStart + 3), *(locStart + 4), *(locStart + 5), kernelSize);
		if (h_coeff != NULL)
			sp.setKernel(std::vector<double>(coeffStart, coeffStart + kernelSize * kernelSize));
		v.push_back(sp);
		coeffOffset += kernelSize * kernelSize;
	}
	return v;
}

std::vector<double2> d2VecFromArray(double *h_loc, size_t numOfLocs) {
	std::vector<double2> v;
	for (int i = 0; i != numOfLocs; ++i) {
		double2 d2;
		d2.x = *(h_loc + 2 * i);
		d2.y = *(h_loc + 2 * i + 1);
		v.push_back(d2);
	}
	return v;
}


extern "C" {
/**
 * Wrappers for the Retina class.
 */
	Retina* Retina_new() { return new Retina(); }

	int Retina_setSamplingFields(Retina *ret, float *h_loc, double *h_coeff, size_t numOfLocs) {
		auto tmp = spVecFromArrays(h_loc, numOfLocs, h_coeff);
		return ret->setSamplingFields(tmp.data(), tmp.size());
	}

	int Retina_getSamplingFields(Retina *ret, float *h_loc, double *h_coeff, size_t retinaSize) {
		return 0;
		/*SamplingPoint *h_points = new SamplingPoint[retinaSize];
		return ret->getSamplingFields(h_points, retinaSize);
		for (int i = 0; i != retinaSize; ++i)
			h_loc[]

		delete [] h_points;*/
	}

	int Retina_setGaussNormImage(Retina *ret, double *h_gauss = NULL, size_t gaussH = 0,
			   	   	   	   	   	 size_t gaussW = 0, size_t gaussC = 0) {
		return ret->setGaussNormImage(h_gauss, gaussH, gaussW, gaussC);
	}

	int Retina_getGaussNormImage(Retina *ret, double *h_gauss, size_t gaussH, size_t gaussW, size_t gaussC) {
		return ret->getGaussNormImage(h_gauss, gaussH, gaussW, gaussC);
	}

	int Retina_sample(Retina *ret, const uchar *h_imageIn, size_t imageH, size_t imageW, size_t imageC,
					  double *h_imageVector, size_t vectorLength, bool keepImageVectorOnDevice = false) {
		return ret->sample(h_imageIn, imageH, imageW, imageC, h_imageVector, vectorLength, keepImageVectorOnDevice);
	}


	int Retina_inverse(Retina *ret, const double *h_imageVector,  size_t vectorLength,
					   uchar *h_imageInverse, size_t imageH, size_t imageW, size_t imageC,
					   bool useImageVectorOnDevice = false) {
		return ret->inverse(h_imageVector, vectorLength, h_imageInverse, imageH, imageW, imageC, useImageVectorOnDevice);
	}

	int Retina_getRetinaSize(Retina *ret) { return ret->getRetinaSize(); }

	int Retina_getImageHeight(Retina *ret) { return ret->getImageHeight(); }
	void Retina_setImageHeight(Retina *ret, const int imageH) { ret->setImageHeight(imageH); }

	int Retina_getImageWidth(Retina *ret) { return ret->getImageWidth(); }
	void Retina_setImageWidth(Retina *ret, const int imageW) { ret->setImageWidth(imageW); }

	bool Retina_getRGB(Retina *ret) { return ret->getRGB(); }
	void Retina_setRGB(Retina *ret, const bool rgb) { ret->setRGB(rgb);	}

	int Retina_getCenterX(Retina *ret) { return ret->getCenterX(); }
	void Retina_setCenterX(Retina *ret, const int centerX) { ret->setCenterX(centerX); }

	int Retina_getCenterY(Retina *ret) { return ret->getCenterY(); }
	void Retina_setCenterY(Retina *ret, const int centerY) { ret->setCenterY(centerY); }

/**
 * Wrappers for the Cortex class.
 */
	Cortex* Cortex_new() { return new Cortex(); }

	int Cortex_locationsFromRetinaFields(Cortex *cort, float *h_loc = NULL, size_t numOfLocs = 0) {
		if (h_loc == NULL || numOfLocs == 0) {
			return cort->locationsFromCortexFields(nullptr, 0, nullptr, 0);
		}
		std::vector<SamplingPoint> left;
		std::vector<SamplingPoint> right;
		for (int i = 0; i != numOfLocs; ++i) {
			float *locStart = h_loc + i * 7;
			size_t kernelSize = *(locStart + 6);
			SamplingPoint sp(*locStart, *(locStart + 1), i,
			*(locStart + 3), *(locStart + 4), *(locStart + 5), kernelSize);
			sp._x < 0 ? left.push_back(sp) : right.push_back(sp);
		}
		return cort->locationsFromCortexFields(left.data(), left.size(), right.data(), right.size());
	}

	float Cortex_getAlpha(Cortex *cort) { return cort->getAlpha(); }
	void Cortex_setAlpha(Cortex *cort, float alpha) { cort->setAlpha(alpha); }

	float Cortex_getShrink(Cortex *cort) { return cort->getShrink(); }
	void Cortex_setShrink(Cortex *cort, float shrink) { cort->setShrink(shrink); }

	bool Cortex_getRGB(Cortex *cort) { return cort->getRGB(); }
	void Cortex_setRGB(Cortex *cort, bool rgb) { cort->setRGB(rgb); }

	uint Cortex_getCortImageX(Cortex *cort) { return cort->getCortImageSize().x; }
	uint Cortex_getCortImageY(Cortex *cort) { return cort->getCortImageSize().y; }
	void Cortex_setCortImageSize(Cortex *cort, uint cortImgX, uint cortImgY) {
		uint2 cortImgSize; cortImgSize.x = cortImgX; cortImgSize.y = cortImgY;
		cort->setCortImageSize(cortImgSize);
	}

	size_t Cortex_getLeftSize(Cortex *cort) { return cort->getLeftSize(); }
	int Cortex_setLeftCortexFields(Cortex *cort, float *h_fields, size_t numOfFields) {
		auto spv = spVecFromArrays(h_fields, numOfFields);
		return cort->setLeftCortexFields(spv.data(), spv.size());
	}

	size_t Cortex_getRightSize(Cortex *cort) { return cort->getRightSize(); }
	int Cortex_setRightCortexFields(Cortex *cort, float *h_fields, size_t numOfFields) {
		auto spv = spVecFromArrays(h_fields, numOfFields);
		return cort->setRightCortexFields(spv.data(), spv.size());
	}

	int Cortex_setLeftCortexLocations(Cortex *cort, double *h_loc, size_t numOfLocs) {
		auto d2v = d2VecFromArray(h_loc, numOfLocs);
		return cort->setLeftCortexLocations(d2v.data(), d2v.size());
	}
	int Cortex_setRightCortexLocations(Cortex *cort, double *h_loc, size_t numOfLocs) {
		auto d2v = d2VecFromArray(h_loc, numOfLocs);
		return cort->setRightCortexLocations(d2v.data(), d2v.size());
	}

	size_t Cortex_getGaussKernelWidth(Cortex *cort) { return cort->getGaussKernelWidth(); }
	float Cortex_getGaussSigma(Cortex *cort) { return cort->getGaussSigma(); }

	int Cortex_setGauss100(Cortex *cort, const uint kernelWidth, const float sigma, double *h_gauss = NULL) {
		return cort->setGauss100(kernelWidth, sigma, h_gauss);
	}

	int Cortex_cortImageLeft(Cortex *cort, double *h_imageVector,  size_t vecLen, uchar *h_result,
				size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = NULL) {
		return cort->cortImageLeft(h_imageVector, vecLen, h_result, cortImgX, cortImgY, rgb, d_imageVector);
	}

	int Cortex_cortImageRight(Cortex *cort, double *h_imageVector,  size_t vecLen, uchar *h_result,
				size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = NULL) {
		return cort->cortImageRight(h_imageVector, vecLen, h_result, cortImgX, cortImgY, rgb, d_imageVector);
	}
}
