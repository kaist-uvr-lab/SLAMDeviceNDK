#ifndef UNITY_LIBARARY_ORB_DETECTOR_H
#define UNITY_LIBARARY_ORB_DETECTOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <FeatureDetector.h>
//#include <FeatureInfo.h>
#include "ORBExtractor.h"

namespace EdgeSLAM {
	class ORBDetector{
	public:
		ORBDetector(int nFeatures = 1000, float fScaleFactor = 1.2, int nLevels = 8, float fInitThFast = 20, float fMinThFast = 7);
		virtual ~ORBDetector();
		void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);
		void Compute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);
		void init_sigma_level();
		float CalculateDescDistance(cv::Mat a, cv::Mat b);
	public:
		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;
	private:
		ORBextractor* detector;
	};
}

#endif