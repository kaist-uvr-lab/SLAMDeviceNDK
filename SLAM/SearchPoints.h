#ifndef UNITYLIBRARY_SEARCH_POINTS_H
#define UNITYLIBRARY_SEARCH_POINTS_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>

namespace EdgeSLAM {
	class ORBDetector;
	class RefFrame;
	class Frame;
	class MapPoint;
	class TrackPoint;
	class SearchPoints {
	public:
		static const int HISTO_LENGTH;
		static ORBDetector* Detector;
		static int SearchFrameByProjection(RefFrame* ref, Frame* curr, float thMaxDesc = 100.0, float thMinDesc = 50.0, float thProjection = 15, bool bCheckOri = true);
		static int SearchFrameByProjection(Frame* prev, Frame* curr, float thMaxDesc = 100.0, float thMinDesc = 50.0, float thProjection = 15, bool bCheckOri = true);
		static int SearchMapByProjection(Frame *F, const std::vector<MapPoint*> &vpMapPoints, float thMaxDesc = 100.0, float thMinDesc = 50.0, float thRadius = 1.0, float thMatchRatio = 0.8, bool bCheckOri = true);
		static void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
		static float RadiusByViewingCos(const float &viewCos);
		static std::string filename;
	};
}
#endif