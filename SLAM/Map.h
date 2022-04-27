#ifndef UNITY_LIBRARY_SLAM_MAP_H
#define UNITY_LIBRARY_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include "ConcurrentMap.h"

namespace EdgeSLAM {
	class Frame;
	class RefFrame;
	class MapPoint;

	class Map {
	public:
		Map();
		virtual ~Map();
    public:
        ConcurrentMap<int, MapPoint*> MapPoints;
		void SetReferenceFrame(RefFrame* pRef);
		RefFrame* GetReferenceFrame();
	private:
		std::mutex mMutexRefFrame;
		RefFrame* mpRefFrame;

	};
}

#endif