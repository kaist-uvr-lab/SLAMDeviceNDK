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

	class LocalMap;
	class Map {
	public:
		Map();
		virtual ~Map();
    public:

        ConcurrentMap<int, MapPoint*> mapMapPoints;
    public:
        void SetLocalMap(LocalMap* pLocal);
        void GetLocalMap(LocalMap* pLocal);
    private:
        std::mutex mMutexLocalMap;
        LocalMap* mpLocalMap;
	public:
		void SetReferenceFrame(RefFrame* pRef);
		RefFrame* GetReferenceFrame();

	private:
		std::mutex mMutexRefFrame;
		RefFrame* mpRefFrame;

	};
}

#endif