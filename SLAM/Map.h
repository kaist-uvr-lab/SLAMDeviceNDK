#ifndef UNITY_LIBRARY_SLAM_MAP_H
#define UNITY_LIBRARY_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

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
    public:
        bool CheckMapPoint(int id);
        void AddMapPoint(int id, MapPoint* pMP);
        MapPoint* GetMapPoint(int id);
        void RemoveMapPoint(int id);
    private:
        std::mutex mMutexMapPoints;
        std::map<int, MapPoint*> mmpMapPoints;
    public:
        void AddImage(cv::Mat img, int id);
        cv::Mat GetImage(int id);
    private:
        std::mutex mMutexImages;
        std::map<int, cv::Mat> mapImages;
	};
}

#endif