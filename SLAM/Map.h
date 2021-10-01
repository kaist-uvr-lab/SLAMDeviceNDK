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
	
	class Map {
	public:
		Map();
		virtual ~Map();
	/*public:
		void AddMapPoint(MapPoint* pMP);
		void RemoveMapPoint(MapPoint* pMP);
		std::vector<MapPoint*> GetAllMapPoints();
		int GetNumMapPoints();

		void AddKeyFrame(RefFrame* pF);
		RefFrame* GetKeyFrame(int id);
		void RemoveKeyFrame(RefFrame* pF);
		std::vector<RefFrame*> GetAllKeyFrames();
		int GetNumKeyFrames();*/

		void AddImage(cv::Mat gray, int id);
		cv::Mat GetImage(int id);

	public:
		void SetReferenceFrame(RefFrame* pRef);
		RefFrame* GetReferenceFrame();
		std::mutex mMutexFrames, mMutexRefFrames, mMutexMapPoints;

		std::map<int, Frame*> mmpFrames;
		std::map<int, RefFrame*> mmpRefFrames;
		std::map<int, MapPoint*> mmpMapPoints;
	private:
	    std::mutex mMutexFrame;
	    std::map<int, cv::Mat> mapGrayImages;

		std::mutex mMutexRefFrame;
		RefFrame* mpRefFrame;
	};
}

#endif