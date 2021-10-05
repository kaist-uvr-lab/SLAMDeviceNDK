#ifndef UNITYLIBRARY_LOCAL_MAP_H
#define UNITYLIBRARY_LOCAL_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace EdgeSLAM {
	class Frame;
	class RefFrame;
	class MapPoint;
	class TrackPoint;
	class LocalMap {
	public:
		LocalMap();
		virtual ~LocalMap();
    public:

	public:
		std::vector<MapPoint*> mvpMapPoints;
		std::vector<TrackPoint*> mvpTrackPoints;
		cv::Mat mvfAngles;
		cv::Mat mvcScales;
		cv::Mat mDescriptors;
	};
}

#endif