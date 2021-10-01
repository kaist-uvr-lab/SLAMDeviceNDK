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
		virtual void UpdateLocalMap(Frame* f, std::vector<RefFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs) {}
		virtual void UpdateLocalKeyFrames(Frame* f, std::vector<RefFrame*>& vpLocalKFs) {}
		virtual void UpdateLocalMapPoitns(Frame* f, std::vector<RefFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs) {}
	public:
	    static std::string logFile;
	private:
		std::mutex mMutexLocalMap;

	};

	class LocalCovisibilityMap :public LocalMap {
	public:
		LocalCovisibilityMap();
		virtual ~LocalCovisibilityMap();
	public:
		void UpdateLocalMap(Frame* f, std::vector<RefFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs);
		void UpdateLocalKeyFrames(Frame* f, std::vector<RefFrame*>& vpLocalKFs);
		void UpdateLocalMapPoitns(Frame* f, std::vector<RefFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs);
		//void UpdateKeyFrames
	private:

	};
}

#endif