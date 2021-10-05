#ifndef UNITYLIBRARY_TRACKER_H
#define UNITYLIBRARY_TRACKER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>
namespace EdgeSLAM {
	class Frame;
	class RefFrame;
	class MapPoint;
	class TrackPoint;
	class ORBDetector;
	class MotionModel;
	class LocalMap;
	enum class TrackingState {
		NotEstimated, Success, Failed
	};
	// = TrackingState::NotEstimated;
	class Tracker {
	public:
		Tracker();
		virtual ~Tracker();
	public:
		static ORBDetector* Detector;
		int TrackWithPrevFrame(Frame* prev, Frame* cur, float thMaxDesc, float thMinDesc);
		int TrackWithLocalMap(Frame* cur, LocalMap* pLocal, float thMaxDesc, float thMinDesc);
		int TrackWithReferenceFrame(RefFrame* ref, Frame* cur);

		int DiscardOutliers(Frame* cur);
		int UpdateVisiblePoints(Frame* cur, std::vector<MapPoint*> vpLocalMPs, std::vector<TrackPoint*> vpLocalTPs);
		int UpdateFoundPoints(Frame* cur, bool bOnlyTracking = false);
		//bool NeedNewKeyFrame(Frame* cur, RefFrame* ref, int nKFs, int nMatchesInliers);

		std::string path;
	public:
		std::atomic<int> mnLastRelocFrameId, mnLastKeyFrameId;
		int mnMaxFrame, mnMinFrame;
		TrackingState mTrackState;
	};

}
#endif