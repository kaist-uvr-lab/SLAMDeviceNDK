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
	class ORBDetector;
	class MotionModel;
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
		int TrackWithLocalMap(Frame* cur, std::vector<MapPoint*> vpLocalMPs, float thMaxDesc, float thMinDesc);
		int TrackWithReferenceFrame(RefFrame* ref, Frame* cur, float thMaxDesc, float thMinDesc);

		int DiscardOutliers(Frame* cur);
		int UpdateVisiblePoints(Frame* cur, std::vector<MapPoint*> vpLocalMPs);
		int UpdateFoundPoints(Frame* cur, bool bOnlyTracking = false);
		//bool NeedNewKeyFrame(Frame* cur, RefFrame* ref, int nKFs, int nMatchesInliers);
	public:
		std::atomic<int> mnLastRelocFrameId, mnLastKeyFrameId;
		std::string filename;
		int mnMaxFrame, mnMinFrame;
		TrackingState mTrackState;
	};

}
#endif