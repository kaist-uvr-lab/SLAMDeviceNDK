#include "Tracker.h"
#include "Frame.h"
#include "RefFrame.h"
#include "MapPoint.h"
#include "LocalMap.h"
#include "SearchPoints.h"
#include "ORBDetector.h"
#include "Optimizer.h"
#include <chrono>

namespace EdgeSLAM {
	Tracker::Tracker():mnLastRelocFrameId(0), mnLastKeyFrameId(0), mnMaxFrame(30), mnMinFrame(3), mTrackState(TrackingState::NotEstimated){}
	Tracker::~Tracker() {}

	int Tracker::TrackWithPrevFrame(Frame* prev, Frame* cur, float thMaxDesc, float thMinDesc) {
		cur->reset_map_points();

		int res = SearchPoints::SearchFrameByProjection(prev, cur, thMaxDesc, thMinDesc);
		if (res < 20) {
			cur->reset_map_points();
			res = SearchPoints::SearchFrameByProjection(prev, cur, thMaxDesc, thMinDesc, 30.0);
		}
		if (res < 20) {
			return res;
		}
		int nopt = Optimizer::PoseOptimization(cur);
		return nopt;
	}
	int Tracker::TrackWithReferenceFrame(RefFrame* ref, Frame* cur, float thMaxDesc, float thMinDesc) {

		cur->reset_map_points();

        cur->SetPose(ref->GetPose());
        int res = SearchPoints::SearchFrameByProjection(ref, cur, thMaxDesc, thMinDesc);

        if (res < 20) {
            cur->reset_map_points();
            res = SearchPoints::SearchFrameByProjection(ref, cur, thMaxDesc, thMinDesc, 30.0);
        }
        if (res < 20) {
            return res;
        }

        /*
        //prev version
		std::vector<EdgeSLAM::MapPoint*> vpMapPointMatches;
		int nMatch = EdgeSLAM::SearchPoints::SearchFrameByBoW(ref, cur, vpMapPointMatches);
		if (nMatch < 10)
			return nMatch;
		cur->SetPose(ref->GetPose());
		cur->mvpMapPoints = vpMapPointMatches;
		//prev version
		*/

		int nopt = EdgeSLAM::Optimizer::PoseOptimization(cur);
		return nopt;
	}
	int Tracker::TrackWithLocalMap(Frame* cur, LocalMap* pLocal, float thMaxDesc, float thMinDesc) {
        int nMatch = UpdateVisiblePoints(cur, pLocal->mvpMapPoints);
		if (nMatch == 0)
			return 0;
        float thRadius = 1.0;
		if (cur->mnFrameID < mnLastRelocFrameId + 2)
			thRadius = 5.0;
		int a = SearchPoints::SearchMapByProjection(cur, pLocal->mvpMapPoints, thMaxDesc, thMinDesc, thRadius);
		std::ofstream ofile;
        ofile.open(path.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<"Track::LocalMap::Optimize:Start"<<std::endl;
		Optimizer::PoseOptimization(cur);
		ofile<<"Track::LocalMap::Optimize:End"<<std::endl;
		ofile.close();
		return UpdateFoundPoints(cur);
	}

	 int Tracker::DiscardOutliers(Frame* cur) {
		 int nres = 0;
		 for (int i = 0; i<cur->N; i++)
		 {
			 if (cur->mvpMapPoints[i] && !cur->mvpMapPoints[i]->isBad())
			 {
				 if (cur->mvbOutliers[i])
				 {
					 MapPoint* pMP = cur->mvpMapPoints[i];

					 cur->mvpMapPoints[i] = nullptr;
					 cur->mvbOutliers[i] = false;
					 cur->mspMapPoints.insert(pMP);
				 }
				 else if (cur->mvpMapPoints[i]->Observations()>0)
					 nres++;
			 }
		 }
		 return nres;
	 }
	 int Tracker::UpdateVisiblePoints(Frame* cur, std::vector<MapPoint*> vpLocalMPs) {

		 for (int i = 0; i<cur->N; i++)
		 {
			 if (cur->mvpMapPoints[i])
			 {
				 MapPoint* pMP = cur->mvpMapPoints[i];
				 if (!pMP || pMP->isBad() || cur->mvbOutliers[i]) {
					 cur->mvpMapPoints[i] = nullptr;
					 cur->mvbOutliers[i] = false;
				 }
				 cur->mspMapPoints.insert(pMP);
			 }
		 }

		 int nTrial = 0;
		 int nToMatch = 0;
		 // Project points in frame and check its visibility
		 for (size_t i = 0, iend = vpLocalMPs.size(); i < iend; i++)
			 //for (auto vit = vpLocalMPs.begin(), vend = vpLocalMPs.end(); vit != vend; vit++)
		 {
			 MapPoint* pMP = vpLocalMPs[i];
             if(!pMP || pMP->isBad())
                continue;
			 if (cur->mspMapPoints.count(pMP))
				 continue;
			 nTrial++;

			 // Project (this fills MapPoint variables for matching)
			 if (cur->is_in_frustum(pMP, 0.5))
			 {
				 nToMatch++;
			 }
		 }
		 cv::Mat pose = cur->GetPose();
		 return nToMatch;
	 }

	 int Tracker::UpdateFoundPoints(Frame* cur, bool bOnlyTracking) {

        std::ofstream ofile;
        ofile.open(path.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<"UpdateFoundPoints::a"<<std::endl;
		 int nres = 0;
		 // Update MapPoints Statistics
		 for (int i = 0; i<cur->N; i++)
		 {
		     auto pMP = cur->mvpMapPoints[i];
			 if (pMP && !pMP->isBad())
			 {
				 if (!cur->mvbOutliers[i])
				 {
					 if (!bOnlyTracking)
					 {
						 if (pMP->Observations()>0)
							 nres++;
					 }
					 else
						 nres++;
				 }
			 }
		 }
		 ofile<<"UpdateFoundPoints::b"<<std::endl;
		 ofile.close();
		 return nres;
	 }

/*
	 bool Tracker::NeedNewKeyFrame(Frame* cur, RefFrame* ref, int nKFs, int nMatchesInliers)
	 {

		 // Do not insert keyframes if not enough frames have passed from last relocalisation
		 if (cur->mnFrameID<mnLastRelocFrameId + mnMaxFrame && nKFs>mnMaxFrame)
			 return false;

		 // Tracked MapPoints in the reference keyframe
		 int nMinObs = 3;
		 if (nKFs <= 2)
			 nMinObs = nKFs;
		 int nRefMatches = ref->TrackedMapPoints(nMinObs);

		 // Thresholds
		 float thRefRatio = 0.9f;

		 const bool c1a = cur->mnFrameID >= mnLastKeyFrameId + mnMaxFrame;
		 const bool c1b = cur->mnFrameID >= mnLastKeyFrameId + mnMinFrame;
		 const bool c2 = (nMatchesInliers<nRefMatches*thRefRatio) && nMatchesInliers>15;

		 if ((c1a || c1b) && c2)
		 {
			 // If the mapping accepts keyframes, insert keyframe.
			 // Otherwise send a signal to interrupt BA
			 return true;
			 
		 }
		 return false;
	 }
	*/
}
