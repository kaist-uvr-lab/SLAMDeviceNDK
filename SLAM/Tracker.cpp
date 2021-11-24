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
		//int nmatchesMap = DiscardOutliers(cur);
		return nopt;
	}
	int Tracker::TrackWithReferenceFrame(RefFrame* ref, Frame* cur) {

		cur->reset_map_points();

		std::vector<EdgeSLAM::MapPoint*> vpMapPointMatches;
		int nMatch = EdgeSLAM::SearchPoints::SearchFrameByBoW(ref, cur, vpMapPointMatches);
		if (nMatch < 10)
			return nMatch;

		cur->SetPose(ref->GetPose());
		cur->mvpMapPoints = vpMapPointMatches;
		
		/*freopen("debug.txt", "a", stdout);
		cv::Mat pose = cur->GetPose();
		printf("Before %d ref = %f %f %f\n%f %f %f \n%f %f %f\n%f %f %f\n", ref->mnId,
			pose.at<float>(0, 0), pose.at<float>(0, 1), pose.at<float>(0, 2),
			pose.at<float>(1, 0), pose.at<float>(1, 1), pose.at<float>(1, 2),
			pose.at<float>(2, 0), pose.at<float>(2, 1), pose.at<float>(2, 2),
			pose.at<float>(0, 3), pose.at<float>(1, 3), pose.at<float>(2, 3));*/
		int nopt = EdgeSLAM::Optimizer::PoseOptimization(cur);
		//int nmatchesMap = DiscardOutliers(cur);
		/*pose = cur->GetPose();
		printf("after ref = %f %f %f\n%f %f %f \n%f %f %f\n%f %f %f\n",
			pose.at<float>(0, 0), pose.at<float>(0, 1), pose.at<float>(0, 2),
			pose.at<float>(1, 0), pose.at<float>(1, 1), pose.at<float>(1, 2),
			pose.at<float>(2, 0), pose.at<float>(2, 1), pose.at<float>(2, 2),
			pose.at<float>(0, 3), pose.at<float>(1, 3), pose.at<float>(2, 3));
		printf("ref matching %d %d\n", nMatch, nopt);*/
		return nopt;
	}
	int Tracker::TrackWithLocalMap(Frame* cur, LocalMap* pLocal, float thMaxDesc, float thMinDesc) {

        int nMatch = UpdateVisiblePoints(cur, pLocal->mvpMapPoints, pLocal->mvpTrackPoints);
        std::ofstream ofile;
        //ofile.open(path.c_str(), std::ios_base::out | std::ios_base::app);
        //ofile<<"1"<<std::endl;
		if (nMatch == 0)
			return 0;

        float thRadius = 1.0;
		if (cur->mnFrameID < mnLastRelocFrameId + 2)
			thRadius = 5.0;
        //ofile<<"2"<<std::endl;
		int a = SearchPoints::SearchMapByProjection(cur, pLocal->mvpMapPoints, pLocal->mvpTrackPoints, thMaxDesc, thMinDesc, thRadius);
        //ofile<<"3"<<std::endl;
		Optimizer::PoseOptimization(cur);
        //ofile<<"4"<<std::endl;
		/*
		LocalMap* pLocalMap = new LocalCovisibilityMap();
		std::vector<MapPoint*> vpLocalMPs;
		std::vector<RefFrame*> vpLocalKFs;
		std::vector<TrackPoint*> vpLocalTPs;
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		pLocalMap->UpdateLocalMap(cur, vpLocalKFs, vpLocalMPs, vpLocalTPs);

		int nMatch = UpdateVisiblePoints(cur, vpLocalMPs, vpLocalTPs);
		if (nMatch == 0)
			return 0;

		float thRadius = 1.0;
		if (cur->mnFrameID < mnLastRelocFrameId + 2)
			thRadius = 5.0;

		int a = SearchPoints::SearchMapByProjection(cur, vpLocalMPs, vpLocalTPs, thMaxDesc, thMinDesc, thRadius);

		Optimizer::PoseOptimization(cur);
        */
		return UpdateFoundPoints(cur);
	}

	 
	 int Tracker::DiscardOutliers(Frame* cur) {
		 int nres = 0;
		 for (int i = 0; i<cur->N; i++)
		 {
			 if (cur->mvpMapPoints[i])
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
	 int Tracker::UpdateVisiblePoints(Frame* cur, std::vector<MapPoint*> vpLocalMPs, std::vector<TrackPoint*> vpLocalTPs) {

		 for (int i = 0; i<cur->N; i++)
		 {
			 if (cur->mvpMapPoints[i])
			 {
				 MapPoint* pMP = cur->mvpMapPoints[i];
				 if (!pMP || cur->mvbOutliers[i]) {
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
			 TrackPoint* pTP = vpLocalTPs[i];

			 if (cur->mspMapPoints.count(pMP))
				 continue;
			 nTrial++;

			 // Project (this fills MapPoint variables for matching)
			 //if (cur->is_in_frustum(pMP, pTP, 0.5))
			 {
				 nToMatch++;
			 }
		 }
		 cv::Mat pose = cur->GetPose();
		 return nToMatch;
	 }

	 int Tracker::UpdateFoundPoints(Frame* cur, bool bOnlyTracking) {
		 int nres = 0;
		 // Update MapPoints Statistics
		 for (int i = 0; i<cur->N; i++)
		 {
			 if (cur->mvpMapPoints[i])
			 {
				 if (!cur->mvbOutliers[i])
				 {
					 if (!bOnlyTracking)
					 {
						 if (cur->mvpMapPoints[i]->Observations()>0)
							 nres++;
					 }
					 else
						 nres++;
				 }
			 }
		 }
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
