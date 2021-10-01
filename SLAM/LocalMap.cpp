#include "LocalMap.h"
#include "Frame.h"
#include "RefFrame.h"
#include "MapPoint.h"

namespace EdgeSLAM {
	LocalMap::LocalMap() {}
	LocalMap::~LocalMap() {}
	LocalCovisibilityMap::LocalCovisibilityMap() :LocalMap()
	{}
	LocalCovisibilityMap::~LocalCovisibilityMap() {

	}
	void LocalCovisibilityMap::UpdateLocalMap(Frame* f, std::vector<RefFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs) {

		UpdateLocalKeyFrames(f, vpLocalKFs);
		UpdateLocalMapPoitns(f, vpLocalKFs, vpLocalMPs, vpLocalTPs);

	}
	void LocalCovisibilityMap::UpdateLocalKeyFrames(Frame* f, std::vector<RefFrame*>& vpLocalKFs) {
		// Each map point vote for the keyframes in which it has been observed
        std::map<RefFrame*, int> keyframeCounter;
		int nFrameID = f->mnFrameID;
		for (int i = 0; i<f->N; i++)
		{
			auto pMP = f->mvpMapPoints[i];
			if (pMP) {
				const std::map<RefFrame*, size_t> observations = pMP->GetObservations();
				for (std::map<RefFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
					keyframeCounter[it->first]++;
			}
			else {
				f->mvpMapPoints[i] = nullptr;
			}
		}

		if (keyframeCounter.empty())
			return;

		int max = 0;
		RefFrame* pKFmax = static_cast<RefFrame*>(nullptr);
		std::set<RefFrame*> spKFs;

		// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
		for (std::map<RefFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
		{
			RefFrame* pKF = it->first;
			
			if (it->second>max)
			{
				max = it->second;
				pKFmax = pKF;
			}

			vpLocalKFs.push_back(it->first);
			spKFs.insert(pKF);
		}
		// Include also some not-already-included keyframes that are neighbors to already-included keyframes
		//for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		for (size_t i = 0, iend = vpLocalKFs.size(); i < iend; i++)
		{
			// Limit the number of keyframes
			if (vpLocalKFs.size()>80)
				break;

			RefFrame* pKF = vpLocalKFs[i];// *itKF;

			const std::vector<RefFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

			for (std::vector<RefFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
			{
				RefFrame* pNeighKF = *itNeighKF;
				if (pNeighKF && !spKFs.count(pNeighKF))
				{
					spKFs.insert(pNeighKF);
					break;
					/*if (pNeighKF->mnTrackReferenceForFrame != nFrameID)
					{
					vpLocalKFs.push_back(pNeighKF);
					pNeighKF->mnTrackReferenceForFrame = nFrameID;
					break;
					}*/
				}
			}

			const std::set<RefFrame*> spChilds = pKF->GetChilds();
			for (std::set<RefFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
			{
				RefFrame* pChildKF = *sit;
				if (pChildKF && !spKFs.count(pChildKF))
				{
					spKFs.insert(pChildKF);
					break;
					/*if (pChildKF->mnTrackReferenceForFrame != nFrameID)
					{
					vpLocalKFs.push_back(pChildKF);
					pChildKF->mnTrackReferenceForFrame = nFrameID;
					break;
					}*/
				}
			}

			RefFrame* pParent = pKF->GetParent();
			if (pParent && !spKFs.count(pParent))
			{
				spKFs.insert(pParent);
				break;
				/*if (pParent->mnTrackReferenceForFrame != nFrameID)
				{
				vpLocalKFs.push_back(pParent);
				pParent->mnTrackReferenceForFrame = nFrameID;
				break;
				}*/
			}

		}
		/*if (pKFmax)
		{
			user->mnReferenceKeyFrameID = pKFmax->mnId;
		}*/
	}
	void LocalCovisibilityMap::UpdateLocalMapPoitns(Frame* f, std::vector<RefFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs) {
		std::set<MapPoint*> spMPs;
		for (std::vector<RefFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			RefFrame* pKF = *itKF;
			const std::vector<MapPoint*> vpMPs = pKF->mvpMapPoints;

			for (std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
			    if(spMPs.size() == 1000)
			        break;
				MapPoint* pMP = *itMP;
				if (!pMP || spMPs.count(pMP))
					continue;
				vpLocalMPs.push_back(pMP);
				vpLocalTPs.push_back(new TrackPoint());
				spMPs.insert(pMP);
				//pMP->mnTrackReferenceForFrame = f->mnFrameID;
			}
		}
	}
}