#include "RefFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "Camera.h"
#include "CameraPose.h"
#include "ORBDetector.h"

namespace EdgeSLAM {
	int RefFrame::nId = 0;
	RefFrame::RefFrame() {}
	RefFrame::RefFrame(Map* map, cv::Mat img, Camera* pCam, float* data) :mpCamera(pCam), mnId(RefFrame::nId++),
		K(pCam->K), D(pCam->D), fx(pCam->fx), fy(pCam->fy), cx(pCam->cx), cy(pCam->cy), invfx(pCam->invfx), invfy(pCam->invfy), mnMinX(pCam->u_min), mnMaxX(pCam->u_max), mnMinY(pCam->v_min), mnMaxY(pCam->v_max), mfGridElementWidthInv(pCam->mfGridElementWidthInv), mfGridElementHeightInv(pCam->mfGridElementHeightInv), FRAME_GRID_COLS(pCam->mnGridCols), FRAME_GRID_ROWS(pCam->mnGridRows), mbDistorted(pCam->bDistorted),
		mnScaleLevels(detector->mnScaleLevels), mfScaleFactor(detector->mfScaleFactor), mfLogScaleFactor(detector->mfLogScaleFactor), mvScaleFactors(detector->mvScaleFactors), mvInvScaleFactors(detector->mvInvScaleFactors), mvLevelSigma2(detector->mvLevelSigma2), mvInvLevelSigma2(detector->mvInvLevelSigma2),
		mpParent(nullptr)
	{
		N = (int)data[0];
		int nIdx = 13;
		mvKeys = std::vector<cv::KeyPoint>(N);
		mvpMapPoints = std::vector<MapPoint*>(N, static_cast<MapPoint*>(nullptr));

		cv::Mat tempT = cv::Mat::eye(4, 4, CV_32FC1);
		tempT.at<float>(0, 0) = data[1];
		tempT.at<float>(0, 1) = data[2];
		tempT.at<float>(0, 2) = data[3];
		tempT.at<float>(1, 0) = data[4];
		tempT.at<float>(1, 1) = data[5];
		tempT.at<float>(1, 2) = data[6];
		tempT.at<float>(2, 0) = data[7];
		tempT.at<float>(2, 1) = data[8];
		tempT.at<float>(2, 2) = data[9];
		tempT.at<float>(0, 3) = data[10];
		tempT.at<float>(1, 3) = data[11];
		tempT.at<float>(2, 3) = data[12];
		mpCamPose = new CameraPose(tempT);

		for (int i = 0; i < N; i++) {
			cv::KeyPoint kp;
			kp.pt.x = data[nIdx++];
			kp.pt.y = data[nIdx++];
			kp.octave = (int)data[nIdx++];
			kp.angle = data[nIdx++];
			int id = (int)data[nIdx++];
			float x = data[nIdx++];
			float y = data[nIdx++];
			float z = data[nIdx++];
			MapPoint* mp = nullptr;
			if (map->mmpMapPoints.count(id)) {
				mp = map->mmpMapPoints[id];
				if(mp)
					mp->SetWorldPos(x, y, z);
			}
			else {
				mp = new MapPoint(x, y, z);
				mp->SetReferenceFrame(this);
				map->mmpMapPoints[id] = mp;
			}
			mvKeys[i] = kp;
			mvpMapPoints[i] = mp;
		}

		//imgGray = img.clone();
		detector->Compute(img, cv::Mat(), mvKeys, mDescriptors);

		if (mbDistorted)
            UndistortKeyPoints();
        else
            mvKeysUn = mvKeys;

	}
	RefFrame::~RefFrame() {}
	void RefFrame::UpdateMapPoints() {
		const std::vector<MapPoint*> vpMapPointMatches = GetMapPointMatches();

		for (size_t i = 0; i<vpMapPointMatches.size(); i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->IsInKeyFrame(this))
				{
					pMP->AddObservation(this, i);
					pMP->UpdateNormalAndDepth();
					pMP->ComputeDistinctiveDescriptors();
				}
			}
		}
	}
	void RefFrame::UpdateConnections() {
		std::map<RefFrame*, int> KFcounter;

		std::vector<MapPoint*> vpMP;

		{
			std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
			vpMP = mvpMapPoints;
		}

		//For all map points in keyframe check in which other keyframes are they seen
		//Increase counter for those keyframes
		for (std::vector<MapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;

			if (!pMP)
				continue;

			auto observations = pMP->GetObservations();

			for (std::map<RefFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				if (mit->first->mnId == mnId)
					continue;
				KFcounter[mit->first]++;
			}
		}

		// This should not happen
		if (KFcounter.empty())
			return;

		//If the counter is greater than threshold add connection
		//In case no keyframe counter is over threshold add the one with maximum counter
		int nmax = 0;
		RefFrame* pKFmax = nullptr;
		int th = 15;

		std::vector<std::pair<int, RefFrame*> > vPairs;
		vPairs.reserve(KFcounter.size());
		for (auto mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
		{
			if (mit->second>nmax)
			{
				nmax = mit->second;
				pKFmax = mit->first;
			}
			if (mit->second >= th)
			{
				vPairs.push_back(std::make_pair(mit->second, mit->first));
				(mit->first)->AddConnection(this, mit->second);
			}
		}

		if (vPairs.empty())
		{
			vPairs.push_back(std::make_pair(nmax, pKFmax));
			pKFmax->AddConnection(this, nmax);
		}

		sort(vPairs.begin(), vPairs.end());
		std::list<RefFrame*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0; i<vPairs.size(); i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		{
			std::unique_lock<std::mutex> lock(mMutexConnections);

			// mspConnectedKeyFrames = spConnectedKeyFrames;
			mConnectedKeyFrameWeights = KFcounter;
			mvpOrderedConnectedKeyFrames = std::vector<RefFrame*>(lKFs.begin(), lKFs.end());
			mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

			if (mbFirstConnection && mnId != 0)
			{
				mpParent = mvpOrderedConnectedKeyFrames.front();
				mpParent->AddChild(this);
				mbFirstConnection = false;
			}

		}
	}
	////Covisibility
	void RefFrame::AddConnection(RefFrame *pKF, const int &weight)
	{
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (!mConnectedKeyFrameWeights.count(pKF))
				mConnectedKeyFrameWeights[pKF] = weight;
			else if (mConnectedKeyFrameWeights[pKF] != weight)
				mConnectedKeyFrameWeights[pKF] = weight;
			else
				return;
		}

		UpdateBestCovisibles();
	}
	void RefFrame::EraseConnection(RefFrame* pKF)
	{
		bool bUpdate = false;
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mConnectedKeyFrameWeights.count(pKF))
			{
				mConnectedKeyFrameWeights.erase(pKF);
				bUpdate = true;
			}
		}

		if (bUpdate)
			UpdateBestCovisibles();
	}
	void RefFrame::UpdateBestCovisibles()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		std::vector<std::pair<int, RefFrame*> > vPairs;
		vPairs.reserve(mConnectedKeyFrameWeights.size());
		for (std::map<RefFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
			vPairs.push_back(std::make_pair(mit->second, mit->first));

		sort(vPairs.begin(), vPairs.end());
		std::list<RefFrame*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0, iend = vPairs.size(); i<iend; i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		mvpOrderedConnectedKeyFrames = std::vector<RefFrame*>(lKFs.begin(), lKFs.end());
		mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
	}

	std::set<RefFrame*> RefFrame::GetConnectedKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		std::set<RefFrame*> s;
		for (std::map<RefFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
			s.insert(mit->first);
		return s;
	}

	std::vector<RefFrame*> RefFrame::GetVectorCovisibleKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mvpOrderedConnectedKeyFrames;
	}

	std::vector<RefFrame*> RefFrame::GetBestCovisibilityKeyFrames(const int &N)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		if ((int)mvpOrderedConnectedKeyFrames.size()<N)
			return mvpOrderedConnectedKeyFrames;
		else
			return std::vector<RefFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);

	}

	std::vector<RefFrame*> RefFrame::GetCovisiblesByWeight(const int &w)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);

		if (mvpOrderedConnectedKeyFrames.empty())
			return std::vector<RefFrame*>();

		std::vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, RefFrame::weightComp);
		if (it == mvOrderedWeights.end())
			return std::vector<RefFrame*>();
		else
		{
			int n = it - mvOrderedWeights.begin();
			return std::vector<RefFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
		}
	}

	int RefFrame::GetWeight(RefFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		if (mConnectedKeyFrameWeights.count(pKF))
			return mConnectedKeyFrameWeights[pKF];
		else
			return 0;
	}

	void RefFrame::AddChild(RefFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mspChildrens.insert(pKF);
	}
	std::set<RefFrame*> RefFrame::GetChilds()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspChildrens;
	}

	RefFrame* RefFrame::GetParent()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mpParent;
	}

	std::vector<MapPoint*> RefFrame::GetMapPointMatches()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mvpMapPoints;
	}

	int RefFrame::TrackedMapPoints(const int &minObs)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);

		int nPoints = 0;
		const bool bCheckObs = minObs>0;
		for (int i = 0; i<N; i++)
		{
			MapPoint* pMP = mvpMapPoints[i];
			if (pMP)
			{
				if (bCheckObs)
				{
					if (mvpMapPoints[i]->Observations() >= minObs)
						nPoints++;
				}
				else
					nPoints++;
			}
		}

		return nPoints;
	}

	void RefFrame::SetPose(const cv::Mat &Tcw) {
		mpCamPose->SetPose(Tcw);
	}
	cv::Mat RefFrame::GetPose() {
		return mpCamPose->GetPose();
	}
	cv::Mat RefFrame::GetPoseInverse() {
		return mpCamPose->GetInversePose();
	}
	cv::Mat RefFrame::GetCameraCenter() {
		return mpCamPose->GetCenter();
	}
	cv::Mat RefFrame::GetRotation() {
		return mpCamPose->GetRotation();
	}
	cv::Mat RefFrame::GetTranslation() {
		return mpCamPose->GetTranslation();
	}

    void RefFrame::UndistortKeyPoints() {
		cv::Mat mat(N, 2, CV_32F);
		for (int i = 0; i<N; i++)
		{
			mat.at<float>(i, 0) = mvKeys[i].pt.x;
			mat.at<float>(i, 1) = mvKeys[i].pt.y;
		}

		// Undistort points
		mat = mat.reshape(2);
		cv::undistortPoints(mat, mat, K, D, cv::Mat(), K);
		mat = mat.reshape(1);

		// Fill undistorted keypoint vector
		mvKeysUn.resize(N);
		for (int i = 0; i<N; i++)
		{
			cv::KeyPoint kp = mvKeys[i];
			kp.pt.x = mat.at<float>(i, 0);
			kp.pt.y = mat.at<float>(i, 1);
			mvKeysUn[i] = kp;
		}
	}
	void RefFrame::AssignFeaturesToGrid() {
		int nReserve = 0.5f*N / (FRAME_GRID_COLS*FRAME_GRID_ROWS);
		for (unsigned int i = 0; i<FRAME_GRID_COLS; i++)
			for (unsigned int j = 0; j<FRAME_GRID_ROWS; j++)
				mGrid[i][j].reserve(nReserve);

		for (int i = 0; i<N; i++)
		{
			const cv::KeyPoint &kp = mvKeysUn[i];

			int nGridPosX, nGridPosY;
			if (PosInGrid(kp, nGridPosX, nGridPosY))
				mGrid[nGridPosX][nGridPosY].push_back(i);
		}
	}
	bool RefFrame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
		posX = round((kp.pt.x - mnMinX)*mfGridElementWidthInv);
		posY = round((kp.pt.y - mnMinY)*mfGridElementHeightInv);

		if (posX<0 || posX >= FRAME_GRID_COLS || posY<0 || posY >= FRAME_GRID_ROWS)
			return false;

		return true;
	}

}