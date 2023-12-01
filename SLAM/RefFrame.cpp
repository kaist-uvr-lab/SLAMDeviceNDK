#include "RefFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "Camera.h"
#include "CameraPose.h"
#include "ORBDetector.h"

namespace EdgeSLAM {
	int RefFrame::nId = 0;
	RefFrame::RefFrame() {}
    RefFrame::RefFrame(Camera* pCam, float* data) :mpCamera(pCam), mnId(RefFrame::nId++),
    		K(pCam->K), D(pCam->D), fx(pCam->fx), fy(pCam->fy), cx(pCam->cx), cy(pCam->cy), invfx(pCam->invfx), invfy(pCam->invfy), mnMinX(pCam->u_min), mnMaxX(pCam->u_max), mnMinY(pCam->v_min), mnMaxY(pCam->v_max), mfGridElementWidthInv(pCam->mfGridElementWidthInv), mfGridElementHeightInv(pCam->mfGridElementHeightInv), FRAME_GRID_COLS(pCam->mnGridCols), FRAME_GRID_ROWS(pCam->mnGridRows), mbDistorted(pCam->bDistorted),
    		mnScaleLevels(detector->mnScaleLevels), mfScaleFactor(detector->mfScaleFactor), mfLogScaleFactor(detector->mfLogScaleFactor), mvScaleFactors(detector->mvScaleFactors), mvInvScaleFactors(detector->mvInvScaleFactors), mvLevelSigma2(detector->mvLevelSigma2), mvInvLevelSigma2(detector->mvInvLevelSigma2)
    {
        N = (int)data[0]; //kf

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

        int nIdx = 13;
        for (int i = 0; i < N; i++) {
            cv::KeyPoint kp;
            kp.pt.x = data[nIdx++];
            kp.pt.y = data[nIdx++];
            kp.octave = (int)data[nIdx++];
            kp.angle = data[nIdx++];
            int id = (int)data[nIdx++];
            int label = (int)data[nIdx++];
            float x = data[nIdx++];
            float y = data[nIdx++];
            float z = data[nIdx++];

            //if(kp.octave >= detector->mnScaleLevels)
            //    continue;

            MapPoint* pMP = nullptr;
            if(MAP->MapPoints.Count(id)){
                pMP = MAP->MapPoints.Get(id);
                pMP->SetWorldPos(x,y,z);
            }else{
                pMP = new MapPoint(id, x, y, z);
                MAP->MapPoints.Update(id, pMP);
            }
            pMP->mpRefKF = this;
            mvpMapPoints[i] = pMP;
            mvKeys[i] = kp;
        }

        if (mbDistorted)
            UndistortKeyPoints();
        else
            mvKeysUn = mvKeys;

    }
	RefFrame::RefFrame(Camera* pCam, cv::Mat desc, float* data) :mpCamera(pCam), mnId(RefFrame::nId++),
    		K(pCam->K), D(pCam->D), fx(pCam->fx), fy(pCam->fy), cx(pCam->cx), cy(pCam->cy), invfx(pCam->invfx), invfy(pCam->invfy), mnMinX(pCam->u_min), mnMaxX(pCam->u_max), mnMinY(pCam->v_min), mnMaxY(pCam->v_max), mfGridElementWidthInv(pCam->mfGridElementWidthInv), mfGridElementHeightInv(pCam->mfGridElementHeightInv), FRAME_GRID_COLS(pCam->mnGridCols), FRAME_GRID_ROWS(pCam->mnGridRows), mbDistorted(pCam->bDistorted),
    		mnScaleLevels(detector->mnScaleLevels), mfScaleFactor(detector->mfScaleFactor), mfLogScaleFactor(detector->mfLogScaleFactor), mvScaleFactors(detector->mvScaleFactors), mvInvScaleFactors(detector->mvInvScaleFactors), mvLevelSigma2(detector->mvLevelSigma2), mvInvLevelSigma2(detector->mvInvLevelSigma2)
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
            int label = (int)data[nIdx++];
            float x = data[nIdx++];
            float y = data[nIdx++];
            float z = data[nIdx++];

            MapPoint* pMP = nullptr;
            if(MAP->MapPoints.Count(id)){
                pMP = MAP->MapPoints.Get(id);
                pMP->SetWorldPos(x,y,z);
            }else{
                pMP = new MapPoint(id, x, y, z);
                MAP->MapPoints.Update(id, pMP);
            }
            mvpMapPoints[i] = pMP;
            //pMP->SetReferenceFrame(this);
            mvKeys[i] = kp;
        }
        mDescriptors = desc.clone();

        if (mbDistorted)
            UndistortKeyPoints();
        else
            mvKeysUn = mvKeys;

    }

	RefFrame::~RefFrame() {
	    /*
	    std::ofstream ofile;
        ofile.open(logfile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<"delete kf"<<std::endl;
        for (size_t i = 0; i<mvpMapPoints.size(); i++)
		{
            MapPoint* pMP = mvpMapPoints[i];
			if (pMP && !pMP->isBad())
			{
                pMP->EraseObservation(this);
			}
		}
        ofile.close();
        */

        std::vector<float>().swap(mvScaleFactors);
        std::vector<float>().swap(mvInvScaleFactors);
        std::vector<float>().swap(mvLevelSigma2);
        std::vector<float>().swap(mvInvLevelSigma2);
        std::vector<cv::KeyPoint>().swap(mvKeys);
        std::vector<cv::KeyPoint>().swap(mvKeysUn);
        std::vector<MapPoint*>().swap(mvpMapPoints);
        std::vector<bool>().swap(mvbOutliers);
	}

    void RefFrame::UpdateMapPoints(){

		for (size_t i = 0; i<mvpMapPoints.size(); i++)
		{
			MapPoint* pMP = mvpMapPoints[i];
			if (pMP && !pMP->isBad())
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
    void RefFrame::EraseMapPointMatch(const size_t &idx)
	{
	    mvpMapPoints[idx] = nullptr;
	}

	bool RefFrame::is_in_frustum(MapPoint* pMP, float viewingCosLimit) {

        cv::Mat P = pMP->GetWorldPos();
        cv::Mat Rw = mpCamPose->GetRotation();
        cv::Mat tw = mpCamPose->GetTranslation();
        cv::Mat Ow = mpCamPose->GetCenter();

        // 3D in camera coordinates
        const cv::Mat Pc = Rw*P + tw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if (PcZ<0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx*PcX*invz + cx;
        const float v = fy*PcY*invz + cy;

        if (u<mnMinX || u>mnMaxX || v < mnMinY || v > mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - Ow;
        const float dist = cv::norm(PO);

        if (dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();
        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos<viewingCosLimit)
            return false;


        //// Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

	void RefFrame::SetPose(cv::Mat Tcw) {
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
}