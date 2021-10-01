#ifndef UNITY_LIBRARY_REFFRAME_H
#define UNITY_LIBRARY_REFFRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "DBoW3.h"
#include "Utils.h"
#include <mutex>

namespace EdgeSLAM {
	class Map;
	class Camera;
	class CameraPose;
	class ORBDetector;
	class MapPoint;
	class RefFrame {
	public:
		RefFrame();
		RefFrame(Map* map,cv::Mat img, Camera* pCam, float* data);
		virtual ~RefFrame();
	public:
		int TrackedMapPoints(const int &minObs);
		std::vector<MapPoint*> GetMapPointMatches();
		void AddConnection(RefFrame* pKF, const int &weight);
		void EraseConnection(RefFrame* pKF);
		void UpdateConnections();
		void UpdateBestCovisibles();
		std::set<RefFrame *> GetConnectedKeyFrames();
		std::vector<RefFrame* > GetVectorCovisibleKeyFrames();
		std::vector<RefFrame*> GetBestCovisibilityKeyFrames(const int &N);
		std::vector<RefFrame*> GetCovisiblesByWeight(const int &w);
		int GetWeight(RefFrame* pKF);
		void AddChild(RefFrame *pKF);
		std::set<RefFrame*> GetChilds();
		RefFrame* GetParent();
		void UpdateMapPoints();
	public:
		static bool weightComp(int a, int b) {
			return a>b;
		}

		static bool lId(RefFrame* pKF1, RefFrame* pKF2) {
			return pKF1->mnId<pKF2->mnId;
		}
	public:
		int N;
		int mnId;
		//cv::Mat imgGray;
		static int nId;
		cv::Mat K, D;
		float fx, fy, cx, cy, invfx, invfy;
		bool mbDistorted;
		int FRAME_GRID_COLS;
		int FRAME_GRID_ROWS;
		float mfGridElementWidthInv;
		float mfGridElementHeightInv;
		std::vector<std::size_t> **mGrid;

		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;

		float mnMinX;
		float mnMaxX;
		float mnMinY;
		float mnMaxY;
	public:
		std::map<RefFrame*, int> mConnectedKeyFrameWeights;
		std::vector<RefFrame*> mvpOrderedConnectedKeyFrames;
		std::vector<int> mvOrderedWeights;
		bool mbFirstConnection;
		RefFrame* mpParent;
		std::set<RefFrame*> mspChildrens;
	public:
		Camera* mpCamera;
		static ORBDetector* detector;

		std::vector<cv::KeyPoint> mvKeys,mvKeysUn;
		std::vector<bool> mvbOutliers;
		std::vector<MapPoint*> mvpMapPoints;
		cv::Mat mDescriptors;
		DBoW3::BowVector mBowVec;
		DBoW3::FeatureVector mFeatVec;
	private:
    		void UndistortKeyPoints();
    		void AssignFeaturesToGrid();
    		bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
	private:
		std::mutex mMutexFeatures, mMutexConnections;
	public:
		CameraPose* mpCamPose;
		void SetPose(const cv::Mat &Tcw);
		cv::Mat GetPose();
		cv::Mat GetPoseInverse();
		cv::Mat GetCameraCenter();
		cv::Mat GetRotation();
		cv::Mat GetTranslation();
	};
}
#endif