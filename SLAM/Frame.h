#ifndef UNITY_LIBRARY_FRAME_H
#define UNITY_LIBRARY_FRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "DBoW3.h"
#include "Utils.h"
#include <mutex>

namespace EdgeSLAM {
	class Camera;
	class CameraPose;
	class ORBDetector;
	class MapPoint;
	class Frame {
	public:
		Frame();
		Frame(cv::Mat img, Camera* pCam, int id, double time_stamp = 0.0);
		Frame(void* data, Camera* pCam, int id, double time_stamp = 0.0);
		Frame(Color* data, Camera* pCam, int id, double time_stamp = 0.0);
		Frame(Color32* data, Camera* pCam, int id, double time_stamp = 0.0);
		virtual ~Frame();

		void reset_map_points();
		bool is_in_frustum(MapPoint* pMP, float viewingCosLimit);

		int N;
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
		Camera* mpCamera;
		static ORBDetector* detector;
		double mdTimeStamp;
		//cv::Mat imgColor, imgGray;
		int mnFrameID;
		std::vector<cv::KeyPoint> mvKeys;
		std::vector<cv::KeyPoint> mvKeysUn;
		std::vector<MapPoint*>  mvpMapPoints;
		std::set<MapPoint*> mspMapPoints;
		
		std::vector<bool> mvbOutliers;
		cv::Mat mDescriptors;
		DBoW3::BowVector mBowVec;
		DBoW3::FeatureVector mFeatVec;
	public:
		std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel = -1, const int maxLevel = -1) const;
	private:
		void UndistortKeyPoints();
		void AssignFeaturesToGrid();
		bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
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