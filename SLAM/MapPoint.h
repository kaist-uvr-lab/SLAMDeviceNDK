#ifndef UNITY_LIBRARY_MAPPOINT_H
#define UNITY_LIBRARY_MAPPOINT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace EdgeSLAM {
	class RefFrame;
	class Frame;
	class ORBDetector;
	class TrackPoint {
	public:
		TrackPoint();
		TrackPoint(float x, float y, float angle, float scale);
		virtual ~TrackPoint();
	public:
		float mTrackProjX;
		float mTrackProjY;
		float mTrackProjXR;
		bool mbTrackInView;
		int mnTrackScaleLevel;
		float mTrackViewCos;
		long unsigned int mnTrackReferenceForFrame;
		long unsigned int mnLastFrameSeen;

	private:
	};
	class MapPoint {
	public:
		MapPoint();
		MapPoint(int id, float _x, float _y, float _z);
		virtual ~MapPoint();

	public:

		//void SetWorldPos(float x, float y, float z);
		cv::Mat GetWorldPos();
		cv::Mat GetDescriptor();
        int GetScale();
        float GetAngle();
        int GetObservation();
        void Update(cv::Mat _pos, cv::Mat _desc, float _angle, int _scale, int _obs);
     private:
        std::mutex mMutexMP;
        cv::Mat mWorldPos, mDescriptor;
        float mfAngle;
        int mnScale;
        RefFrame* mpRefKF;
        int mnObservation;
    public:
	    int mnID;
		static ORBDetector* Detector;
        /*
		cv::Mat GetNormal();
		std::map<RefFrame*, size_t> GetObservations();
		int Observations();
		void AddObservation(RefFrame* pKF, size_t idx);
		void EraseObservation(RefFrame* pKF);
		void ComputeDistinctiveDescriptors();


		void UpdateNormalAndDepth();
		float GetMinDistanceInvariance();
		float GetMaxDistanceInvariance();
		int PredictScale(const float &currentDist, Frame* pF);

		bool IsInKeyFrame(RefFrame *pKF);
		void SetReferenceFrame(RefFrame* pRef);
		RefFrame* GetReferenceFrame();

	private:
		cv::Mat mWorldPos, mNormalVector;
		std::map<RefFrame*, size_t> mObservations;
		cv::Mat mDescriptor;
		

		int nObs;

		float mfMinDistance;
		float mfMaxDistance;

		std::mutex mMutexPos;
		std::mutex mMutexFeatures;
		*/
	};
}
#endif