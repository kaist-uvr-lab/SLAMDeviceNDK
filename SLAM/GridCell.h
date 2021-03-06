#ifndef SEMANTIC_SLAM_GRID_CELL_H
#define SEMANTIC_SLAM_GRID_CELL_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "ConcurrentMap.h"
#include <atomic>

namespace EdgeSLAM {

	class GridFrame;
	class Label {
	public:
		Label();
		Label(int n);
		virtual ~Label();
		void Update(int label);
		int GetLabel();
		cv::Mat GetLabels();
		int Count(int label);

	private:
		int mnLabel;
		int mnCount;
		cv::Mat matLabels;
		std::mutex mMutexObject;
	};

	class GridCell {
	public:
		GridCell();
		virtual ~GridCell();
	public:
	    cv::Point2f pt;
	    std::map<int, int> mapObjectID;
	    std::map<int, int> mapSegLabel;
		//void AddObservation(GridFrame* pGF, int idx);
		//void EraseObservation(GridFrame* pGF);
		//void SetBadFlag();
		//bool isBad();
	public:
		//std::atomic<bool> mbBad;
		//ConcurrentMap<GridFrame*, int> mapObservation;
		//Label *mpObject, *mpSegLabel;
	};

}


#endif