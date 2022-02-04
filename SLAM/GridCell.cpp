#include "GridCell.h"

namespace EdgeSLAM {
	GridCell::GridCell(){
	    //mpObject = new Label();
	    //mpSegLabel = new Label();
	}
    GridCell::~GridCell(){
        //delete mpObject;
        //delete mpSegLabel;
        //mapObservation.Release();
        std::map<int,int>().swap(mapObjectID);
        std::map<int,int>().swap(mapSegLabel);
    }

	Label::Label() :mnLabel(0), mnCount(0) {
		matLabels = cv::Mat::zeros(200, 1, CV_16UC1);
	}
	Label::Label(int n) : mnLabel(0), mnCount(0) {
		matLabels = cv::Mat::zeros(n, 1, CV_16UC1);
	}
	Label::~Label() {
		matLabels.release();
	}

	void Label::Update(int nLabel) {

		std::unique_lock<std::mutex> lock(mMutexObject);
		matLabels.at<ushort>(nLabel)++;
		if (mnLabel == nLabel) {
			mnCount++;
		}
		else {
			int count = matLabels.at<ushort>(nLabel);
			if (count > mnCount) {
				double minVal;
				double maxVal;
				int minIdx, maxIdx;
				cv::minMaxIdx(matLabels, &minVal, &maxVal, &minIdx, &maxIdx, cv::Mat());
				mnLabel = maxIdx;
				mnCount = matLabels.at<ushort>(maxIdx);
			}
		}
	}
	int Label::GetLabel() {
		std::unique_lock<std::mutex> lock(mMutexObject);
		return mnLabel;
	}
	cv::Mat Label::GetLabels() {
		std::unique_lock<std::mutex> lock(mMutexObject);
		return matLabels.clone();
	}
	int Label::Count(int l) {
		std::unique_lock<std::mutex> lock(mMutexObject);
		return matLabels.at<ushort>(l);
	}
/*
	void GridCell::AddObservation(GridFrame* pGF, int idx){
		mapObservation.Update(pGF, idx);
	}
	void GridCell::EraseObservation(GridFrame* pGF){
		mapObservation.Erase(pGF);
		if (mapObservation.Size() == 0)
		{
			SetBadFlag();
		}
	}
	void GridCell::SetBadFlag(){
		mbBad = true;
	}
	bool GridCell::isBad(){
		return mbBad.load();
	}
*/
}