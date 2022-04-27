//
// Created by wiseuiux on 2022-03-02.
//

#include "PlaneProcessor.h"

namespace EdgeSLAM{
    PluckerLine::PluckerLine(){}
    PluckerLine::PluckerLine(cv::Mat _Lw){
        N = _Lw.rowRange(0,3);
        D = _Lw.rowRange(3,6);
    }
    PluckerLine::PluckerLine(cv::Mat p1, cv::Mat p2){
        D = p2-p1;
        N = p2.cross(p1);
    }
    PluckerLine::~PluckerLine(){
        D.release();
        N.release();
    }
    PluckerPlane::PluckerPlane(){}
    PluckerPlane::~PluckerPlane(){
        std::vector<PluckerLine*>().swap(vecLines);
    }
    bool PluckerPlane::CheckCollision(PluckerLine* ray){
        int res1 = 0;
        int res2 = 0;
        for(int i = 0; i < vecLines.size(); i++){
            float val = CheckSide(ray, vecLines[i]);
            if(val >= 0)
                res1++;
            if(val <= 0)
                res2++;
        }
        if(res1 == vecLines.size() || res2 == vecLines.size())
            return true;
        return false;
    }
    float PluckerPlane::CheckSide(PluckerLine* ray, PluckerLine* line){
        float a = ray->D.dot(line->N) + ray->N.dot(line->D);
        if(a>0.0)
            a = 1.0;
        if(a < 0.0)
            a = -1.0;
        return a;

    }

    cv::Mat PlaneProcessor::CalcFlukerLine(const cv::Mat& P1, const cv::Mat& P2) {
		cv::Mat PLw1, Lw1, NLw1;
		PLw1 = P1*P2.t() - P2*P1.t();
		Lw1 = cv::Mat::zeros(6, 1, CV_32FC1); //0 - 2 : N, 3 ~ 5 : D
		Lw1.at<float>(3) = PLw1.at<float>(2, 1);
		Lw1.at<float>(4) = PLw1.at<float>(0, 2);
		Lw1.at<float>(5) = PLw1.at<float>(1, 0);
		NLw1 = PLw1.col(3).rowRange(0, 3);
		NLw1.copyTo(Lw1.rowRange(0, 3));

		return Lw1;
	}
    cv::Mat PlaneProcessor::LineProjection(const cv::Mat& R, const cv::Mat& t, const cv::Mat& Lw1, const cv::Mat& K) {
		cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
		R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
		R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
		cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
		tempSkew.at<float>(0, 1) = -t.at<float>(2);
		tempSkew.at<float>(1, 0) = t.at<float>(2);
		tempSkew.at<float>(0, 2) = t.at<float>(1);
		tempSkew.at<float>(2, 0) = -t.at<float>(1);
		tempSkew.at<float>(1, 2) = -t.at<float>(0);
		tempSkew.at<float>(2, 1) = t.at<float>(0);
		tempSkew *= R;
		tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
		cv::Mat Lc = T2*Lw1;
		cv::Mat Nc = Lc.rowRange(0, 3);
		cv::Mat res = K*Nc;
		float a = res.at<float>(0);
		float b = res.at<float>(1);
		float d = sqrt(a*a + b*b);
		res /= d;
		return res.clone();
	}
	cv::Point2f PlaneProcessor::GetLinePoint(float val, const cv::Mat& mLine) {
        float x, y;
        y = 0.0;
        x = val;
        if (mLine.at<float>(1) != 0)
            y = (-mLine.at<float>(2) - mLine.at<float>(0)*x) / mLine.at<float>(1);
        return cv::Point2f(x, y);
    }
    cv::Mat PlaneProcessor::CreateWorldPoint(const cv::Mat& Xcam, const cv::Mat& Tinv, float depth) {
        if (depth <= 0.0) {
        }
        cv::Mat X = Xcam*depth;
        X.push_back(cv::Mat::ones(1, 1, CV_32FC1));
        cv::Mat estimated = Tinv*X;
        return estimated.rowRange(0, 3);
    }
    float PlaneProcessor::CalculateDepth(const cv::Mat& Xcam, const cv::Mat& Pinv) {
        float depth = -Pinv.at<float>(3) / Xcam.dot(Pinv.rowRange(0, 3));
        return depth;
    }
    cv::Mat PlaneProcessor::CalcInverPlaneParam(const cv::Mat& P, const cv::Mat& Tinv) {
        return Tinv.t()*P;
    }
}