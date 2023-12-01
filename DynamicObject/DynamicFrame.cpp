#include "DynamicFrame.h"

DynamicFrame::DynamicFrame(){

}
DynamicFrame::DynamicFrame(const cv::Mat& _img, const cv::Mat& _K){
    image = _img.clone();
    _K.convertTo(K, CV_64FC1);
}
DynamicFrame::DynamicFrame(int id, const cv::Mat& _img, std::vector<cv::Point2f>& _imgPts, std::vector<cv::Point3f>& _objPts, const cv::Mat& _Pose, const cv::Mat& _K):mnObjectId(id){
    image = _img.clone();
    imagePoints = std::vector<cv::Point2f>(_imgPts.begin(), _imgPts.end());
    objectPoints = std::vector<cv::Point3f>(_objPts.begin(), _objPts.end());
    Pco = _Pose.clone();
    _K.convertTo(K, CV_64FC1);
}
DynamicFrame::~DynamicFrame(){

}