
#ifndef DYNAMIC_FRAME_H
#define DYNAMIC_FRAME_H

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class DynamicFrame
{
public:
    DynamicFrame();
    DynamicFrame(const cv::Mat& _img, const cv::Mat& _K);
    DynamicFrame(const cv::Mat& _img, std::vector<cv::Point2f>& _imgPts, std::vector<cv::Point3f>& _objPts, const cv::Mat& _Pose, const cv::Mat& _K);
    virtual ~DynamicFrame();

    //frame id
    //object id
    //object label
    cv::Mat image;
    int mnFrameId;
    int mnObjectId;
    int mnObjectLabel;
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point3f> worldPoints;
    std::vector<bool> inliers;
    cv::Mat Pco;
    cv::Mat K;
private:
    //2f
    //3f
    //image
};

#endif /* PNPPROBLEM_H_ */
