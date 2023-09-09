
#ifndef DYNAMIC_OBJECT_MAP_H
#define DYNAMIC_OBJECT_MAP_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>

class KalmanFilter;

class DynamicObjectMap
{
public:
    DynamicObjectMap();
    virtual ~DynamicObjectMap();
public:
    void SetPose(cv::Mat aP);
    cv::Mat GetPose();
public:
    KalmanFilter* mpKalmanFilter;
private:
    std::mutex mMutexPose;
    cv::Mat P;

};

#endif /* PNPPROBLEM_H_ */
