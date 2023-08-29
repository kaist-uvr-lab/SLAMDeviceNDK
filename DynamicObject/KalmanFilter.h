
#ifndef DYNAMIC_ESTIMATOR_KALMAN_FILTER_H
#define DYNAMIC_ESTIMATOR_KALMAN_FILTER_H

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

class KalmanFilter
{
public:
    KalmanFilter();
    virtual ~KalmanFilter();

public:
    void initKalmanFilter(cv::KalmanFilter& KF, int nStates, int nMeasurements, int nInputs, double dt);
    void updateKalmanFilter(cv::KalmanFilter& KF, cv::Mat& measurement, cv::Mat& translation_estimated, cv::Mat& rotation_estimated);
    void fillMeasurements(cv::Mat& measurements, const cv::Mat& translation_measured, const cv::Mat& rotation_measured);
private:
    cv::Mat euler2rot(const cv::Mat& euler);
    cv::Mat rot2euler(const cv::Mat& rotationMatrix);
};

#endif /* PNPPROBLEM_H_ */
