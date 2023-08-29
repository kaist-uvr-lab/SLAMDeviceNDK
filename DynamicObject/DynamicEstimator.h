
#ifndef DYNAMIC_ESTIMATOR_H
#define DYNAMIC_ESTIMATOR_H

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

class DynamicFrame;
class PnPProblem;

class DynamicEstimator
{
public:
    DynamicEstimator();
    virtual ~DynamicEstimator();

public:
    int SearchPointsByOpticalFlow(DynamicFrame* pRef, DynamicFrame* pCur);
    int DynamicPoseEstimation(DynamicFrame* pCur);
private:
    PnPProblem* pPnPSolver;
    PnPProblem* pPnPSolver_est;
    //2f
    //3f
    //image
};

#endif /* PNPPROBLEM_H_ */
