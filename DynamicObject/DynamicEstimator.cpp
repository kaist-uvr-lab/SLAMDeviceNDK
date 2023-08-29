#include "DynamicEstimator.h"
#include "DynamicFrame.h"
#include "PnPProblem.h"
DynamicEstimator::DynamicEstimator(){
    pPnPSolver = new PnPProblem();
}
DynamicEstimator::~DynamicEstimator(){

}
int DynamicEstimator::SearchPointsByOpticalFlow(DynamicFrame* pRef, DynamicFrame* pCur){

    int win_size = 10;
    std::vector<cv::Point2f> imagePoints;
    std::vector<uchar> inliers;
    //std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    cv::calcOpticalFlowPyrLK(
        pRef->image,                         // Previous image
        pCur->image,                         // Next image
        pRef->imagePoints,                     // Previous set of corners (from imgA)
        imagePoints,                     // Next set of corners (from imgB)
        inliers,               // Output vector, each is 1 for tracked
        cv::noArray(),                // Output vector, lists errors (optional)
        cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
        5,                            // Maximum pyramid level to construct
        cv::TermCriteria(
            cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
            20,                         // Maximum number of iterations
            0.3                         // Minimum change per iteration
        )
    );

    int nGood = 0;
    for (int i = 0, N = inliers.size(); i < N; ++i) {
        if (!inliers[i]) {
            continue;
        }
        pCur->objectPoints.push_back(pRef->objectPoints[i]);
        pCur->imagePoints.push_back(imagePoints[i]);
        pCur->inliers.push_back(true);
        nGood++;
    }
    pCur->Pose = pRef->Pose.clone();
    return nGood;
}
int DynamicEstimator::DynamicPoseEstimation(DynamicFrame* pCur){
    //SolvePnP Method
    int pnpMethod = cv::SOLVEPNP_EPNP;

    // RANSAC parameters
    int iterationsCount = 1000;      // number of Ransac iterations.
    float reprojectionError = 8.0;  // maximum allowed distance to consider it an inlier.
    double confidence = 0.9;       // ransac successful confidence.

    // Kalman Filter parameters
    int minInliersKalman = 15;    // Kalman threshold updating
    int nMeasurements = 6;
    cv::Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
    bool good_measurement = false;

    cv::Mat R = pCur->Pose.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = pCur->Pose.rowRange(0, 3).col(3);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Rodrigues(R, rvec);
    rvec.convertTo(rvec, CV_64FC1);
    t.convertTo(tvec, CV_64FC1);

    // -- Step 3: Estimate the pose using RANSAC approach
    cv::Mat inliers_idx;
    pPnPSolver->estimatePoseRANSAC(pCur->objectPoints, pCur->imagePoints,
        pCur->K, rvec, tvec, pnpMethod, inliers_idx,
        iterationsCount, reprojectionError, confidence, true);
    return inliers_idx.rows;
}
