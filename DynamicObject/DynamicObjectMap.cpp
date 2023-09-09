#include "DynamicObjectMap.h"
#include "KalmanFilter.h"

DynamicObjectMap::DynamicObjectMap() {
    P = cv::Mat::eye(4, 4, CV_32FC1);
    int nStates = 18;            // the number of states
    int nMeasurements = 6;       // the number of measured states
    int nInputs = 0;             // the number of control actions
    double dt = 0.125;           // time between measurements (1/FPS)
    mpKalmanFilter = new KalmanFilter(nStates, nMeasurements, nInputs, dt);
}
DynamicObjectMap::~DynamicObjectMap(){

}
void DynamicObjectMap::SetPose(cv::Mat aP){
    std::unique_lock<std::mutex> lock(mMutexPose);
    P = aP.clone();
}
cv::Mat DynamicObjectMap::GetPose() {
    std::unique_lock<std::mutex> lock(mMutexPose);
    return P.clone();
}