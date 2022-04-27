//
// Created by wiseuiux on 2022-03-02.
//

#ifndef EDGESLAMNDK_PLANEPROCESSOR_H
#define EDGESLAMNDK_PLANEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "Utils.h"

namespace EdgeSLAM {

    class PluckerLine{
        public:
            PluckerLine();
            PluckerLine(cv::Mat _Lw);
            PluckerLine(cv::Mat p1, cv::Mat p2);
            virtual ~PluckerLine();
        public:
            cv::Mat D, N;
    };

    class PluckerPlane{
        public:
            PluckerPlane();
            virtual ~PluckerPlane();
            cv::Mat param;
            std::vector<PluckerLine*> vecLines;
            bool CheckCollision(PluckerLine* ray);
        private:
            float CheckSide(PluckerLine* ray, PluckerLine* line);
    };

    class PlaneProcessor {
        public:
            static cv::Mat CalcFlukerLine(const cv::Mat& P1, const cv::Mat& P2) ;
            static cv::Mat LineProjection(const cv::Mat& R, const cv::Mat& t, const cv::Mat& Lw1, const cv::Mat& K);
            static cv::Point2f GetLinePoint(float val, const cv::Mat& mLine);

            static cv::Mat CreateWorldPoint(const cv::Mat& Xcam, const cv::Mat& Tinv, float depth);
            static float CalculateDepth(const cv::Mat& Xcam, const cv::Mat& Pinv);
            static cv::Mat CalcInverPlaneParam(const cv::Mat& P, const cv::Mat& Tinv);

    };
}


#endif //EDGESLAMNDK_PLANEPROCESSOR_H
