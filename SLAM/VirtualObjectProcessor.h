//
// Created by wiseuiux on 2022-05-25.
//

#ifndef EDGESLAMNDK_VIRTUALOBJECTPROCESSOR_H
#define EDGESLAMNDK_VIRTUALOBJECTPROCESSOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "ConcurrentMap.h"
namespace EdgeSLAM{
    enum class VOManageState {
		None, Registration, Manipulation
	};
    class VirtualObject{
        public:
        VirtualObject();
        VirtualObject(int _model, int _id, const cv::Mat& _pos, int _TTL);
        VirtualObject(int _model, int _id, int _nid, const cv::Mat& _pos, const cv::Mat& _epos, bool bPath, int _TTL);
        virtual~ VirtualObject();
        void Update(cv::Mat& _pos, int _TTL);
        void Update(cv::Mat& _pos, cv::Mat& _epos, int _nid, bool bPath, int _TTL);
        cv::Mat UpdateWorldPos(float t);
        void Set();
        bool Check(float t);
        void Reset();
        public:

        int mnId, mnNextId;
        bool mbPath;
        bool mbMoving;
        int mnModelCategory;
        cv::Mat pos, epos, dir;
        bool mbSelected;
        int mnTTL;
        float mfTotalTime;
        std::chrono::high_resolution_clock::time_point time_start;
    };

    class VirtualObjectProcessor {
        public:
            VirtualObjectProcessor();
            virtual ~VirtualObjectProcessor();
            void Reset();
        public:
            VOManageState VOState;
            VirtualObject* LastObject;
            ConcurrentMap<int, VirtualObject*> VOManageMap;
    };
}




#endif //EDGESLAMNDK_VIRTUALOBJECTPROCESSOR_H
