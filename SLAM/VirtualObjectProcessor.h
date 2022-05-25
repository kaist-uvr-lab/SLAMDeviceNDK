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
        virtual~ VirtualObject();
        public:
        int mnId;
        int mnModelCategory;
        cv::Mat pos;
        bool mbSelected;
        int mnTTL;
    };

    class VirtualObjectProcessor {
        public:
            VirtualObjectProcessor();
            virtual ~VirtualObjectProcessor();
            void Reset();
        public:
            VirtualObject* LastObject;
            ConcurrentMap<int, VirtualObject*> VOManageMap;
    };
}




#endif //EDGESLAMNDK_VIRTUALOBJECTPROCESSOR_H
