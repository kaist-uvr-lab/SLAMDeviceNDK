//
// Created by wiseuiux on 2022-05-25.
//
#include "VirtualObjectProcessor.h"

namespace EdgeSLAM{
        VirtualObject::VirtualObject(){}
        VirtualObject::VirtualObject(int _model, int _id, const cv::Mat& _pos, int _TTL):mnId(id), mnModelCategory(_model), mnTTL(_TTL), pos(_pos){}
        VirtualObject::~VirtualObject(){
            pos.release();
        }
        VirtualObjectProcessor::VirtualObjectProcessor(){
            LastObject = nullptr;
        }
        void VirtualObjectProcessor::Reset(){

        }
        VirtualObjectProcessor::~VirtualObjectProcessor(){
            if(LastObject)
                delete LastObject;
            auto mapVOs = VOManageMap.Get();
            for(auto iter = mapVOs.begin(), iend = mapVOs.end(); iter != iend; iter++)
                delete iter->second;
            VOManageMap.Release();
        }
}