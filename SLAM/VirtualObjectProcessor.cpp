//
// Created by wiseuiux on 2022-05-25.
//
#include "VirtualObjectProcessor.h"

namespace EdgeSLAM{
        VirtualObject::VirtualObject(){}
        VirtualObject::VirtualObject(int _model, int _id, const cv::Mat& _pos, int _TTL):mnId(_id), mnModelCategory(_model), mnTTL(_TTL), pos(_pos){}
        VirtualObject::VirtualObject(int _model, int _id, int _nid, const cv::Mat& _pos, const cv::Mat& _epos, bool bPath, int _TTL):mfTotalTime(5.0), mnId(_id), mnNextId(_nid), mnModelCategory(_model), mnTTL(_TTL), pos(_pos), epos(_epos), mbPath(bPath), mbMoving(false){}
        VirtualObject::~VirtualObject(){
            pos.release();
            if(mbPath){
                epos.release();
            }
        }
        cv::Mat VirtualObject::UpdateWorldPos(float t){
            return pos+dir*t;
        }
        void VirtualObject::Set(){
            dir = (epos - pos)/mfTotalTime;
            time_start = std::chrono::high_resolution_clock::now();
            mbMoving = true;
        }
        bool VirtualObject::Check(float t){
            if(t >= mfTotalTime)
                return true;
            return false;
        }
        void VirtualObject::Reset(){
            mbMoving = false;
            dir.release();
        }
        void VirtualObject::Update(cv::Mat& _pos, int _TTL){
            _pos.copyTo(pos);
            mnTTL+=_TTL;
        }
        void VirtualObject::Update(cv::Mat& _pos, cv::Mat& _epos, int _nid, bool bPath, int _TTL){
            mnNextId = _nid;
            _pos.copyTo(pos);
            _epos.copyTo(epos);
            mbPath = bPath;
            mnTTL+=_TTL;
        }
        VirtualObjectProcessor::VirtualObjectProcessor():VOState(VOManageState::None), LastObject(nullptr){
        }
        void VirtualObjectProcessor::Reset(){

        }
        VirtualObjectProcessor::~VirtualObjectProcessor(){
            //if(LastObject)
            //delete LastObject;
            auto mapVOs = VOManageMap.Get();
            for(auto iter = mapVOs.begin(), iend = mapVOs.end(); iter != iend; iter++)
                delete iter->second;
            VOManageMap.Release();
        }
}