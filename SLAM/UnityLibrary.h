#pragma once
#include "Utils.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core.hpp"
#include <mutex>

extern "C" {
    void SetPath(char* path);
    void LoadVocabulary();
    void SetInit(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, int nfeature, int nlevel, float fscale, int nSkip, int nKFs);//char* vocName,
    void ConnectDevice();

    void SetDataFromUnity(void* data, char* path, int len, int strlen);
    cv::Mat GetDataFromUnity(std::string keyword);

    bool Track(void* pose);

    int  SetFrame(void* data, int id, double ts);
    void SetReferenceFrame(int id);
    void SetLocalMap();
    void SetIMUAddress(void* addr, bool bIMU);

    void AddObjectInfos();
    void AddContentInfo(int id, float x, float y, float z);

    void SemanticColorInit();
    void Segmentation();

    void WriteLog(char* data);

    bool VisualizeFrame(void* data);
    //bool GetMatchingImage(void* data);

}