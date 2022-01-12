#pragma once
#include "Utils.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core.hpp"
#include <mutex>

extern "C" {
    void SetPath(char* path);
    void LoadVocabulary();
    void SetInit(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, int nfeature, int nlevel, float fscale, int nSkip, int nKFs);//char* vocName,
    void SetUserName(char* c_src, int len);
    void ConnectDevice();

    void SetDataFromUnity(void* data, char* path, int len, int strlen);
    cv::Mat GetDataFromUnity(std::string keyword);
    void ReleaseUnityData(std::string keyword);

    void Parsing(int id, std::string key, cv::Mat data, bool bTracking);
    void LoadData(std::string key, int id, std::string src, bool bTracking);
    void CreateReferenceFrame(int id, cv::Mat data);
    bool Track(void* pose);

    int  SetFrame(void* data, int id, double ts);
    void SetReferenceFrame(int id);
    void SetIMUAddress(void* addr, bool bIMU);

    void AddObjectInfos(int id);
    void AddContentInfo(int id, float x, float y, float z);

    void SemanticColorInit();
    void Segmentation(int id);

    void WriteLog(std::string str);

    bool VisualizeFrame(void* data);
    //bool GetMatchingImage(void* data);

    void TestUploaddata(char* data, int datalen, int id, char* ckey, int clen1, char* csrc, int clen2, double ts);
    void TestDownloaddata(int id, char* ckey, int clen1, char* csrc, int clen2, bool bTracking);


    bool Localization(void* data, int id, double ts, int nQuality, bool bTracking, bool bVisualization);
}