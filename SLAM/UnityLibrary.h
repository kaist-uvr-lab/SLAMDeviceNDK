#pragma once
#include "Utils.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core.hpp"
#include <mutex>

extern "C" {
    void SetPath(char* path);
    void SetInit(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, int nfeature, int nlevel, float fscale, int nSkip, int nKFs);//char* vocName,
    void SetUserName(char* c_src, int len);
    void ConnectDevice();
    void DisconnectDevice();

    void Parsing(int id, std::string key, const cv::Mat& data, bool bTracking);
    void LoadData(std::string key, int id, std::string src, bool bTracking);
    void CreateReferenceFrame(int id, const cv::Mat& data);

    void SetIMUAddress(void* addr, bool bIMU);

    void IndirectSyncLatency(int id, const cv::Mat& data);
    void DirectSyncLatency(int id, const cv::Mat& data);
    void MovingObjectSync(const cv::Mat& data);
    void UpdateLocalMapContent(const cv::Mat& data);
    void UpdateLocalMapPlane(const cv::Mat& data);
    void AddObjectInfo(int id, cv::Mat data);
    void AddObjectInfos(int id);
    void AddContentInfo(int id, float x, float y, float z);

    void SemanticColorInit();
    void Segmentation(int id);

    void TouchProcessInit(int touchID, int touchPhase, float x, float y, double ts);
    void WriteLog(std::string str, std::ios_base::openmode mode = std::ios_base::out | std::ios_base::app);

    void TestUploaddata(char* data, int datalen, int id, char* ckey, int clen1, char* csrc, int clen2, double ts);
    void TestDownloaddata(int id, char* ckey, int clen1, char* csrc, int clen2, bool bTracking);


    bool Localization(void* data, void* posedata, int id, double ts, int nQuality, bool bTracking, bool bVisualization);
}