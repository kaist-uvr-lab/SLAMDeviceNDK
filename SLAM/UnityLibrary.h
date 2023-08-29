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
    void SetIMUAddress(void* addr, bool bIMU);

    void IndirectSyncLatency(int id, const cv::Mat& data);
    void DirectSyncLatency(int id, const cv::Mat& data);
    void MovingObjectSync(const cv::Mat& data);
    void UpdateLocalMapContent(const cv::Mat& data);
    void UpdateLocalMapPlane(const cv::Mat& data);
    void AddObjectInfo(int id, cv::Mat data);
    void AddObjectInfos(int id);
    void AddContentInfo(int id, float x, float y, float z);

    void CreateDynamicObjectFrame(int id, float* data, int sidx);

    void SemanticColorInit();
    void Segmentation(int id);

    void TouchProcessInit(int touchID, int touchPhase, float x, float y, double ts);
    void WriteLog(std::string str, std::ios_base::openmode mode = std::ios_base::out | std::ios_base::app);

    void StoreImage(int id, void* addr);
    void EraseImage(int id);
    bool NeedNewKeyFrame(int fid);
    void NeedNewKeyFrame2(int fid);
    int CreateReferenceFrame(int id, bool bNotBase, float* data);
    int CreateReferenceFrame2(int id, float* data);
    void UpdateLocalMap(int id, int n, void* data);
    bool Localization(void* data, void* posedata, int id, double ts, int nQuality, bool bNotBase, bool bTracking, bool bVisualization);
    float UploadData(char* data, int datalen, int id, char* ckey, int clen1, char* csrc, int clen2, double ts);
    void DownloadData(int id, char* ckey, int clen1, char* csrc, int clen2, void* addr, int& N, float& t);
}