#pragma once
#include "Utils.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core.hpp"
#include <mutex>

extern "C" {
    void SetPath(char* path);
    void LoadVocabulary();
    void SetInit(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, int nfeature, int nlevel, float fscale);//char* vocName,
    void ConnectDevice();
    void SetIMUAddress(void* addr, bool bIMU);
    void SetLocalMap(void* data1, int len1, void* data2, int len2, void* data3, int len3, void* data4, int len4);


    int  SetFrame(void* data, int id, double ts, float& t1, float& t2);
    bool GetMatchingImage(void* data);
    void AddContentInfo(int id, float x, float y, float z);
    bool Track(void* pose);
    void WriteLog(char* data);


    void* GetResultAddr(bool& bres);

void ReleaseImage();
    /*
    int SetFrameByPtr(void* addr, int w, int h, int id);
	int SetFrameByFile(char* name, int id, double ts, float& t1, float& t2);
    void SetReferenceFrame(int id, float* data);
    int SetFrameByImage(unsigned char* raw, int len, int id, double ts, float& t1, float& t2);
    */



	//Hololens_DLL_API int TrackWithReferenceFrame(int id);
	/*Hololens_DLL_API void ResizeImage(Color* raw, Color** resized, int w, int h, int rw, int rh);
	void SetCameraDevice(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4);*/
}