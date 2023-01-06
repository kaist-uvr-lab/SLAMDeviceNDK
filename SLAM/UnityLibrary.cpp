#include "UnityLibrary.h"
#include "Camera.h"
#include "ORBDetector.h"

#include "Map.h"
#include "Frame.h"
#include "RefFrame.h"
#include "MapPoint.h"
#include "SearchPoints.h"
#include "Optimizer.h"
#include "Tracker.h"
#include "CameraPose.h"
#include "MotionModel.h"
#include "PlaneProcessor.h"
#include "VirtualObjectProcessor.h"

#include "ThreadPool.h"
#include "ConcurrentVector.h"
#include "ConcurrentMap.h"
#include "ConcurrentDeque.h"
#include <atomic>
#include <thread>

#include "WebAPI.h"
//#pragma comment(lib, "ws2_32")

//���� https://darkstart.tistory.com/42
extern "C" {

    //std::map<int, int> testUploadCount;
    //std::map<int, float> testUploadTime;
    ConcurrentMap<int, int> testIndirectCount;
    ConcurrentMap<int, std::chrono::high_resolution_clock::time_point> testIndirectClock;
    ConcurrentVector<float> testIndirectLatency;
    ConcurrentMap<int, std::chrono::high_resolution_clock::time_point> testDirectClock;
        ConcurrentVector<float> testDirectLatency;

    ConcurrentMap<std::string, int> testUploadCount;
    ConcurrentMap<std::string, float> testUploadTime;

    std::map<std::string, int> testDownloadCount;
    std::map<std::string, float> testDownloadTime;

    std::string strSource;
    std::vector<int> param = std::vector<int>(2);

    ConcurrentMap< int, std::chrono::high_resolution_clock::time_point> MapReferenceLatency;
    ConcurrentMap< int, std::chrono::high_resolution_clock::time_point> MapTouchLatency;

    ConcurrentMap<int, cv::Mat> TouchScreenImage; //가상 객체, 물체 위치 등을 체크하기 위한 것
    ConcurrentMap<int, cv::Mat> LocalMapPlanes; //floor, ceil
    ConcurrentMap<int, cv::Mat> LocalMapWallPlanes; //wall
    ConcurrentMap<int, cv::Mat> LocalMapPlaneLines; //lw
    ConcurrentMap<int, cv::Mat> LocalMapPlaneProjectionLines; //lc

    ConcurrentMap<int, cv::Mat> mapSendedImages;
    ConcurrentDeque<EdgeSLAM::RefFrame*> dequeRefFrames;
    ConcurrentVector<EdgeSLAM::MapPoint*> LocalMapPoints;

	EdgeSLAM::Frame* pCurrFrame = nullptr;
	EdgeSLAM::Frame* pPrevFrame = nullptr;
	EdgeSLAM::Camera* pCamera;
	EdgeSLAM::ORBDetector* pDetector;
	EdgeSLAM::MotionModel* pMotionModel;
	EdgeSLAM::CameraPose* pCameraPose;
	EdgeSLAM::Tracker* pTracker;
	EdgeSLAM::Map* pMap;
	EdgeSLAM::VirtualObjectProcessor* pVOProcessor= nullptr;
    ThreadPool::ThreadPool* POOL = nullptr;

	//std::map<int, EdgeSLAM::MapPoint*> EdgeSLAM::RefFrame::MapPoints;
	EdgeSLAM::ORBDetector* EdgeSLAM::Tracker::Detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::SearchPoints::Detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::MapPoint::Detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::RefFrame::detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::Frame::detector;
	EdgeSLAM::Map* EdgeSLAM::RefFrame::MAP;
	EdgeSLAM::Map* EdgeSLAM::MapPoint::MAP;

    ////object detection
    cv::Mat ObjectInfo = cv::Mat::zeros(0,6,CV_32FC1);
    std::mutex mMutexObjectInfo;
    ////object detection

    ////segmentation
    std::mutex mMutexSegmentation;
    std::vector<cv::Vec4b> SemanticColors;
    cv::Mat LabeledImage = cv::Mat::zeros(0,0,CV_8UC4);
    ////segmentation

    std::string strPath;
	int mnWidth, mnHeight;
	int mnSkipFrame;
	int mnFeature, mnLevel;
	float mfScale;
    int mnKeyFrame;
    std::atomic<int> mnSendedID, mnLastUpdatedID;

    std::ifstream inFile;
    char x[1000];

    std::ofstream ofile;
    std::string strLogFile;

    std::map<std::string, cv::Mat> UnityData;
    std::mutex mMutexUnityData;

	void SemanticColorInit() {
		cv::Mat colormap = cv::Mat::zeros(256, 3, CV_8UC1);
		cv::Mat ind = cv::Mat::zeros(256, 1, CV_8UC1);
		for (int i = 1; i < ind.rows; i++) {
			ind.at<uchar>(i) = i;
		}

		for (int i = 7; i >= 0; i--) {
			for (int j = 0; j < 3; j++) {
				cv::Mat tempCol = colormap.col(j);
				int a = pow(2, j);
				int b = pow(2, i);
				cv::Mat temp = ((ind / a) & 1) * b;
				tempCol |= temp;
				tempCol.copyTo(colormap.col(j));
			}
			ind /= 8;
		}

		for (int i = 0; i < colormap.rows; i++) {
			cv::Vec4b color = cv::Vec4b(colormap.at<uchar>(i, 0), colormap.at<uchar>(i, 1), colormap.at<uchar>(i, 2),255);
			SemanticColors.push_back(color);
		}
	}
    void Segmentation(int id){
        /*
        std::unique_lock<std::mutex> lock(mMutexSegmentation);
        cv::Mat tdata = GetDataFromUnity("Segmentation");
        ReleaseUnityData("Segmentation");

        //cv::Mat temp = cv::Mat::zeros(tdata.rows, 1, CV_8UC1);
        //std::memcpy(temp.data, tdata.data(), res.size());
        cv::Mat labeled = cv::imdecode(tdata, cv::IMREAD_GRAYSCALE);

        int w = labeled.cols;
        int h = labeled.rows;

        LabeledImage = cv::Mat::zeros(h, w, CV_8UC4);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int label = labeled.at<uchar>(y, x) + 1;
                LabeledImage.at<cv::Vec4b>(y, x) = SemanticColors[label];
            }
        }
        */
    }

    void SetUserName(char* c_src, int len){
        strSource = std::string(c_src, len);
    }
    void SetPath(char* path) {
        strPath = std::string(path);
        std::stringstream ss;
        ss << strPath << "/debug.txt";
        strLogFile = strPath+"/debug.txt";
    }

    void SetDataFromUnity(void* data, char* ckeyword, int len, int strlen){
        cv::Mat tempData(len,1,CV_8UC1,data);
        std::string keyword(ckeyword, strlen);
        {
            std::unique_lock < std::mutex> lock(mMutexUnityData);
            UnityData[keyword] = tempData;
        }
    }

    cv::Mat GetDataFromUnity(std::string keyword){
        std::unique_lock < std::mutex> lock(mMutexUnityData);
        return UnityData[keyword].clone();
    }
    void ReleaseUnityData(std::string keyword){
        std::unique_lock < std::mutex> lock(mMutexUnityData);
        UnityData[keyword].release();
        UnityData.erase(keyword);
    }

    void SetInit(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, int nfeature, int nlevel, float fscale, int nSkip, int nKFs) {//char* vocName,

        ofile.open(strLogFile.c_str(), std::ios::trunc);
        ofile<<"start\n";
        ofile.close();

        param[0] = cv::IMWRITE_JPEG_QUALITY;

		mnWidth  = _w;
		mnHeight = _h;
        mnSkipFrame = nSkip;
        mnKeyFrame = nKFs;

        mnFeature = nfeature;
        mnLevel = nlevel;
        mfScale = fscale;

        POOL = new ThreadPool::ThreadPool(8);

        pDetector = new EdgeSLAM::ORBDetector(nfeature,fscale,nlevel);
        pCamera = new EdgeSLAM::Camera(_w, _h, _fx, _fy, _cx, _cy, _d1, _d2, _d3, _d4);
        SemanticColorInit();

        EdgeSLAM::Tracker::Detector = pDetector;
        EdgeSLAM::SearchPoints::Detector = pDetector;
        EdgeSLAM::RefFrame::detector = pDetector;
        EdgeSLAM::Frame::detector = pDetector;
        EdgeSLAM::MapPoint::Detector = pDetector;


        return;

    }
    void DisconnectDevice(){

        if(pPrevFrame)
            delete pPrevFrame;
        pPrevFrame = nullptr;
        if(pCurrFrame)
            delete pCurrFrame;
        pCurrFrame = nullptr;
        if(pMotionModel)
            delete pMotionModel;
        pMotionModel = nullptr;
        if(pCameraPose)
            delete pCameraPose;
        pCameraPose = nullptr;

        if(pTracker)
            delete pTracker;
        pTracker = nullptr;

        if(pVOProcessor)
            delete pVOProcessor;
        pVOProcessor = nullptr;

        if(pMap)
            delete pMap;
        pMap = nullptr;

        /*
        auto tempRefFrames = dequeRefFrames.get();
        for(int i = 0; i < tempRefFrames.size(); i++)
            delete tempRefFrames[i];
        */
        MapReferenceLatency.Release();
        MapTouchLatency.Release();
        TouchScreenImage.Release();
        LocalMapPoints.Release();
        dequeRefFrames.Release();
        mapSendedImages.Release();
        testIndirectLatency.Release();
        testIndirectClock.Release();
        testDirectLatency.Release();
        testDirectClock.Release();

        ////save txt
        std::string testfile = strPath+"/Experiment/upload.txt";
        std::ofstream ofile2;
        ofile2.open(testfile.c_str(), std::ios::trunc);
        auto upCountData = testUploadCount.Get();
        auto upTimeData = testUploadTime.Get();
        for(auto iter = upCountData.begin(), iend = upCountData.end(); iter != iend; iter++){
            auto key = iter->first;
            int c = iter->second;
            float t= upTimeData[key];
            std::stringstream ss;
            ss<<key<<" "<<c<<" "<<t<<" "<<t/c<<std::endl;
            ofile2<<ss.str();
        }
        ofile2.close();
        /*
        for(int i = 10; i <= 100; i+=10){
            std::stringstream ss;
            ss<<i<<" "<<testUploadCount[i]<<" "<<testUploadTime[i]<<" "<<testUploadTime[i]/testUploadCount[i]<<std::endl;
            ofile2<<ss.str();
        }
        */

        testfile = strPath+"/Experiment/indirect.txt";
        ofile2.open(testfile.c_str(), std::ios::trunc);
        auto latencyVec = testIndirectLatency.get();
        {
            std::stringstream ss;
            ss<<latencyVec.size()<<std::endl;
            ofile2<<ss.str();
        }

        for(int i = 0; i < latencyVec.size(); i++){
            float t = latencyVec[i];
            std::stringstream ss;
            ss<<t<<std::endl;
            ofile2<<ss.str();
        }
        ofile2.close();

        testfile = strPath+"/Experiment/download.txt";
        ofile2.open(testfile.c_str(), std::ios::trunc);
        for(auto iter = testDownloadTime.begin(), iend = testDownloadTime.end(); iter != iend ; iter++){
            auto key = iter->first;
            std::stringstream ss;
            ss<<key<<" "<<testDownloadCount[key]<<" "<<testDownloadTime[key]<<" "<<testDownloadTime[key]/testDownloadCount[key]<<std::endl;
            ofile2<<ss.str();
        }
        ofile2.close();



        ////
    }
    void ConnectDevice() {

        pTracker = new EdgeSLAM::Tracker();

        pMap = new EdgeSLAM::Map();

        pVOProcessor = new EdgeSLAM::VirtualObjectProcessor();

        EdgeSLAM::RefFrame::nId = 0;
        pCameraPose = new EdgeSLAM::CameraPose();
        pMotionModel = new EdgeSLAM::MotionModel();

        EdgeSLAM::RefFrame::MAP = pMap;
        EdgeSLAM::MapPoint::MAP = pMap;

        /*
        EdgeSLAM::Tracker::Detector = pDetector;
        EdgeSLAM::SearchPoints::Detector = pDetector;
        EdgeSLAM::RefFrame::detector = pDetector;
        EdgeSLAM::Frame::detector = pDetector;
        EdgeSLAM::MapPoint::Detector = pDetector;
        EdgeSLAM::RefFrame::MAP = pMap;
        EdgeSLAM::MapPoint::MAP = pMap;
        */
        //EdgeSLAM::LocalMap::logFile = strLogFile;
        //EdgeSLAM::RefFrame::MapPoints = mapMapPoints;
        mnLastUpdatedID = -1;
        pTracker->mTrackState = EdgeSLAM::TrackingState::NotEstimated;
        LabeledImage = cv::Mat::zeros(0,0,CV_8UC4);

        ////load txt
        {
            std::string s;
            std::string testfile = strPath+"/Experiment/upload.txt";
            std::ifstream ifile;
            ifile.open(testfile);
            while(!ifile.eof()){
                getline(ifile, s);
                std::stringstream ss;
                std::string key;
                int n;
                float total, avg;
                ss<< s;
                ss >>key>> n >> total >> avg;
                testUploadCount.Update(key, n);
                testUploadTime.Update(key, total);
                if(ifile.eof())
                    break;
            }
            ifile.close();
            /*
            for(int i = 10; i <= 100; i+=10){
                getline(ifile, s);
                std::stringstream ss;
                int n, q;
                float total, avg;
                ss<< s;
                ss >> q >> n >> total >> avg;

                testUploadCount[i] = n;
                testUploadTime[i] = total;
                //ss<<i<<" = "<<testUploadCount[i]<<" "<<testUploadTime[i]<<" "<<testUploadTime[i]/testUploadCount[i]<<std::endl;
                //ofile2<<ss.str();
            }
            */
            testfile = strPath+"/Experiment/indirect.txt";
            ifile.open(testfile);
            while(!ifile.eof()){
                getline(ifile, s);
                std::stringstream ss;
                float t;
                ss<< s;
                ss >>t;
                testIndirectLatency.push_back(t);
                if(ifile.eof())
                    break;
            }
            ifile.close();

            testfile = strPath+"/Experiment/download.txt";
            ifile.open(testfile);
            for(int i = 0; i < 3; i++){
                getline(ifile, s);
                std::stringstream ss;
                std::string key;
                int n;
                float total, avg;
                ss<< s;
                ss >>key>> n >> total >> avg;
                testDownloadCount[key] = n;
                testDownloadTime[key] = total;
            }
            ifile.close();
        }

    }

    void* imuAddr;
    void* accAddr;
    bool bIMU = false;
	void SetIMUAddress(void* addr, bool bimu){
        imuAddr = addr;
        bIMU = bimu;
	}

    int scale = 4;

    bool bResReference = false;
    void Parsing(int id, std::string key, const cv::Mat& data, bool bTracking){
        if(key == "ReferenceFrame"){
            if(bTracking){
                CreateReferenceFrame(id, data);
            }else{
                float* tdata = (float*)data.data;
                int N = (int)tdata[0];
                if( N > 30){
                    bResReference = true;
                }else{
                    bResReference = false;
                }
            }
        }
    }

    ////추후 삭제

    void IndirectSyncLatency(int id, const cv::Mat& data){
        auto ts = testIndirectClock.Get(id);
        std::chrono::high_resolution_clock::time_point te = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
        float t = d / 1000.0;
        testIndirectLatency.push_back(t);
        std::stringstream ss;
        ss<<"Indirect = res "<<id<<" "<<t<<" "<<testDirectLatency.size();
        WriteLog(ss.str());
    }

    void DirectSyncLatency(int id, const cv::Mat& data){
        auto ts = testDirectClock.Get(id);
        std::chrono::high_resolution_clock::time_point te = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
        float t = d / 1000.0;
        testDirectLatency.push_back(t);
        std::stringstream ss;
        ss<<"Direct = res "<<id<<" "<<t<<" "<<testDirectLatency.size();
        WriteLog(ss.str());
    }

    void StoreData(std::string key, int id, std::string src, double ts, const void* data, int lendata){
        std::stringstream ss;
		ss <<"/Store?keyword="<<key<<"&id="<<id<<"&src="<<src<<"&ts="<<std::fixed<<std::setprecision(6)<<ts;
		//std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();
		WebAPI api("143.248.6.143", 35005);
        auto res = api.Send(ss.str(), (const unsigned char*)data, lendata);

        //std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
        //auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
        //float t = d / 1000.0;
        //ss.str("");
        //ss<<"upload test ="<<t<<" "<<lendata;
        //WriteLog(ss.str());
    }
    void LoadData(std::string key, int id, std::string src, bool bTracking){
        std::stringstream ss;
		ss <<"/Load?keyword="<<key<<"&id="<<id<<"&src="<<src;//"<< "/Load?keyword=Keypoints" << "&id=" << id << "&src=" << user->userName;
		std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();
		WebAPI api("143.248.6.143", 35005);
        auto res = api.Send(ss.str(),"");
        int n2 = res.size();
        std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
        float t = d / 1000.0;
        ss.str("");
        ss<<"download test ="<<t<<" "<<n2;
        testDownloadCount[key]++;
        testDownloadTime[key]+=t;

        //set data
        cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
        std::memcpy(temp.data, res.data(), n2);
        Parsing(id, key, temp, bTracking);

    }

    bool CreateWorldPosition(float x, float y, cv::Mat& _pos){
        cv::Mat x3D = cv::Mat::ones(1,3,CV_32FC1);
        x3D.at<float>(0) = x;
        x3D.at<float>(1) = y;

        cv::Mat R, t;
        pCameraPose->GetPose(R,t);

        cv::Mat Xw = pCamera->Kinv * x3D.t();
        Xw.push_back(cv::Mat::ones(1,1,CV_32FC1)); //3x1->4x1
        Xw = pCameraPose->GetInversePose()*Xw; // 4x4 x 4 x 1
        float testaaasdf = Xw.at<float>(3);
        Xw = Xw.rowRange(0,3)/Xw.at<float>(3); // 4x1 -> 3x1
        cv::Mat Ow = pCameraPose->GetCenter(); // 3x1
        cv::Mat dir = Xw-Ow; //3x1

        bool bres = false;
        auto planes = LocalMapPlanes.Get();
        float min_val = 10000.0;
        cv::Mat min_param;
        for(auto iter = planes.begin(), iend = planes.end(); iter != iend; iter++){
            cv::Mat param = iter->second; //4x1
            cv::Mat normal = param.rowRange(0,3); //3x1
            float dist = param.at<float>(3);
            float a = normal.dot(-dir);
            if(std::abs(a) < 0.000001)
                continue;
            float u = (normal.dot(Ow)+dist)/a;
            if(u > 0.0 && u < min_val){
                min_val = u;
                min_param = param;
            }

        }
        if(min_val < 10000.0){
            _pos =  Ow+dir*min_val;
            bres = true;
        }
        return bres;
    }

    enum class TouchRegionState {
        None, RealObject, VirtualObject
    };

    int indirectID = 1;

    void TestUploaddata(char* data, int datalen, int id, char* ckey, int clen1, char* csrc, int clen2, double ts){
        std::string key(ckey, clen1);
        std::string src(csrc, clen2);
        POOL->EnqueueJob(StoreData, key, id, src, ts, data, datalen);
    }
    void TestDownloaddata(int id, char* ckey, int clen1, char* csrc, int clen2, bool bTracking){
        std::string key(ckey, clen1);
        std::string src(csrc, clen2);
        POOL->EnqueueJob(LoadData, key, id, src, bTracking);
    }


    void CreateReferenceFrame(int id, const cv::Mat& data){
        //WriteLog("SetReference::Start!!!!!!!!!!!!!!!!!!!!!!!!!!!!");//, std::ios::trunc
        //cv::Mat f1 = GetDataFromUnity("ReferenceFrame");
        //ReleaseUnityData("ReferenceFrame");
        float* tdata = (float*)data.data;
        int N = (int)tdata[0];
        if(N > 30){
            auto pRefFrame = new EdgeSLAM::RefFrame(pCamera, tdata);
            cv::Mat img = mapSendedImages.Get(id);
            mapSendedImages.Erase(id);
            pDetector->Compute(img, cv::Mat(), pRefFrame->mvKeys, pRefFrame->mDescriptors);
            //std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pRefFrame->mDescriptors);
            //pVoc->transform(vCurrentDesc, pRefFrame->mBowVec, pRefFrame->mFeatVec, 4);

            pRefFrame->UpdateMapPoints();
            dequeRefFrames.push_back(pRefFrame);

            ////local map 갱신
            std::set<EdgeSLAM::MapPoint*> spMPs;
            //WriteLog("Reference::Delete::Start");
            ////일정 레퍼런스 프레임이 생기면 디큐의 처음 레퍼런스 프레임 제거
            EdgeSLAM::RefFrame* firstRef = nullptr;
            if(dequeRefFrames.size() >= mnKeyFrame){
                firstRef = dequeRefFrames.front();
                dequeRefFrames.pop_front();
                ////delete ref frame
                if(firstRef){
                    auto vpMPs = firstRef->mvpMapPoints;
                    for(int i =0; i < firstRef->N; i++){
                        auto pMP = vpMPs[i];
                        if(!pMP || pMP->isBad()){
                            continue;
                        }
                        pMP->EraseObservation(firstRef);
                    }
                    //delete firstRef;
                }
            }
            //WriteLog("Reference::Delete::End");
            std::vector<EdgeSLAM::MapPoint*> vecMPs;
            auto vecRefFrames = dequeRefFrames.get();
            for(int i = 0; i < vecRefFrames.size(); i++){
                auto ref = vecRefFrames[i];
                if(!ref)
                    continue;
                auto vpMPs = ref->mvpMapPoints;
                for(int j =0; j < ref->N; j++){
                    auto pMP = vpMPs[j];
                    if(!pMP || pMP->isBad() || spMPs.count(pMP)){
                        continue;
                    }
                    if(pRefFrame->is_in_frustum(pMP, 0.5)){
                        vecMPs.push_back(pMP);
                        spMPs.insert(pMP);
                    }
                }
            }
            //WriteLog("Reference::UpdateLocalMap::End");
            pMap->SetReferenceFrame(pRefFrame);
            LocalMapPoints.set(vecMPs);
            //f1.release();
        }
        //WriteLog("SetReference::End!!!!!!!!!!!!!!!!!!!");
    }

    std::mutex mMutexLogFile;
    void WriteLog(std::string str, std::ios_base::openmode mode){
        //std::string log(data);
        std::unique_lock<std::mutex> lock(mMutexLogFile);
        ofile.open(strLogFile.c_str(), mode);
        ofile<<str<<"\n";
        ofile.close();
    }

    void SendDevicePose(int id, double ts, cv::Mat T){
        //cv::Mat T = pCurrFrame->GetPose();
        cv::Mat P = cv::Mat::zeros(4,3, CV_32FC1);
        T.rowRange(0, 3).colRange(0, 3).copyTo(P.rowRange(0, 3));
        cv::Mat t = T.col(3).rowRange(0, 3).t();
        t.copyTo(P.row(3));
        StoreData("DevicePosition", id, strSource, ts, P.data, 48);
    }

    bool Localization(void* texdata, void* posedata, int id, double ts, int nQuality, bool bTracking, bool bVisualization){

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        bool res = true;

        //유니티에서 온 프레임
        cv::Mat ori_frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, texdata);
        cv::Mat frame = ori_frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
        frame.convertTo(frame, CV_8UC3);

        //NDK에서 이용하는 이미지로 변환
        cv::flip(frame, frame,0);
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY

        bool bTrack = false;
        int nMatch = -1;
        if(bTracking){

            ////이전 프레임 해제
            if(pPrevFrame)
                delete pPrevFrame;
            pPrevFrame = pCurrFrame;
            pCurrFrame = new EdgeSLAM::Frame(gray, pCamera, id);

            if (pTracker->mTrackState == EdgeSLAM::TrackingState::NotEstimated || pTracker->mTrackState == EdgeSLAM::TrackingState::Failed) {
                EdgeSLAM::RefFrame* rf = pMap->GetReferenceFrame();
                if (rf) {
                    nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame, 100.0, 50.0);
                    bTrack = nMatch >= 10;
                    if (bTrack) {
                        pTracker->mnLastRelocFrameId = pCurrFrame->mnFrameID;
                    }
                }
            }

            if (pTracker->mTrackState == EdgeSLAM::TrackingState::Success) {
                //predict
                if(bIMU){
                    cv::Mat Rgyro = cv::Mat(3,3, CV_32FC1,imuAddr);
                    cv::Mat tacc = cv::Mat::zeros(3,1,CV_32FC1);
                    cv::Mat Timu = cv::Mat::eye(4,4,CV_32FC1);
                    Rgyro.copyTo(Timu.rowRange(0,3).colRange(0,3));
                    tacc.copyTo(Timu.rowRange(0,3).col(3));
                    cv::Mat Tprev = pPrevFrame->GetPose();
                    pCurrFrame->SetPose(Timu*Tprev);
                }else
                    pCurrFrame->SetPose(pMotionModel->predict());
                //WriteLog("Localization::TrackPrev::Start");

                nMatch = pTracker->TrackWithPrevFrame(pPrevFrame, pCurrFrame, 100.0, 50.0);
                //WriteLog("Localization::TrackPrev::End");
                bTrack = nMatch >= 10;
                if (!bTrack) {
                    EdgeSLAM::RefFrame* rf = pMap->GetReferenceFrame();
                    if (rf) {
                        nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame, 100.0, 50.0);
                        bTrack = nMatch >= 10;
                    }
                }
                //WriteLog("Localization::TrackPrev::End2");
            }

            if (bTrack) {
                auto vecLocalMPs = LocalMapPoints.get();
                nMatch = 4;
                //WriteLog("Localization::TrackLocalMap::Start");
                nMatch = pTracker->TrackWithLocalMap(pCurrFrame, vecLocalMPs, 100.0, 50.0);
                //WriteLog("Localization::TrackLocalMap::End");
                if (pCurrFrame->mnFrameID < pTracker->mnLastRelocFrameId + 30 && nMatch < 30) {
                    bTrack = false;
                }
                else if (nMatch < 30) {
                    bTrack = false;
                }
                else {
                    bTrack = true;
                }
            }
            if (bTrack) {
                pTracker->mTrackState = EdgeSLAM::TrackingState::Success;
                cv::Mat T = pCurrFrame->GetPose();
                pCameraPose->SetPose(T);
                pMotionModel->update(T);
                POOL->EnqueueJob(SendDevicePose, id, ts, T.clone());

                ////유니티에 카메라 포즈 복사
                ////R과 카메라 센터
                //cv::Mat t = T.col(3).rowRange(0, 3).t();
                cv::Mat P = cv::Mat(4,3, CV_32FC1, posedata);
                T.rowRange(0, 3).colRange(0, 3).copyTo(P.rowRange(0, 3));
                cv::Mat Ow = pCameraPose->GetCenter().t();
                Ow.copyTo(P.row(3));
            }
            else {
                //WriteLog("Tracking::Failed", std::ios::trunc);
                pTracker->mTrackState = EdgeSLAM::TrackingState::Failed;
                pCameraPose->Init();
                pMotionModel->reset();

            }

        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		float t = d / 1000.0;
		/*
		{
		    std::stringstream ss;
		    ss<<"Localizatio="<<id<<"="<<nMatch<<", "<<t<<std::endl;
		    WriteLog(ss.str());
		}
		*/
		bool bres = bTrack;
		if(!bTracking){
            bres = bResReference;
		}
        return bres;
    }
}
