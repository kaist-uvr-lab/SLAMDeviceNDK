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
#include "/DynamicObject/DynamicObjectMap.h"
#include "/DynamicObject/DynamicFrame.h"
#include "/DynamicObject/DynamicEstimator.h"

#include "ThreadPool.h"
#include "ConcurrentVector.h"
#include "ConcurrentMap.h"
#include "ConcurrentDeque.h"
#include <atomic>
#include <thread>

#include "WebAPI.h"
#include <atomic>
//#pragma comment(lib, "ws2_32")

//���� https://darkstart.tistory.com/42
extern "C" {

    std::string strSource;
    std::vector<int> param = std::vector<int>(2);

    ConcurrentDeque<EdgeSLAM::RefFrame*> dequeRefFrames;
    ConcurrentVector<EdgeSLAM::MapPoint*> LocalMapPoints;
    ConcurrentMap<int, DynamicObjectMap*> LocalObjectMap;

	EdgeSLAM::Frame* pCurrFrame = nullptr;
	EdgeSLAM::Frame* pPrevFrame = nullptr;
	EdgeSLAM::Camera* pCamera;
	EdgeSLAM::ORBDetector* pDetector;
	EdgeSLAM::MotionModel* pMotionModel;
	EdgeSLAM::CameraPose* pCameraPose;
	EdgeSLAM::Tracker* pTracker;
	EdgeSLAM::Map* pMap;
    ThreadPool::ThreadPool* POOL = nullptr;

    DynamicEstimator* pDynamicEstimator = nullptr;
    DynamicFrame* pDynaRefFrame = nullptr;
    DynamicFrame* pDynaPrevFrame = nullptr;
    DynamicFrame* pDynaCurrFrame = nullptr;

	//std::map<int, EdgeSLAM::MapPoint*> EdgeSLAM::RefFrame::MapPoints;
	EdgeSLAM::ORBDetector* EdgeSLAM::Tracker::Detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::SearchPoints::Detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::MapPoint::Detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::RefFrame::detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::Frame::detector;
	EdgeSLAM::Map* EdgeSLAM::RefFrame::MAP;
	EdgeSLAM::Map* EdgeSLAM::MapPoint::MAP;

    ConcurrentMap<int, cv::Mat> mapSendedImages;

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

    std::ifstream inFile;
    char x[1000];

    std::ofstream ofile;
    std::string strLogFile;

    std::ofstream write_latency;
    std::string strLatencyFile;

    std::map<std::string, cv::Mat> UnityData;
    std::mutex mMutexUnityData;

    std::atomic<int> nRefMatches;
    std::atomic<int> nLastKeyFrameId;
    std::atomic<int> nMatch;
    std::atomic<bool> bReqLocalMap;

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
    bool bSetReferenceFrame = false;
    bool bSetLocalMap = false;
    void SetUserName(char* c_src, int len){
        strSource = std::string(c_src, len);
    }
    void SetPath(char* path) {
        strPath = std::string(path);
        std::stringstream ss;
        ss << strPath << "/debug.txt";
        strLogFile = strPath+"/debug.txt";
        strLatencyFile = strPath+"/latency.csv";
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

        nRefMatches = -1;
        nLastKeyFrameId = -1;
        nMatch = -1;
        bReqLocalMap = false;

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

        pDynamicEstimator = new DynamicEstimator();
{
    std::stringstream ss;
    ss<<"init = "<<_d1<<", "<<_d2<<" "<<_d3<<", "<<_d4<<", "<<pCamera->bDistorted;
    WriteLog(ss.str());
}
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


        if(pMap)
            delete pMap;
        pMap = nullptr;

        /*
        auto tempRefFrames = dequeRefFrames.get();
        for(int i = 0; i < tempRefFrames.size(); i++)
            delete tempRefFrames[i];
        */
        LocalMapPoints.Release();
        dequeRefFrames.Release();
        mapSendedImages.Release();
        ////
    }
    void ConnectDevice() {

        pTracker = new EdgeSLAM::Tracker();
        pTracker->filename = strLogFile;
        EdgeSLAM::SearchPoints::filename = strLogFile;
        pMap = new EdgeSLAM::Map();

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
        pTracker->mTrackState = EdgeSLAM::TrackingState::NotEstimated;
        LabeledImage = cv::Mat::zeros(0,0,CV_8UC4);

    }
    cv::Mat gray;
    void ConvertImage(int id, void* addr){
        cv::Mat ori_frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, addr);
        cv::Mat frame = ori_frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
        frame.convertTo(frame, CV_8UC3);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY
    }

    void StoreImage(int id, void* addr){

        cv::Mat ori_frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, addr);
        cv::Mat frame = ori_frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
        frame.convertTo(frame, CV_8UC3);
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY
        mapSendedImages.Update(id,gray.clone());

    }
    void EraseImage(int id){
        if(mapSendedImages.Count(id))
            mapSendedImages.Erase(id);
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
                //CreateReferenceFrame(id, data);
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

    float  StoreData(std::string key, int id, std::string src, double ts, const void* data, int lendata){
        std::stringstream ss;
		ss <<"/Store?keyword="<<key<<"&id="<<id<<"&src="<<src<<"&ts="<<std::fixed<<std::setprecision(6)<<ts;
		std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();
		WebAPI api("143.248.6.143", 35005);
        auto res = api.Send(ss.str(), (const unsigned char*)data, lendata);
        std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
        //float t = d / 1000.0;
        return d;
    }
    std::tuple<std::string, int, float> LoadData(std::string key, int id, std::string src){
        std::stringstream ss;
		ss <<"/Load?keyword="<<key<<"&id="<<id<<"&src="<<src;//"<< "/Load?keyword=Keypoints" << "&id=" << id << "&src=" << user->userName;
		std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();
		WebAPI api("143.248.6.143", 35005);
        auto res = api.Send(ss.str(),"");
        std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
        //set data
        int n = res.size() / 4;
        return std::make_tuple(res,n,d);
    }

    int indirectID = 1;

    float UploadData(char* data, int datalen, int id, char* ckey, int clen1, char* csrc, int clen2, double ts){
        std::string key(ckey, clen1);
        std::string src(csrc, clen2);
        auto res = POOL->EnqueueJob(StoreData, key, id, src, ts, data, datalen);
        float t = res.get();
        return t;
    }
    void DownloadData(int id, char* ckey, int clen1, char* csrc, int clen2, void* addr, int& N, float& t){
        std::string key(ckey, clen1);
        std::string src(csrc, clen2);
        //auto res = POOL->EnqueueJob(LoadData, key, id, src);
        //auto var = res.get();
        auto var = LoadData(key,id,src);
        auto temp = std::get<0>(var);
        N = std::get<1>(var);
        t = std::get<2>(var);
        memcpy(addr,temp.data(), sizeof(float)*N);
    }

    int mpid = 0;
    const int mpInfoSize = 36; //id + 3d + min+max+normal
    const int nNumElement = mpInfoSize/4;

    void UpdateLocalMap(int id, float* data){

        WriteLog("UpdateLocalMap::Start");
        int nSize = (int)data[0];
        if(nSize ==2){
            WriteLog("Not Updated LocalMap");
            return;
        }
        int nLocalMap = (int)data[2];
        int nObsIdx = 3;
        int nUpdatedMPs =(int)data[nObsIdx+nLocalMap];
        int nUpdatedIdx = nObsIdx+nLocalMap+1;

        std::vector<EdgeSLAM::MapPoint*> vpMPs;// = std::vector<EdgeSLAM::MapPoint*>(n, static_cast<EdgeSLAM::MapPoint*>(nullptr));

        for(int i = 0; i < nUpdatedMPs; i++){

            int id = (int)data[nUpdatedIdx++];
            float minDist = data[nUpdatedIdx++];
            float maxDist = data[nUpdatedIdx++];

            //Xw
            float x = data[nUpdatedIdx++];
            float y = data[nUpdatedIdx++];
            float z = data[nUpdatedIdx++];
            cv::Mat X = (cv::Mat_<float>(3, 1) <<  x,y,z);

            //normal
            float nx = data[nUpdatedIdx++];
            float ny = data[nUpdatedIdx++];
            float nz = data[nUpdatedIdx++];
            cv::Mat norm = (cv::Mat_<float>(3, 1) << nx,ny,nz);

            //desc
            void* ptrdesc = data+nUpdatedIdx;
            nUpdatedIdx+=8;
            cv::Mat desc(1,32,CV_8UC1,ptrdesc);

            EdgeSLAM::MapPoint* pMP = nullptr;
            if(pMap->MapPoints.Count(id)){
                pMP = pMap->MapPoints.Get(id);
                pMP->SetWorldPos(X.at<float>(0), X.at<float>(1), X.at<float>(2));
            }else{
                pMP = new EdgeSLAM::MapPoint(id, X.at<float>(0), X.at<float>(1), X.at<float>(2));
                pMap->MapPoints.Update(id, pMP);
            }
            pMP->SetMapPointInfo(minDist,maxDist,norm);
            pMP->SetDescriptor(desc);

        }

        int nError = 0;
        //add not updated mps
        for(int i = 0; i < nLocalMap; i++){
            int id = (int)data[nObsIdx++];

            EdgeSLAM::MapPoint* pMP = nullptr;
            if(pMap->MapPoints.Count(id)){
                pMP = pMap->MapPoints.Get(id);
                vpMPs.push_back(pMP);
            }else
                nError++;
        }
        {
            std::stringstream ss;
            ss<<"update local map = "<<nLocalMap<<" "<<nUpdatedMPs<<" "<<nError<<std::endl;
            WriteLog(ss.str());
        }
//WriteLog("B");
        //auto prevMPs = LocalMapPoints.get();
        LocalMapPoints.set(vpMPs);
        if(!bSetLocalMap){
            bSetLocalMap = true;
        }

        WriteLog("UpdateLocalMap::End");
        bReqLocalMap = false;
        nLastKeyFrameId = id;
    }

    int CreateReferenceFrame2(int id, float* data){
        return -1;
    }

    void CreateDynamicObjectFrame(int id, float* data, int startIdx){
        if(!mapSendedImages.Count(id))
            return;

        float* fdata = data+startIdx+1;
        //int Ndatasize = (int)fdata[0];
        int Nobject = (int)fdata[0];
        int nObjID = (int)fdata[1];
        //2 3 4
        //5 6 7
        //8 9 10
        //11 12 13
        cv::Mat Pwo = cv::Mat::eye(4,4,CV_32FC1);
        Pwo.at<float>(0, 0) = fdata[5];
        Pwo.at<float>(0, 1) = fdata[6];
        Pwo.at<float>(0, 2) = fdata[7];
        Pwo.at<float>(1, 0) = fdata[8];
        Pwo.at<float>(1, 1) = fdata[9];
        Pwo.at<float>(1, 2) = fdata[10];
        Pwo.at<float>(2, 0) = fdata[11];
        Pwo.at<float>(2, 1) = fdata[12];
        Pwo.at<float>(2, 2) = fdata[13];
        Pwo.at<float>(0, 3) = fdata[14];
        Pwo.at<float>(1, 3) = fdata[15];
        Pwo.at<float>(2, 3) = fdata[16];

        DynamicObjectMap* pObj = nullptr;
        if(!LocalObjectMap.Count(nObjID)){
            pObj = new DynamicObjectMap();
            LocalObjectMap.Update(nObjID, pObj);
        }else
            pObj = LocalObjectMap.Get(nObjID);
        pObj->SetPose(Pwo);

        int idx = 17;
        std::vector<cv::Point2f> imagePoints;
        std::vector<cv::Point3f> objectPoints;
        for(int i = 0; i < Nobject; i++){
            float x = fdata[idx++];
            float y = fdata[idx++];
            float X = fdata[idx++];
            float Y = fdata[idx++];
            float Z = fdata[idx++];
            cv::Point2f imgPt(x,y);
            cv::Point3f objPt(X,Y,Z);
            imagePoints.push_back(imgPt);
            objectPoints.push_back(objPt);
        }
        cv::Mat img = mapSendedImages.Get(id);
        if(pDynaRefFrame)
            delete pDynaRefFrame;
        pDynaRefFrame = new DynamicFrame(nObjID, img, imagePoints, objectPoints, Pwo, pCamera->K);

        std::stringstream ss;
        ss<<"Object Frame Test = "<<" "<<Nobject<<" "<<pDynaRefFrame->Pco;
        WriteLog(ss.str());

    }

    int CreateReferenceFrame(int id, bool bNotBase, float* data, int idx, float* mdata){
        //void CreateReferenceFrame(int id, const cv::Mat& data){
        //WriteLog("SetReference::Start!!!!!!!!!!!!!!!!!!!!!!!!!!!!");//, std::ios::trunc
        float* tdata = data+idx;
        int N = (int)tdata[2];

        //무조건 빼내야 하는 것임.
        if(!mapSendedImages.Count(id))
            return dequeRefFrames.size();
        cv::Mat img = mapSendedImages.Get(id);

//WriteLog("GetFrame");
        if(N > 30){
            WriteLog("CreateReference Start");
            //{
            //int Nmp = (int)data[13 + N*5+2];
            //std::stringstream ss;
            //ss<<"test = "<<N<<" "<<Nmp<<" "<<std::endl;
            //WriteLog(ss.str());
            //}

            //맵포인트 갱신
            int Nmp = (int)tdata[15 + N*5];
            int nMPidx = 16 + N*5;
            {
                std::stringstream ss;
                ss<<"Test "<<N<<" "<<Nmp<<std::endl;
                WriteLog(ss.str());
            }
            WriteLog("update mp start");
            if(bNotBase){
                for(int i = 0; i < Nmp; i++){
                    int id = (int)tdata[nMPidx++];
                    int label = (int)tdata[nMPidx++];
                    float x = tdata[nMPidx++];
                    float y = tdata[nMPidx++];
                    float z = tdata[nMPidx++];

                    EdgeSLAM::MapPoint* pMP = nullptr;
                    if(pMap->MapPoints.Count(id)){
                        pMP = pMap->MapPoints.Get(id);
                        pMP->SetWorldPos(x,y,z);
                    }else{
                        pMP = new EdgeSLAM::MapPoint(id, x, y, z);
                        pMap->MapPoints.Update(id, pMP);
                    }
                }
            }

            WriteLog("update mp end");
            {
                int nIdx = 15;
                int nMapIdx = 0;
                int nError = 0;
                int nError2 = 0;
                cv::Mat mapData = cv::Mat(N*3,1,CV_32FC1,mdata);
                for (int i = 0; i < N; i++) {
                    int id = (int)tdata[nIdx];
                    nIdx+=5;
                    if(pMap->MapPoints.Count(id)){
                        int midx = 3*i;
                        auto pMPi = pMap->MapPoints.Get(id);
                        if(pMPi && !pMPi->isBad()){
                            cv::Mat X = pMPi->GetWorldPos();
                            mapData.at<float>(midx) = X.at<float>(0);
                            mapData.at<float>(midx+1) = X.at<float>(1);
                            mapData.at<float>(midx+2) = X.at<float>(2);
                        }else
                        {
                            nError2++;
                            mapData.at<float>(midx) = 0.0;
                            mapData.at<float>(midx+1) = 0.0;
                            mapData.at<float>(midx+2) = 0.0;
                        }
                    }else{
                        nError++;
                    }
                }
                std::stringstream ss;
                ss<<"reference::error = "<<N<<"=="<<nError<<" "<<nError2<<std::endl;
                WriteLog(ss.str());
            }
            auto pRefFrame = new EdgeSLAM::RefFrame(pCamera, tdata+2);
            WriteLog("update kf end");
            pDetector->Compute(img, cv::Mat(), pRefFrame->mvKeys, pRefFrame->mDescriptors);
            //std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pRefFrame->mDescriptors);
            //pVoc->transform(vCurrentDesc, pRefFrame->mBowVec, pRefFrame->mFeatVec, 4);
            nRefMatches = pRefFrame->mvpMapPoints.size();
            pRefFrame->UpdateMapPoints();
            WriteLog("update kf end2");
            if(bNotBase){

                dequeRefFrames.push_back(pRefFrame);
                ////local map 갱신
                std::set<EdgeSLAM::MapPoint*> spMPs;

                ////일정 레퍼런스 프레임이 생기면 디큐의 처음 레퍼런스 프레임 제거
                //옵저베이션 제거
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

                ////로컬맵 생성
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
                LocalMapPoints.set(vecMPs);
                if(!bSetLocalMap){
                    bSetLocalMap = true;
                }
            }

            WriteLog("Reference::UpdateLocalMap::End");
            pMap->SetReferenceFrame(pRefFrame);
            if(!bSetReferenceFrame){
                bSetReferenceFrame = true;
            }

            //LocalMapPoints.set(vecMPs);
            //f1.release();
        }

WriteLog("SetReference::End!!!!!!!!!!!!!!!!!!!");
        return N;
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
    void SendImage(int id, double ts, int nQuality, cv::Mat frame){
        param[1] = nQuality;
        std::vector<uchar> buffer;
        cv::imencode(".jpg", frame, buffer, param);

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        StoreData("Image", id, strSource, ts, buffer.data(), buffer.size());
    }

    int nMinFrames = 0;
    int nMaxFrames = 30;
    float thRefRatio = 0.9f;

    bool NeedNewKeyFrame(int fid){

        //int nRefMatches = pRefFrame->mvpMapPoints.size();
        bool bLocalMappingIdle = !bReqLocalMap;
        const bool c1a = fid >= nLastKeyFrameId + nMaxFrames;
        const bool c1b = fid >= nLastKeyFrameId + nMinFrames && bLocalMappingIdle;
        const bool c2 = (nMatch<nRefMatches*thRefRatio) && nMatch>15;

        bool bres = (c1a||c1b)&&c2;
        if(bres){
            bReqLocalMap = true;
            //nLastKeyFrameId = fid;
        }
        return bres;
    }
    void NeedNewKeyFrame2(int fid){
        bReqLocalMap = true;
        //return bLocalMappingIdle;
    }
    int DynamicObjectTracking(int id, int &objID, void* posedata, void* oposedata){
        if(pDynaPrevFrame)
            delete pDynaPrevFrame;
        pDynaPrevFrame = pDynaCurrFrame;
        pDynaCurrFrame = new DynamicFrame(gray,pCamera->K);

        if(pDynaRefFrame){
            WriteLog("test object detection");

            std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();
            int nMatchedObject = pDynamicEstimator->SearchPointsByOpticalFlow(pDynaRefFrame, pDynaCurrFrame);
            /*
            std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
            {
                auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
                float t = d / 1000.0;
                std::stringstream ss;
                ss<<"Optical flow = "<<nMatchedObject<<" "<<t<<std::endl;
                WriteLog(ss.str());
                ss.str("");
                ss<<pDynaCurrFrame->imagePoints.size()<<" "<<pDynaCurrFrame->objectPoints.size()<<" "<<pDynaCurrFrame->Pose;
                WriteLog(ss.str());
            }
            */

            cv::Mat Ptcw = cv::Mat(4,3, CV_32FC1, posedata);
            cv::Mat Rcw = Ptcw.rowRange(0,3).colRange(0,3);
            cv::Mat tcw = Ptcw.row(3).t();
            cv::Mat Pcw = cv::Mat::eye(4,4,CV_32FC1);
            Rcw.copyTo(Pcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Pcw.rowRange(0,3).col(3));
            pDynaCurrFrame->Pco = Pcw * pDynaRefFrame->Pco;

            int nPnP = 0;
            if(nMatchedObject>10){
                 nPnP = pDynamicEstimator->DynamicPoseEstimation(pDynaCurrFrame);
                //칼만필터 업데이트도 필요함.
                //포즈 갱신하기
            }
            objID = pDynaRefFrame->mnObjectId;
            cv::Mat Pco = cv::Mat(4,3, CV_32FC1, oposedata);
            cv::Mat Rco = pDynaCurrFrame->Pco.rowRange(0,3).colRange(0,3);
            cv::Mat tco = pDynaCurrFrame->Pco.rowRange(0,3).col(3).t();
            Rco.copyTo(Pco.rowRange(0, 3));
            tco.copyTo(Pco.row(3));

            /*
            std::chrono::high_resolution_clock::time_point s3 = std::chrono::high_resolution_clock::now();
            auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s3 - s2).count();
            float t = d / 1000.0;
            {
                auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s3 - s2).count();
                float t = d / 1000.0;
                std::stringstream ss;
                ss<<"PnP Estimation= "<<nPnP<<" "<<t<<std::endl;
            }
            std::stringstream ss;
            ss<<"Localizatio=OBJ="<<id<<"="<<nMatchedObject<<" "<<nPnP<<", "<<t<<std::endl;
            WriteLog(ss.str());
            */
            return nPnP;
        }
        return -1;
    }

    void UpdateDynamicObjectPoints(VECTOR3* addr, int size){
        for(int i = 0; i < size; i++){
            VECTOR3& tempVec = addr[i];
            cv::Point3f objPt= pDynaCurrFrame->objectPoints[i];
            tempVec.x = objPt.x;
            tempVec.y = objPt.y;
            tempVec.z = objPt.z;
        }
    }

    bool Localization(void* texdata, void* posedata, int id, double ts, int nQuality, bool bNotBase, bool bTracking, bool bVisualization){

        bool res = true;

        //유니티에서 온 프레임
        /*
        cv::Mat ori_frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, texdata);
        cv::Mat frame = ori_frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
        frame.convertTo(frame, CV_8UC3);

        //NDK에서 이용하는 이미지로 변환
        //cv::flip(frame, frame,0);
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY
        */
        bool bTrack = false;
        nMatch = -1;
        if(bTracking){

            ////이전 프레임 해제
            if(pPrevFrame)
                delete pPrevFrame;
            pPrevFrame = pCurrFrame;
            pCurrFrame = new EdgeSLAM::Frame(gray, pCamera, id);

            if(!bSetReferenceFrame || !bSetLocalMap)
                return false;
//WriteLog("LOCALIZATION START");
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
//WriteLog("LOCALIZATION::TrackWithPrevFrame");
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
//WriteLog("2");
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
//WriteLog("3");
            if (bTrack) {
                pTracker->mTrackState = EdgeSLAM::TrackingState::Success;
                cv::Mat T = pCurrFrame->GetPose();
                pCameraPose->SetPose(T);
                pMotionModel->update(T);
                //POOL->EnqueueJob(SendDevicePose, id, ts, T.clone());

                ////유니티에 카메라 포즈 복사
                ////R과 카메라 센터
                //cv::Mat t = T.col(3).rowRange(0, 3).t();
                cv::Mat P = cv::Mat(4,3, CV_32FC1, posedata);
                T.rowRange(0, 3).colRange(0, 3).copyTo(P.rowRange(0, 3));
                //센터 또는 t임.
                cv::Mat Ow = pCameraPose->GetCenter().t();
                //cv::Mat Ow = T.col(3).rowRange(0,3).t();
                Ow.copyTo(P.row(3));
            }
            else {
                //WriteLog("Tracking::Failed", std::ios::trunc);
                pTracker->mTrackState = EdgeSLAM::TrackingState::Failed;
                pCameraPose->Init();
                pMotionModel->reset();

            }
//WriteLog("4");
        }
//WriteLog("Localization::End");

/*
		{
		    std::stringstream ss;
		    ss<<"Localizatio="<<id<<"="<<nMatch<<", "<<std::endl;
		    WriteLog(ss.str());
		}
*/


		bool bres = bTrack;
		//if(!bTracking){
        //    bres = bResReference;
		//}
        return bres;
    }
    /*
    bool Localization(void* texdata, void* posedata, int id, double ts, int nQuality, bool bNotBase, bool bTracking, bool bVisualization){
        POOL->EnqueueJob(_Localization, texdata, posedata, id, ts, nQuality, bNotBase, bTracking, bVisualization);
        return true;
    }
    */
}
