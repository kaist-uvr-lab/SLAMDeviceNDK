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
#include <atomic>
//#pragma comment(lib, "ws2_32")

//���� https://darkstart.tistory.com/42
extern "C" {

    std::string strSource;
    std::vector<int> param = std::vector<int>(2);

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
    ThreadPool::ThreadPool* POOL = nullptr;

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
    std::atomic<int> mnSendedID, mnLastUpdatedID;

    std::ifstream inFile;
    char x[1000];

    std::ofstream ofile;
    std::string strLogFile;

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
        mnLastUpdatedID = -1;
        pTracker->mTrackState = EdgeSLAM::TrackingState::NotEstimated;
        LabeledImage = cv::Mat::zeros(0,0,CV_8UC4);

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

        //set data
        cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
        std::memcpy(temp.data, res.data(), n2);
        Parsing(id, key, temp, bTracking);

    }

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

    int mpid = 0;
    void UpdateLocalMap(int id, int n, void* data){
        /*
        //pts
        int nPointDataSize = n*12;
        cv::Mat pts_data = cv::Mat(n*3,1,CV_32FC1,data);
        //cv::Mat pts = cv::Mat(
        //desc
        int nDescDataSize = n*32;
        uchar* descptr = (uchar*)data+nDescDataSize;
        cv::Mat desc_data = cv::Mat(n*32,1,CV_8UC1,descptr);
        //std::memcpy(desc_data.data,data+nPointDataSize,nDescDataSize);

        std::vector<EdgeSLAM::MapPoint*> vpMPs = std::vector<EdgeSLAM::MapPoint*>(n, static_cast<EdgeSLAM::MapPoint*>(nullptr));
        for(int i = 0; i < n; i++){
            cv::Mat X = pts_data.rowRange(3*i,3*i+3);
            cv::Mat desc = desc_data.rowRange(32*i,32*i+32).t();
            int pid = ++mpid;
            auto pMP = new EdgeSLAM::MapPoint(pid, X.at<float>(0), X.at<float>(1), X.at<float>(2));
            pMP->SetDescriptor(desc);
            vpMPs[i] = pMP;
        }
        LocalMapPoints.set(vpMPs);
        */
        bReqLocalMap = false;
    }

    int CreateReferenceFrame(int id, float* data){
    //void CreateReferenceFrame(int id, const cv::Mat& data){
        //WriteLog("SetReference::Start!!!!!!!!!!!!!!!!!!!!!!!!!!!!");//, std::ios::trunc
        //cv::Mat f1 = GetDataFromUnity("ReferenceFrame");
        //ReleaseUnityData("ReferenceFrame");
        float* tdata = data;
        int N = (int)tdata[0];
        if(N > 30){
//WriteLog("CreateReference Start");
            auto pRefFrame = new EdgeSLAM::RefFrame(pCamera, tdata);

            if(!mapSendedImages.Count(id))
                return dequeRefFrames.size();
            cv::Mat img = mapSendedImages.Get(id);
            mapSendedImages.Erase(id);

            pDetector->Compute(img, cv::Mat(), pRefFrame->mvKeys, pRefFrame->mDescriptors);
            //std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pRefFrame->mDescriptors);
            //pVoc->transform(vCurrentDesc, pRefFrame->mBowVec, pRefFrame->mFeatVec, 4);
            nRefMatches = pRefFrame->mvpMapPoints.size();
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
            if(!bSetReferenceFrame)
                bSetReferenceFrame = true;
            LocalMapPoints.set(vecMPs);
            //f1.release();
        }
        //WriteLog("SetReference::End!!!!!!!!!!!!!!!!!!!");
        return dequeRefFrames.size();
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
        /*
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        float t = d / 1000.0;
        //testUploadCount[nQuality]++;
        //testUploadTime[nQuality]=testUploadTime[nQuality]+t;
        std::stringstream ss;
        ss<<nQuality;
        int c = testUploadCount.Get(ss.str())+1;
        float total = testUploadTime.Get(ss.str())+t;
        testUploadCount.Update(ss.str(), c);
        testUploadTime.Update(ss.str(), total);
        */
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

        bReqLocalMap = true;
        nLastKeyFrameId = fid;
        return (c1a||c1b)&&c2;
    }


    bool Localization(void* texdata, void* posedata, int id, double ts, int nQuality, bool bTracking, bool bVisualization){

        bool res = true;

        //유니티에서 온 프레임
        cv::Mat ori_frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, texdata);
        cv::Mat frame = ori_frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
        frame.convertTo(frame, CV_8UC3);

        //NDK에서 이용하는 이미지로 변환
        //cv::flip(frame, frame,0);
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY
//WriteLog("1");
        if(id % mnSkipFrame == 0)
        {
            //POOL->EnqueueJob(SendImage, id, ts, nQuality, frame);
            mapSendedImages.Update(id,gray.clone());
            mnSendedID = id;
        }

        ////이 위 까지는 서버 전송을 위해 무조건 동작해야 함.

//WriteLog("2");
        bool bTrack = false;
        nMatch = -1;
        if(bTracking){

            ////이전 프레임 해제
            if(pPrevFrame)
                delete pPrevFrame;
            pPrevFrame = pCurrFrame;
            pCurrFrame = new EdgeSLAM::Frame(gray, pCamera, id);
//WriteLog("3");
            if(!bSetReferenceFrame)
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
//WriteLog("1");
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
//WriteLog("2:a");
                nMatch = pTracker->TrackWithLocalMap(pCurrFrame, vecLocalMPs, 100.0, 50.0);
//WriteLog("2:b");
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
            /*
            if (bTrack) {

                cv::Mat T = pCurrFrame->GetPose();
                cv::Mat R = T.colRange(0, 3).rowRange(0, 3);
                cv::Mat t = T.col(3).rowRange(0, 3);
                cv::Mat K = pCamera->K.clone();

                {
                    std::stringstream ss;
                    ss<<T<<K<<std::endl;
                    WriteLog(ss.str());
                }

                for (int i = 0; i < pCurrFrame->mvKeys.size(); i++) {
                    auto pMP = pCurrFrame->mvpMapPoints[i];
                    int r = 2;
                    if (pMP && !pMP->isBad())
                    {
                        cv::Mat x3D = pMP->GetWorldPos();
                        cv::Mat proj= K*(R*x3D + t);
                        float d = proj.at<float>(2);
                        cv::Point2f pt(proj.at<float>(0) / d, proj.at<float>(1) / d);
                        cv::circle(frame, pt, r, cv::Scalar(255,0,255), -1);
                    }
                }//for frame

                std::stringstream ss;
                ss<<strPath<<"/color.jpg";
                cv::imwrite(ss.str(), frame);
            }//if
            */
        }
//WriteLog("7");

		/*
		{
		    std::stringstream ss;
		    ss<<"Localizatio="<<id<<"="<<nMatch<<", "<<t<<std::endl;
		    WriteLog(ss.str());
		}
		*/
		bool bres = bTrack;
		//if(!bTracking){
        //    bres = bResReference;
		//}
        return bres;
    }
}
