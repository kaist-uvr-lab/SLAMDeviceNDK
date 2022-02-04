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
#include "LocalMap.h"
#include "GridCell.h"
#include "GridFrame.h"

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

    std::map<int, int> testUploadCount;
    std::map<int, float> testUploadTime;

    std::map<std::string, int> testDownloadCount;
    std::map<std::string, float> testDownloadTime;

    std::string strSource;
    std::vector<int> param = std::vector<int>(2);

    ConcurrentMap<int, cv::Mat> mapSendedImages;
    ConcurrentDeque<EdgeSLAM::RefFrame*> dequeRefFrames;

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

    std::map<int, cv::Mat> mapContentInfos;
    std::mutex mMutexContentInfo;

    //플로우를 이용해서 데이터 트랜스퍼
    ConcurrentMap<int, EdgeSLAM::GridFrame*> cvGridFrames;

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

        dequeRefFrames.Release();
        mapSendedImages.Release();

        auto mapGrids = cvGridFrames.Get();
        for(auto iter = mapGrids.begin(), iend = mapGrids.end(); iter != iend; iter++){
            delete iter->second;
        }
        cvGridFrames.Release();

        ////save txt
        std::string testfile = strPath+"/Experiment/upload.txt";
        std::ofstream ofile2;
        ofile2.open(testfile.c_str(), std::ios::trunc);
        for(int i = 10; i <= 100; i+=10){
            std::stringstream ss;
            ss<<i<<" "<<testUploadCount[i]<<" "<<testUploadTime[i]<<" "<<testUploadTime[i]/testUploadCount[i]<<std::endl;
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

        EdgeSLAM::RefFrame::nId = 0;
        //pCameraPose = new EdgeSLAM::CameraPose();
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
        mapContentInfos.clear();
        LabeledImage = cv::Mat::zeros(0,0,CV_8UC4);

        ////load txt
        {
            std::string s;
            std::string testfile = strPath+"/Experiment/upload.txt";
            std::ifstream ifile;
            ifile.open(testfile);
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
    cv::Mat prevGray = cv::Mat::zeros(0,0,CV_8UC1);
    //https://learnopencv.com/optical-flow-in-opencv/

    void DenseFlow(int id, cv::Mat prev, cv::Mat curr){

        std::stringstream ss;
        ss<<"DenseFlow::Start::!!!!!! "<<id;
        WriteLog(ss.str());
        std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();

        cv::Mat flow;
		cv::calcOpticalFlowFarneback(prev, curr, flow, 0.5, 3, 15, 3, 5, 1.1, 0);

        auto pCurrGrid = new EdgeSLAM::GridFrame(flow.rows, flow.cols);
        pCurrGrid->mFlow = flow.clone();
        cvGridFrames.Update(id,pCurrGrid);

        std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
		auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
		float t = d / 1000.0;

        /*
        // Visualization part
        cv::Mat flow_parts[2];
        cv::split(flow, flow_parts);
        // Convert the algorithm's output into Polar coordinates
        cv::Mat magnitude, angle, magn_norm;
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));
        // Build hsv image
        cv::Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32FC1);
        _hsv[2] = magn_norm;
        cv::merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8UC1, 255.0);
        // Display the results
        cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
		ss.str("");
        ss<<strPath<<"/denseflow.jpg";
        cv::imwrite(ss.str(), bgr);
        */

        ss.str("");
		ss<<"DenseFlow::End::!!!!!! "<<id<<"="<<t;
        WriteLog(ss.str());

    }

    int SetFrame(void* data, int id, double ts) {
        //ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        //ofile<<"Frame start"<<std::endl;
        //WriteLog("SetFrame::Start");
        cv::Mat gray;
        bool res = true;
        cv::Mat frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        //cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGBA);
        cv::flip(frame, frame,0);
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY

        cv::Mat gray_resized;
        cv::resize(gray, gray_resized, gray.size() / scale);
        if(prevGray.rows > 0){
            //POOL->EnqueueJob(DenseFlow, id, prevGray.clone(), gray_resized.clone());
            DenseFlow(id, prevGray.clone(), gray_resized.clone());
            prevGray.release();
        }
        prevGray = gray_resized.clone();

        if(id % mnSkipFrame == 0){
            mapSendedImages.Update(id,gray.clone());
        }
        if(pPrevFrame)
            delete pPrevFrame;
        pPrevFrame = pCurrFrame;
        pCurrFrame = new EdgeSLAM::Frame(gray, pCamera, id);

        return pCurrFrame->N;
    }
    void Parsing(int id, std::string key, cv::Mat data, bool bTracking){
        if(key == "ReferenceFrame"){
            if(bTracking)
                CreateReferenceFrame(id, data);
        }else if(key == "ObjectDetection"){
            AddObjectInfo(id, data);
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
        testDownloadCount[key]++;
        testDownloadTime[key]+=t;

        //set data
        cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
        std::memcpy(temp.data, res.data(), n2);
        Parsing(id, key, temp, bTracking);
        temp.release();
        //WriteLog(ss.str());
    }

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

    void CreateReferenceFrame(int id, cv::Mat data){
        WriteLog("SetReference::Start!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        //cv::Mat f1 = GetDataFromUnity("ReferenceFrame");
        //ReleaseUnityData("ReferenceFrame");

        auto pRefFrame = new EdgeSLAM::RefFrame(pCamera, (float*)data.data);
        cv::Mat img = mapSendedImages.Get(id);
        mapSendedImages.Erase(id);
        pDetector->Compute(img, cv::Mat(), pRefFrame->mvKeys, pRefFrame->mDescriptors);
        //std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pRefFrame->mDescriptors);
        //pVoc->transform(vCurrentDesc, pRefFrame->mBowVec, pRefFrame->mFeatVec, 4);
        pRefFrame->logfile = strLogFile;
        pRefFrame->UpdateMapPoints();
        dequeRefFrames.push_back(pRefFrame);

        ////local map 갱신
        EdgeSLAM::LocalMap* pLocal = new EdgeSLAM::LocalMap();
        std::set<EdgeSLAM::MapPoint*> spMPs;
        WriteLog("Reference::Delete::Start");
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
        WriteLog("Reference::Delete::End");
        auto vecRefFrames = dequeRefFrames.get();
        for(int i = 0; i < vecRefFrames.size(); i++){
            auto ref = vecRefFrames[i];
            auto vpMPs = ref->mvpMapPoints;
            for(int i =0; i < ref->N; i++){
                auto pMP = vpMPs[i];
                if(!pMP || pMP->isBad() || spMPs.count(pMP)){
                    continue;
                }
                if(pRefFrame->is_in_frustum(pMP, 0.5)){
                    pLocal->mvpMapPoints.push_back(pMP);
                    spMPs.insert(pMP);
                }
            }
        }
        WriteLog("Reference::UpdateLocalMap::End");
        pMap->SetReferenceFrame(pRefFrame);
        pMap->SetLocalMap(pLocal);
        //f1.release();
        WriteLog("SetReference::End!!!!!!!!!!!!!!!!!!!");
    }

    void SetReferenceFrame(int id) {
        //POOL->EnqueueJob(CreateReferenceFrame, id);
        //CreateReferenceFrame(id);
	}
	void VisualizeObjectFlow(cv::Mat& img, int endID){
        int startID = mnLastUpdatedID.load();
        if(startID < 0)
            return;
        int SX = mnWidth/scale-1;
        int SY = mnHeight/scale-1;

        auto flowGrids = cvGridFrames.Get();
        auto vecCells  = flowGrids[startID]->vecCells.get();
        //auto vecCells2 = flowGrids[endID]->vecCells.get();
        bool bUpdate = false;//endID%mnSkipFrame == 0 && vecCells2.size() == 0;

        for(int i = startID+1; i <= endID; i++){
            std::stringstream ss;
            ss<<"vis = "<<i<<" = "<<startID<<" "<<endID;
            WriteLog(ss.str());
            if(!flowGrids.count(i))
                continue;
            auto flow = flowGrids[i]->mFlow;
            for(int j = 0, jend = vecCells.size(); j < jend; j++){
                auto cell = vecCells[j];
                auto pt = cell.pt;
                if(pt.x < 0.0)
                    continue;
                if(pt.x >= SX-1 || pt.y >= SY-1 || pt.x <=1 || pt.y <= 1)
                {
                   pt.x = -1.0;
                }else{
                   cv::Vec2f val = flow.at<cv::Vec2f>(pt);
                   if(val.val[0] == 0.0 && val.val[1] == 0.0)
                   {
                        pt.x = -1.0;
                   }else{
                    pt.x += val.val[0];
                    pt.y += val.val[1];
                   }

                }
                cell.pt = pt;
                vecCells[j] = cell;
            }
        }
        int n = 0;
        for(int i = 0, iend = vecCells.size(); i < iend; i++){
            auto pt = vecCells[i].pt;
            if(pt.x < 0.0)
                continue;
            n++;
            cv::circle(img, pt*scale, 4, cv::Scalar(255, 0, 0, 255));
            if(bUpdate)
                flowGrids[endID]->vecCells.push_back(vecCells[i]);
        }
        std::stringstream ss;
        ss<<"VIS = "<<n<<" "<<vecCells.size();
        WriteLog(ss.str());
	}

	void AddObjectInfo(int id, cv::Mat tdata){

        /*
        std::stringstream ss;
        ss<<"ObjectProcessing::Start::!!!!!! "<<id;
        WriteLog(ss.str());
        std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();
        */
        //std::unique_lock<std::mutex> lock(mMutexObjectInfo);

        int n = tdata.rows/24;
        cv::Mat obj = cv::Mat(n, 6, CV_32FC1, tdata.data);

        auto flowGrids = cvGridFrames.Get();
        int nMaxID = mnSendedID.load();

        if(flowGrids.size() == 0)
            return;

        int SX = mnWidth/scale-1;
        int SY = mnHeight/scale-1;

        //id 프레임에 포인트 추가 현재 진행 된 프레임.
        //flow는 id-1을 id로 옮김.
        //따라서, id를 id+1에 옮기기 위해서는 flow+1이 필요함.
        std::vector<EdgeSLAM::GridCell> vecCells;
        for(int j = 0, jend = obj.rows; j < jend ;j++){

            cv::Point2f left(obj.at<float>(j, 2),  obj.at<float>(j, 3));
            cv::Point2f right(obj.at<float>(j, 4), obj.at<float>(j, 5));

            for(int x = left.x, xend = right.x; x < xend; x+=scale){
                for(int y = left.y, yend = right.y; y < yend; y+=scale){
                    int sx = x/scale;
                    int sy = (mnHeight-y)/scale;
                    if (sx <= 0 || sy <= 0 || sx >= SX || sy >= SY )
                        continue;
                    cv::Point2f pt = cv::Point2f(sx,sy);

                    EdgeSLAM::GridCell cell;
                    cell.pt = pt;
                    flowGrids[id]->vecCells.push_back(cell);
                }
            }
        }
        //기존 프레임에서 정보 추가
        /*
        for(int j = 0; j < flowGrids[id]->vecCells.size(); j++){
            auto cell = flowGrids[id]->vecCells[j];
            auto pt = cell.pt;
            if(pt.x < 0.0)
                continue;
            vecCells.push_back(cell);
        }
        flowGrids[id]->vecCells.clear();
        */
        ////중복 체크와 인식 정보 갱신

        /*
        for(int i = id+1; i <= nMaxID; i++){
            int tempCell = 0;
            if(!flowGrids.count(i))
                continue;
            auto flow = flowGrids[i]->mFlow;

            for(int j = 0; j < vecCells.size(); j++){
                auto pt = vecCells[j].pt;
                if(pt.x < 0.0)
                    continue;
                if(pt.x >= SX || pt.y >= SY || pt.x <=0 || pt.y <= 0)
                {
                    pt.x = -1.0;
                }else{
                    cv::Vec2f val = flow.at<cv::Vec2f>(pt);
                    pt.x += val.val[0];
                    pt.y += val.val[1];
                    tempCell++;
                }
                vecCells[j].pt = pt;
            }
            {
                std::stringstream ss;
                ss<<"OBJ=id="<<i<<" "<<tempCell<<std::endl;
                WriteLog(ss.str());
            }
            if(i % mnSkipFrame != 0){
                continue;
            }
            for(int j = 0; j < flowGrids[i]->vecCells.size(); j++){
                auto cell = flowGrids[i]->vecCells[j];
                auto pt = cell.pt;
                if(pt.x < 0.0)
                    continue;
                if(pt.x >= SX || pt.y >= SY || pt.x <=0 || pt.y <= 0)
                {
                    pt.x = -1.0;
                }else{
                    cv::Vec2f val = flow.at<cv::Vec2f>(pt);
                    pt.x += val.val[0];
                    pt.y += val.val[1];
                }
                cell.pt = pt;
                vecCells.push_back(cell);
            }
        }
        */

        /*
        ////추가
        int nCell = 0;
        for(int j = 0; j < vecCells.size(); j++){
            auto pt = vecCells[j].pt;
            if(pt.x < 0.0)
                continue;
            nCell++;
            flowGrids[id]->vecCells.push_back(vecCells[j]);
        }
        */
        mnLastUpdatedID = id;
        for(int i = id; i < mnLastUpdatedID; i++){
            if(cvGridFrames.Count(i)){
                auto pFrame = cvGridFrames.Get(i);
                cvGridFrames.Erase(i);
                delete pFrame;
            }
        }

        /*
        std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
        float t = d / 1000.0;
        ss.str("");
        ss<<"ObjectProcessing::End::!!!!!!  "<<id<<"="<<t;
        WriteLog(ss.str());
        */
	}
    void AddObjectInfos(int id){
        //POOL->EnqueueJob(AddObjectInfo, id);
    }

    void AddContent(int id, float x, float y, float z){
        cv::Mat pos = (cv::Mat_<float>(3, 1) <<x, y, z);
		std::unique_lock<std::mutex> lock(mMutexContentInfo);
		mapContentInfos.insert(std::make_pair(id, pos));
	}

	void AddContentInfo(int id, float x, float y, float z) {
        POOL->EnqueueJob(AddContent, id, x, y, z);
	}



	bool Track(void* data) {
        //ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        //ofile<<"tracking start"<<std::endl;
		bool bTrack = false;
		int nMatch = -1;
		ofile.open(strLogFile.c_str(), std::ios::trunc);
        ofile<<"Track::start\n";
        ofile.close();
        // WriteLog("Track::Start");
		if (pTracker->mTrackState == EdgeSLAM::TrackingState::NotEstimated || pTracker->mTrackState == EdgeSLAM::TrackingState::Failed) {
			EdgeSLAM::RefFrame* rf = pMap->GetReferenceFrame();

			if (rf) {
				//std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pCurrFrame->mDescriptors);
				//pVoc->transform(vCurrentDesc, pCurrFrame->mBowVec, pCurrFrame->mFeatVec, 4);
				nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame, 100.0, 50.0);
				bTrack = nMatch >= 10;

				if (bTrack) {
					pTracker->mnLastRelocFrameId = pCurrFrame->mnFrameID;
				}
                //ofile<<"ref matching = "<<nMatch<<" "<<rf->mDescriptors.rows<<" "<<rf->mvpMapPoints.size()<<" "<<rf->N<<"\n";
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
            WriteLog("predict");
            {
                std::stringstream ss;
                ss<<pMotionModel->predict();
                WriteLog(ss.str());
            }
            {
                cv::Mat T = pCurrFrame->GetPose();
                std::stringstream ss;
                ss<<T<<std::endl;
                WriteLog(ss.str());
            }
			//prev

			nMatch = pTracker->TrackWithPrevFrame(pPrevFrame, pCurrFrame, 100.0, 50.0);

			bTrack = nMatch >= 10;
			if (!bTrack) {
				EdgeSLAM::RefFrame* rf = pMap->GetReferenceFrame();
				if (rf) {
					//std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pCurrFrame->mDescriptors);
					//pVoc->transform(vCurrentDesc, pCurrFrame->mBowVec, pCurrFrame->mFeatVec, 4);
					nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame, 100.0, 50.0);
					bTrack = nMatch >= 10;
				}
			}
		}
		WriteLog("Track::1");
		if (bTrack) {
			//local map
			//ofile<<"local map matching = start"<<std::endl;
			EdgeSLAM::LocalMap* pLocal = new EdgeSLAM::LocalMap();
			//ofile<<"local map matching = 1"<<std::endl;
			pMap->GetLocalMap(pLocal);
			WriteLog("Track::1.1");
			//ofile<<"local map matching = 2 = "<<pLocal->mvpMapPoints.size()<<" "<<pLocal->mvpTrackPoints.size()<<std::endl;
			nMatch = 4;
			nMatch = pTracker->TrackWithLocalMap(pCurrFrame, pLocal, 100.0, 50.0);
            //ofile<<"local map matching = "<<nMatch<<"\n";

            WriteLog("Track::1.2");
			if (pCurrFrame->mnFrameID < pTracker->mnLastRelocFrameId + 30 && nMatch < 30) {
				bTrack = false;
				//bTrack = true;
			}
			else if (nMatch < 30) {
				bTrack = false;
			}
			else {
				bTrack = true;
			}
			delete pLocal;
		}
		if (bTrack) {
			pTracker->mTrackState = EdgeSLAM::TrackingState::Success;
			pMotionModel->update(pCurrFrame->GetPose());

            ////unity에 포즈 정보 갱신함.
			cv::Mat T = pCurrFrame->GetPose();
		    cv::Mat P = cv::Mat(4,3, CV_32FC1, data);
		    T.rowRange(0, 3).colRange(0, 3).copyTo(P.rowRange(0, 3));
            cv::Mat t = T.col(3).rowRange(0, 3).t();
            t.copyTo(P.row(3));
		}
		else {
			pTracker->mTrackState = EdgeSLAM::TrackingState::Failed;
			pMotionModel->reset();
		}
		
        //ofile<<"Tracking="<<t_total<<std::endl;
        //ofile.close();
		return bTrack;
	}
    std::mutex mMutexLogFile;
    void WriteLog(std::string str, std::ios_base::openmode mode){
        //std::string log(data);
        std::unique_lock<std::mutex> lock(mMutexLogFile);
        ofile.open(strLogFile.c_str(), mode);
        ofile<<str<<"\n";
        ofile.close();
    }

    bool VisualizeFrame(void* data){
        if (!pCurrFrame)
			return false;
        //이미 data는 setframe에서 변경되어서 옴
        //ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        //ofile<<"visualize start"<<std::endl;
        cv::Mat img = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);

        //(a,r,g,b) = cv ->(rgba)
        //cv::cvtColor(img, img, cv::COLOR_BGRA2RGBA);

        std::vector<cv::Mat> colors(4);
		cv::split(img, colors);
		std::vector<cv::Mat> colors2(4);
		colors2[0] = colors[3];
		colors2[1] = colors[0];//2
		colors2[2] = colors[1];//1
		colors2[3] = colors[2];//0
		cv::merge(colors2, img);

        /*
        cv::Mat obj;
        {
            std::unique_lock<std::mutex> lock(mMutexObjectInfo);
            obj = ObjectInfo.clone();
        }

        for(int j = 0, jend = obj.rows; j < jend ;j++){
            cv::Point2f left(obj.at<float>(j, 2), obj.at<float>(j, 3));
            cv::Point2f right(obj.at<float>(j, 4), obj.at<float>(j, 5));
            cv::rectangle(img,left, right, cv::Scalar(255, 255, 255, 255));
        }
        */

        /*
        if(pCurrGrid){
            WriteLog("vis::start");


             for(int y = 0, rows = pCurrGrid->mGrid.size(); y < rows; y++){
                for(int x = 0, cols = pCurrGrid->mGrid[0].size(); x < cols; x++){
                    auto pCell = pCurrGrid->mGrid[y][x];
                    if(pCell){
                        if(pCell->mpObject->GetLabel()>0){
                            cv::Point2f pt(x*scale,y*scale);
                            cv::circle(img, pt, 4, SemanticColors[pCell->mpObject->GetLabel()], -1);
                        }
                    }
                }
            }
            WriteLog("vis::end");
        }
        */

        cv::Mat labeled;
        {
            std::unique_lock<std::mutex> lock(mMutexSegmentation);
            labeled = LabeledImage.clone();
        }
        if(labeled.rows > 0)
            cv::addWeighted(img, 0.7, labeled, 0.3,0.0, img);

        bool res = false;
		if (pTracker->mTrackState == EdgeSLAM::TrackingState::Success) {

			for (int i = 0; i < pCurrFrame->N; i++) {
				auto pt = pCurrFrame->mvKeys[i].pt;
                //pt.y = mnHeight -pt.y;
                cv::circle(img, pt, 1, cv::Scalar(255, 0, 0, 255), -1);
				//if (!pCurrFrame->mvpMapPoints[i] || pCurrFrame->mvbOutliers[i])
				//	continue;
				//auto pt = pCurrFrame->mvKeys[i].pt;
				//pt.y = mnHeight -pt.y;
				//cv::circle(img, pt, 2, cv::Scalar(255, 255, 0, 255), -1);
			}
            /*
            cv::Mat obj;
            {
                std::unique_lock<std::mutex> lock(mMutexObjectInfo);
                obj = ObjectInfo.clone();
            }

            for(int j = 0, jend = obj.rows; j < jend ;j++){
                cv::Point2f left(obj.at<float>(j, 2), mnHeight-obj.at<float>(j, 3));
                cv::Point2f right(obj.at<float>(j, 4), mnHeight-obj.at<float>(j, 5));
                cv::rectangle(img,left, right, cv::Scalar(255, 255, 255, 255));
            }
            */
            /*
			std::map<int, cv::Mat> tempInfos;
			{
				std::unique_lock<std::mutex> lock(mMutexContentInfo);
				tempInfos = mapContentInfos;
			}
			cv::Mat T = pCurrFrame->GetPose();
			cv::Mat R = T.colRange(0, 3).rowRange(0, 3);
			cv::Mat t = T.col(3).rowRange(0, 3);
			cv::Mat K = pCamera->K.clone();
			for (auto iter = tempInfos.begin(), iend = tempInfos.end(); iter != iend; iter++) {
				auto pos = iter->second;
				cv::Mat temp = K * (R * pos + t);
				float depth = temp.at<float>(2);
				if (depth < 0.0)
					continue;
				cv::Point2f pt(temp.at<float>(0) / depth, mnHeight-temp.at<float>(1) / depth);
				cv::circle(img, pt, 3, cv::Scalar(255, 0, 255, 0), -1);
			}
			*/
		}
		cv::flip(img, img, 0);

		//cv::cvtColor(img, img, cv::COLOR_RGBA2BGRA);
		//memcpy(data, img.data, sizeof(cv::Vec4b) * img.rows * img.cols);

        /*
        std::stringstream ss;
        ss<<strPath<<"/color.jpg";
        cv::imwrite(ss.str(), img);
        */
        //ofile<<"visualize end"<<std::endl;
        //ofile.close();
		return true;
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
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        float t = d / 1000.0;
        testUploadCount[nQuality]++;
        testUploadTime[nQuality]=testUploadTime[nQuality]+t;

    }

    bool Localization(void* data, int id, double ts, int nQuality, bool bTracking, bool bVisualization){

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        //ofile.open(strLogFile.c_str(), std::ios::trunc);
        //ofile<<"Localization::start\n";
        //ofile.close();
        WriteLog("Localization::start");

        cv::Mat gray;
        bool res = true;

        //유니티에서 온 프레임
        cv::Mat ori_frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        cv::Mat frame = ori_frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
        frame.convertTo(frame, CV_8UC3);

        //NDK에서 이용하는 이미지로 변환
        cv::flip(frame, frame,0);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY

        if(id % mnSkipFrame == 0)
        {
            POOL->EnqueueJob(SendImage, id, ts, nQuality, frame);
            mapSendedImages.Update(id,gray.clone());
            mnSendedID = id;
        }

        ////데이터 트랜스퍼 용
        cv::Mat gray_resized;
        cv::resize(gray, gray_resized, gray.size() / scale);
        if(prevGray.rows > 0){
            /*
            if(bTracking)
                POOL->EnqueueJob(DenseFlow, id, prevGray.clone(), gray_resized.clone());
            else
                DenseFlow(id, prevGray.clone(), gray_resized.clone());
            prevGray.release();
            */
        }
        prevGray = gray_resized.clone();
        ////데이터 트랜스퍼 용

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
                WriteLog("Localization::TrackPrev::Start");
                nMatch = pTracker->TrackWithPrevFrame(pPrevFrame, pCurrFrame, 100.0, 50.0);
                WriteLog("Localization::TrackPrev::End");
                bTrack = nMatch >= 10;
                if (!bTrack) {
                    EdgeSLAM::RefFrame* rf = pMap->GetReferenceFrame();
                    if (rf) {
                        nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame, 100.0, 50.0);
                        bTrack = nMatch >= 10;
                    }
                }
                WriteLog("Localization::TrackPrev::End2");
            }

            if (bTrack) {
                EdgeSLAM::LocalMap* pLocal = new EdgeSLAM::LocalMap();
                pMap->GetLocalMap(pLocal);
                nMatch = 4;
                WriteLog("Localization::TrackLocalMap::Start");
                nMatch = pTracker->TrackWithLocalMap(pCurrFrame, pLocal, 100.0, 50.0);
                WriteLog("Localization::TrackLocalMap::End");
                if (pCurrFrame->mnFrameID < pTracker->mnLastRelocFrameId + 30 && nMatch < 30) {
                    bTrack = false;
                }
                else if (nMatch < 30) {
                    bTrack = false;
                }
                else {
                    bTrack = true;
                }
                delete pLocal;
            }
            if (bTrack) {
                pTracker->mTrackState = EdgeSLAM::TrackingState::Success;
                cv::Mat T = pCurrFrame->GetPose();
                pMotionModel->update(T);
                POOL->EnqueueJob(SendDevicePose, id, ts, T.clone());
            }
            else {
                WriteLog("Tracking::Failed", std::ios::trunc);
                pTracker->mTrackState = EdgeSLAM::TrackingState::Failed;
                pMotionModel->reset();

            }



        }

        //시각화
        if(bVisualization){

            std::vector<cv::Mat> colors(4);
            cv::split(ori_frame, colors);
            std::vector<cv::Mat> colors2(4);
            colors2[0] = colors[3];
            colors2[1] = colors[0];//2
            colors2[2] = colors[1];//1
            colors2[3] = colors[2];//0
            cv::merge(colors2, ori_frame);

            //VisualizeObjectFlow(ori_frame, id);

            /*
            if(pCurrGrid){
                 for(int y = 0, rows = pCurrGrid->mGrid.size(); y < rows; y++){
                    for(int x = 0, cols = pCurrGrid->mGrid[0].size(); x < cols; x++){
                        auto pCell = pCurrGrid->mGrid[y][x];

                        float fx = x+pCurrGrid->mFlow.at<cv::Vec2f>(y,x).val[0];
                        float fy = y+pCurrGrid->mFlow.at<cv::Vec2f>(y,x).val[1];

                        float nx = fx*scale;
                        float ny = fy*scale;

                        if(nx <= 0 || ny <= 0 || nx >= mnWidth-1 || ny >= mnHeight-1){
                            continue;
                        }

                        cv::Point2f pt(fx*scale,fy*scale);
                        //cv::circle(ori_frame, pt, 4, cv::Scalar(255,255,0,0), -1);

                        if(pCell){
                            if(pCell->mpObject->GetLabel()>0){
                                cv::circle(ori_frame, pt, 4, SemanticColors[pCell->mpObject->GetLabel()], -1);
                            }
                        }
                    }
                }
            }
            */
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		float t = d / 1000.0;
		{
		    std::stringstream ss;
		    ss<<"Localizatio="<<id<<"="<<nMatch<<", "<<t<<std::endl;
		    WriteLog(ss.str());
		}
        return bTrack;
    }
}
