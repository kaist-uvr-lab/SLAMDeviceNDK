#include "UnityLibrary.h"
#include "Camera.h"
#include "ORBDetector.h"
#include "DBoW3.h"

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
#include <atomic>
#include <thread>

#include "WebAPI.h"
//#pragma comment(lib, "ws2_32")

//���� https://darkstart.tistory.com/42
extern "C" {
    std::string strSource;
    std::vector<int> param = std::vector<int>(2);

	EdgeSLAM::Frame* pCurrFrame = nullptr;
	EdgeSLAM::Frame* pPrevFrame = nullptr;
	EdgeSLAM::Camera* pCamera;
	EdgeSLAM::ORBDetector* pDetector;
	EdgeSLAM::MotionModel* pMotionModel;
	EdgeSLAM::CameraPose* pCameraPose;
	EdgeSLAM::Tracker* pTracker;
	DBoW3::Vocabulary* pVoc;
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
	std::string EdgeSLAM::SearchPoints::LogFile;

    std::map<int, cv::Mat> mapContentInfos;
    std::mutex mMutexContentInfo;

    //플로우를 이용해서 데이터 트랜스퍼
    EdgeSLAM::GridFrame* pPrevGrid = nullptr;
    EdgeSLAM::GridFrame* pCurrGrid = nullptr;
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
    std::atomic<int> mnCurrFrameID, mnLastUpdatedID;

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

    void LoadVocabulary(){
        std::string strVoc = strPath+"/orbvoc.dbow3";//std::string(vocName);
        pVoc = new DBoW3::Vocabulary();
        pVoc->load(strVoc);
    }

    void SetInit(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, int nfeature, int nlevel, float fscale, int nSkip, int nKFs) {//char* vocName,

        ofile.open(strLogFile.c_str(), std::ios::trunc);
        ofile<<"start\n";
        ofile.close();

        param[0] = cv::IMWRITE_JPEG_QUALITY;
        EdgeSLAM::SearchPoints::LogFile = strLogFile;

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

        //WSAData wsaData;
        //int code = WSAStartup(MAKEWORD(2, 2), &wsaData);
        return;

        pTracker = new EdgeSLAM::Tracker();
        pMap = new EdgeSLAM::Map();
        pCurrFrame = nullptr;
        pPrevFrame = nullptr;
        //pRefFrame = nullptr;
        //mapMapPoints.clear();
        //EdgeSLAM::RefFrame::MapPoints.clear();
        //mapFrames.clear();
        //mapRefFrames.clear();
        EdgeSLAM::RefFrame::nId = 0;

        pCameraPose = new EdgeSLAM::CameraPose();
        pMotionModel = new EdgeSLAM::MotionModel();

        EdgeSLAM::Tracker::Detector = pDetector;
        EdgeSLAM::SearchPoints::Detector = pDetector;
        EdgeSLAM::Frame::detector = pDetector;
        EdgeSLAM::MapPoint::Detector = pDetector;
        //EdgeSLAM::RefFrame::MapPoints = mapMapPoints;

        pTracker->mTrackState = EdgeSLAM::TrackingState::NotEstimated;
        mapContentInfos.clear();
        printf("init=end\n");
    }

    void ConnectDevice() {

        pTracker = new EdgeSLAM::Tracker();
        pTracker->path = strLogFile;

        pMap = new EdgeSLAM::Map();

        if(pPrevFrame)
            delete pPrevFrame;
        if(pCurrFrame)
            delete pCurrFrame;
        pCurrFrame = nullptr;
        pPrevFrame = nullptr;

        EdgeSLAM::RefFrame::nId = 0;
        pCameraPose = new EdgeSLAM::CameraPose();
        pMotionModel = new EdgeSLAM::MotionModel();
        EdgeSLAM::Tracker::Detector = pDetector;
        EdgeSLAM::SearchPoints::Detector = pDetector;
        EdgeSLAM::RefFrame::detector = pDetector;
        EdgeSLAM::Frame::detector = pDetector;
        EdgeSLAM::MapPoint::Detector = pDetector;
        EdgeSLAM::RefFrame::MAP = pMap;
        EdgeSLAM::MapPoint::MAP = pMap;

        //EdgeSLAM::LocalMap::logFile = strLogFile;
        //EdgeSLAM::RefFrame::MapPoints = mapMapPoints;

        pTracker->mTrackState = EdgeSLAM::TrackingState::NotEstimated;
        mapContentInfos.clear();
        LabeledImage = cv::Mat::zeros(0,0,CV_8UC4);

        if(pPrevGrid)
            delete pPrevGrid;
        pPrevGrid = nullptr;
        if(pCurrGrid)
            delete pCurrGrid;
        pCurrGrid = nullptr;
        auto mapGrids = cvGridFrames.Get();
        for(auto iter = mapGrids.begin(), iend = mapGrids.end(); iter != iend; iter++){
            delete iter->second;
        }
        cvGridFrames.Release();
    }

    void* imuAddr;
    void* accAddr;
    bool bIMU = false;
	void SetIMUAddress(void* addr, bool bimu){
        imuAddr = addr;
        bIMU = bimu;
	}

    int scale = 8;
    cv::Mat prevGray = cv::Mat::zeros(0,0,CV_8UC1);
    //https://learnopencv.com/optical-flow-in-opencv/

    void DenseFlow(int id, cv::Mat prev, cv::Mat curr){
        std::stringstream ss;
        ss<<"DenseFlow::Start::!!!!!! "<<id;
        WriteLog(ss.str());
        std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();

        cv::Mat flow;
		cv::calcOpticalFlowFarneback(prev, curr, flow, 0.5, 3, 15, 3, 5, 1.1, 0);
        mnCurrFrameID = id;

        if(pPrevGrid){
            WriteLog("delete = dense = start");
            delete pPrevGrid;
            WriteLog("delete = dense = end");
        }
        pPrevGrid = pCurrGrid;
        pCurrGrid = new EdgeSLAM::GridFrame(flow.rows, flow.cols);
        //flow update
        if(pPrevGrid){
            for(int y = 0, rows = flow.rows; y < rows; y++){
                for(int x = 0, cols = flow.cols; x < cols; x++){
                    cv::Point2f pt1(x,y);
                    cv::Point2f pt2(x+flow.at<cv::Vec2f>(pt1).val[0], y+flow.at<cv::Vec2f>(pt1).val[1]);
                    if(pt2.x <= 0 || pt2.y <= 0 || pt2.x >= flow.cols-1 || pt2.y >= flow.rows-1){
                        continue;
                    }
                    auto pCell = pPrevGrid->mGrid[y][x];
                    if(!pCell){
                        continue;
                    }
                    pCurrGrid->mGrid[pt2.y][pt2.x] = pCell;
                }
            }
        }
        //WriteLog("b");
        for(int y = 0, rows = flow.rows; y < rows; y++){
            for(int x = 0, cols = flow.cols; x < cols; x++){
                auto pCell = pCurrGrid->mGrid[y][x];
                if(!pCell){
                    pCell = new EdgeSLAM::GridCell();
                    pCurrGrid->mGrid[y][x] = pCell;
                }
                pCell->AddObservation(pCurrGrid, y*rows+x);
            }
        }
        //flow update
        //WriteLog("c");
        if(id % mnSkipFrame == 0){
            auto newPGrid = new EdgeSLAM::GridFrame(flow.rows, flow.cols);
            newPGrid->Copy(pCurrGrid);
            cvGridFrames.Update(id,newPGrid);
        }

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
            prevGray.release();
        }
        prevGray = gray_resized.clone();
        /*
        std::stringstream ss;
        ss<<strPath<<"/color.jpg";
        cv::imwrite(ss.str(), frame);
        ss.str("");
        ss<<strPath<<"/gray.jpg";
        cv::imwrite(ss.str(), gray);
        */

        if(id % mnSkipFrame == 0){
            pMap->AddImage(gray.clone(), id);
        }
        if(pPrevFrame)
            delete pPrevFrame;
        pPrevFrame = pCurrFrame;
        pCurrFrame = new EdgeSLAM::Frame(gray, pCamera, id);
        pCurrFrame->logfile = strLogFile;

        //ofile<<"SetFrame="<<t_total<<std::endl;
        //ofile.close();
        //WriteLog("SetFrame::End");

        return pCurrFrame->N;
    }
    void Parsing(int id, std::string key, cv::Mat data, bool bTracking){
        if(key == "ReferenceFrame"){
            if(bTracking)
                CreateReferenceFrame(id, data);
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
        //WriteLog("SetReference::Start!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        //cv::Mat f1 = GetDataFromUnity("ReferenceFrame");
        //ReleaseUnityData("ReferenceFrame");

        auto pRefFrame = new EdgeSLAM::RefFrame(pCamera, (float*)data.data);
        cv::Mat img = pMap->GetImage(id);
        pDetector->Compute(img, cv::Mat(), pRefFrame->mvKeys, pRefFrame->mDescriptors);
        //std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pRefFrame->mDescriptors);
        //pVoc->transform(vCurrentDesc, pRefFrame->mBowVec, pRefFrame->mFeatVec, 4);
        pRefFrame->logfile = strLogFile;
        pRefFrame->UpdateMapPoints();
        auto pPrefRef = pMap->GetReferenceFrame();
        pRefFrame->mpParent = pPrefRef;

        ////local map 갱신
        EdgeSLAM::LocalMap* pLocal = new EdgeSLAM::LocalMap();
        std::set<EdgeSLAM::MapPoint*> spMPs;
        int nkf = 0;
        EdgeSLAM::RefFrame* ref = nullptr;
        EdgeSLAM::RefFrame* last = nullptr;

        for(ref = pRefFrame; ref; ref = ref->mpParent, nkf++){
            if(!ref || nkf >= mnKeyFrame){
                break;
            }
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
            last = ref;
        }

        ////delete ref frame
        if(ref){
            auto vpMPs = ref->mvpMapPoints;
            for(int i =0; i < ref->N; i++){
                auto pMP = vpMPs[i];
                if(!pMP || pMP->isBad()){
                    continue;
                }
                pMP->EraseObservation(ref);
            }
            last->mpParent = nullptr;
            delete ref;
            //ofile<<"delete end"<<std::endl;
        }

        pMap->SetReferenceFrame(pRefFrame);
        pMap->SetLocalMap(pLocal);
        //f1.release();
        //WriteLog("SetReference::End!!!!!!!!!!!!!!!!!!!");
    }

    void SetReferenceFrame(int id) {
        //POOL->EnqueueJob(CreateReferenceFrame, id);
        //CreateReferenceFrame(id);
	}

	void AddObjectInfo(int id){

        std::stringstream ss;
        ss<<"ObjectProcessing::Start::!!!!!! "<<id;
        WriteLog(ss.str());
        std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();

        //std::unique_lock<std::mutex> lock(mMutexObjectInfo);
        cv::Mat tdata = GetDataFromUnity("ObjectDetection");
        ReleaseUnityData("ObjectDetection");
        int n = tdata.rows/24;
        cv::Mat obj = cv::Mat(n, 6, CV_32FC1, tdata.data);

        if(cvGridFrames.Count(id)){
            auto pGridFrame = cvGridFrames.Get(id);
            int SX = pGridFrame->mGrid[0].size()-1;
            int SY = pGridFrame->mGrid.size()-1;

            ss.str("");
            ss<<"ObjectProcessing::"<<SX<<" "<<SY;
            WriteLog(ss.str());

            for(int j = 0, jend = obj.rows; j < jend ;j++){
                cv::Point2f left(obj.at<float>(j, 2), obj.at<float>(j, 3));
                cv::Point2f right(obj.at<float>(j, 4), obj.at<float>(j, 5));
                int label = (int)obj.at<float>(j, 0)+1;
                for(int x = left.x, xend = right.x; x < xend; x+=scale){
                    for(int y = left.y, yend = right.y; y < yend; y+=scale){
                        int sx = x/scale;
                        int sy = y/scale;
                        if (sx < 0 || sy < 0 || sx > SX || sy > SY )
                            continue;
                        auto pCell = pGridFrame->mGrid[sy][sx];
                        if(pCell)
                            pCell->mpObject->Update(label);
                    }
                }
            }
            cvGridFrames.Erase(id);
            if(pGridFrame){
                WriteLog("Object = Delete = Start");
                delete pGridFrame;
                WriteLog("Object = Delete = End");
            }
        }

        std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
        float t = d / 1000.0;
        ss.str("");
        ss<<"ObjectProcessing::End::!!!!!!  "<<id<<"="<<t;
        WriteLog(ss.str());

	}
    void AddObjectInfos(int id){
        POOL->EnqueueJob(AddObjectInfo, id);
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
    void WriteLog(std::string str){
        //std::string log(data);
        std::unique_lock<std::mutex> lock(mMutexLogFile);
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
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
        //cv::Mat(buffer).clone().data
        StoreData("Image", id, strSource, ts, buffer.data(), buffer.size());
    }

    bool Localization(void* data, int id, double ts, int nQuality, bool bTracking, bool bVisualization){

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        ofile.open(strLogFile.c_str(), std::ios::trunc);
        ofile<<"Localization::start\n";
        ofile.close();

        cv::Mat gray;
        bool res = true;

        //유니티에서 온 프레임
        cv::Mat ori_frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        cv::Mat frame = ori_frame.clone();

        ////convert color(a,r,g,b)->bgr
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGB);
        /*
        std::vector<cv::Mat> colors(4);
        cv::split(frame, colors);
        std::vector<cv::Mat> colors2(3);
        colors2[0] = colors[2];//a->3
        colors2[1] = colors[1];//r->2
        colors2[2] = colors[0];//g->1
        //colors2[3] = colors[0];//b->0
        cv::merge(colors2, frame);
        */
        frame.convertTo(frame, CV_8UC3);

        ////convert color(a,r,g,b)->bgr
        //NDK에서 이용하는 이미지로 변환

        cv::flip(frame, frame,0);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY


        if(id % mnSkipFrame == 0)
        {
            POOL->EnqueueJob(SendImage, id, ts, nQuality, frame);
        }

        ////데이터 트랜스퍼 용
        /*
        cv::Mat gray_resized;
        cv::resize(gray, gray_resized, gray.size() / scale);
        if(prevGray.rows > 0){
            //POOL->EnqueueJob(DenseFlow, id, prevGray.clone(), gray_resized.clone());
            prevGray.release();
        }
        prevGray = gray_resized.clone();
        */
        ////데이터 트랜스퍼 용
        if(!bTracking){
            return false;
        }

        if(id % mnSkipFrame == 0)
        {
            //그레이 이미지 기록
            pMap->AddImage(gray.clone(), id);
        }

        ////이전 프레임 해제
        if(pPrevFrame)
            delete pPrevFrame;
        pPrevFrame = pCurrFrame;
        pCurrFrame = new EdgeSLAM::Frame(gray, pCamera, id);
        pCurrFrame->logfile = strLogFile;

        //트래킹
        bool bTrack = false;
		int nMatch = -1;

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

			nMatch = pTracker->TrackWithPrevFrame(pPrevFrame, pCurrFrame, 100.0, 50.0);
            WriteLog("Tracker::Prev::123123");
			bTrack = nMatch >= 10;
			if (!bTrack) {
				EdgeSLAM::RefFrame* rf = pMap->GetReferenceFrame();
				if (rf) {
					nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame, 100.0, 50.0);
					bTrack = nMatch >= 10;
				}
			}
			WriteLog("Tracker::Prev::End");
		}

		if (bTrack) {
			EdgeSLAM::LocalMap* pLocal = new EdgeSLAM::LocalMap();
			pMap->GetLocalMap(pLocal);
			nMatch = 4;
			nMatch = pTracker->TrackWithLocalMap(pCurrFrame, pLocal, 100.0, 50.0);

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
			pTracker->mTrackState = EdgeSLAM::TrackingState::Failed;
			pMotionModel->reset();
		}

        //시각화
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
