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
#include "ThreadPool.h"
#include "ConcurrentVector.h"
#include "ConcurrentMap.h"
#include <atomic>
#include <thread>

//���� https://darkstart.tistory.com/42
extern "C" {

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

    std::map<int, cv::Mat> mapContentInfos;
    std::mutex mMutexContentInfo;

    ConcurrentVector<cv::Point2f> cvObjects;
    ConcurrentMap<int, cv::Mat> cvFlows;

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
            return UnityData[keyword];
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

		mnWidth  = _w;
		mnHeight = _h;
        mnSkipFrame = nSkip;
        mnKeyFrame = nKFs;

        mnFeature = nfeature;
        mnLevel = nlevel;
        mfScale = fscale;

        POOL = new ThreadPool::ThreadPool(24);

        pDetector = new EdgeSLAM::ORBDetector(nfeature,fscale,nlevel);
        pCamera = new EdgeSLAM::Camera(_w, _h, _fx, _fy, _cx, _cy, _d1, _d2, _d3, _d4);
        SemanticColorInit();
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

        cvFlows.Release();
        cvObjects.Release();
    }

    void* imuAddr;
    void* accAddr;
    bool bIMU = false;
	void SetIMUAddress(void* addr, bool bimu){
        imuAddr = addr;
        bIMU = bimu;
	}

    int scale = 16;
    cv::Mat prevGray = cv::Mat::zeros(0,0,CV_8UC1);
    //https://learnopencv.com/optical-flow-in-opencv/

    void DenseFlow(int id, cv::Mat prev, cv::Mat curr){
        std::stringstream ss;
        ss<<"DenseFlow::Start::!!!!!! "<<id;
        WriteLog(ss.str());
        std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();

        cv::Mat flow;
		cv::calcOpticalFlowFarneback(prev, curr, flow, 0.5, 3, 15, 3, 5, 1.1, 0);
        cvFlows.Update(id, flow);
        mnCurrFrameID = id;

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
		ss<<"DenseFlow::End::!!!!!! "<<t;
        WriteLog(ss.str());
    }

    int SetFrame(void* data, int id, double ts) {
        //ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        //ofile<<"Frame start"<<std::endl;
        WriteLog("SetFrame::Start");
        cv::Mat gray;
        bool res = true;
        cv::Mat frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        //cv::cvtColor(frame, frame, cv::COLO  R_BGRA2RGBA);
        cv::flip(frame, frame,0);
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);//COLOR_BGRA2GRAY, COLOR_RGBA2GRAY

        cv::Mat gray_resized;
        cv::resize(gray, gray_resized, gray.size() / scale);
        if(prevGray.rows > 0){
            POOL->EnqueueJob(DenseFlow, id, prevGray.clone(), gray_resized.clone());
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
        WriteLog("SetFrame::End");

        return pCurrFrame->N;
    }
    void SetLocalMap(){

    }

    void CreateReferenceFrame(int id){
        WriteLog("SetReference::Start!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        cv::Mat f1 = GetDataFromUnity("ReferenceFrame");
        auto pRefFrame = new EdgeSLAM::RefFrame(pCamera, (float*)f1.data);
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
                if(!pMP || spMPs.count(pMP)){
                    continue;
                }
                auto pTP = new EdgeSLAM::TrackPoint();
                 if(pRefFrame->is_in_frustum(pMP, pTP,0.5)){
                    pLocal->mvpMapPoints.push_back(pMP);
                    pLocal->mvpTrackPoints.push_back(pTP);
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
        f1.release();
        WriteLog("SetReference::End!!!!!!!!!!!!!!!!!!!");
    }

    void SetReferenceFrame(int id) {
        POOL->EnqueueJob(CreateReferenceFrame, id);
	}

	void AddObjectInfo(int id){

        std::stringstream ss;
        ss<<"ObjectProcessing::Start::!!!!!! "<<id;
        WriteLog(ss.str());
        std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();

        //std::unique_lock<std::mutex> lock(mMutexObjectInfo);
        cv::Mat tdata = GetDataFromUnity("ObjectDetection");
        int n = tdata.rows/24;
        cv::Mat obj = cv::Mat(n, 6, CV_32FC1, tdata.data);

        std::map<int, cv::Mat> flows = cvFlows.Get();
        cvFlows.Erase(id);
        int nMaxID = mnCurrFrameID.load();

        int SX = mnWidth/scale;
        int SY = mnHeight/scale;

        std::vector<cv::Point2f> vecPTs;

        for(int j = 0, jend = obj.rows; j < jend ;j++){
            cv::Point2f left(obj.at<float>(j, 2), obj.at<float>(j, 3));
            cv::Point2f right(obj.at<float>(j, 4), obj.at<float>(j, 5));

            for(int x = left.x, xend = right.x; x < xend; x+=scale){
                for(int y = left.y, yend = right.y; y < yend; y+=scale){
                    int sx = x/scale;
                    int sy = y/scale;
                    if (sx <= 0 || sy <= 0 || sx >= SX || sy >= SY )
                        continue;
                    vecPTs.push_back(cv::Point2f(sx,sy));
                }
            }
        }

        for(int i = id; i <= nMaxID; i++){
            if(!flows.count(i))
                continue;
            auto flow = flows[i];

            for(int j = 0; j < vecPTs.size(); j++){
                auto pt = vecPTs[j];
                if(pt.x < 0.0)
                    continue;
                if(pt.x >= SX || pt.y >= SY || pt.x <=0 || pt.y <= 0)
                {
                    vecPTs[j].x = -1.0;
                    //ss.str("");
                    //ss<<"A = "<<pt.x<<" "<<pt.y;
                    //WriteLog(ss.str());
                    continue;
                }
                vecPTs[j].x = flow.at<cv::Vec2f>(pt).val[0]+pt.x;
                vecPTs[j].y = flow.at<cv::Vec2f>(pt).val[1]+pt.y;

                /*
                if (sx <= 0 || sy <= 0 || sx >= SX || sy >= SY )
                {
                    sx = -1.0;
                    //continue;
                }
                vecPTs[j].x = sx;
                vecPTs[j].y = sy;
                */
            }
        }

        cvObjects.Clear();
        int n1 = 0;
        int n2 = 0;
        for(int j = 0; j < vecPTs.size(); j++){
            n1++;
            auto pt = vecPTs[j];
            if(pt.x < 0.0){
                n2++;
                continue;
            }
            cvObjects.push_back(pt);
        }

        std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
        float t = d / 1000.0;
        ss.str("");
        ss<<"ObjectProcessing::End::!!!!!!  "<<cvObjects.size()<<", "<<n1<<" "<<n2<<", "<<t;
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
        WriteLog("Track::Start");
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
		if (bTrack) {
			//local map
			//ofile<<"local map matching = start"<<std::endl;
			EdgeSLAM::LocalMap* pLocal = new EdgeSLAM::LocalMap();
			//ofile<<"local map matching = 1"<<std::endl;
			pMap->GetLocalMap(pLocal);
			//ofile<<"local map matching = 2 = "<<pLocal->mvpMapPoints.size()<<" "<<pLocal->mvpTrackPoints.size()<<std::endl;
			nMatch = 4;
			nMatch = pTracker->TrackWithLocalMap(pCurrFrame, pLocal, 100.0, 50.0);
            //ofile<<"local map matching = "<<nMatch<<"\n";

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
        WriteLog("Track::End");
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
        auto vecPTs = cvObjects.get();
        std::stringstream ss;
        ss<<"VIS = "<<vecPTs.size();
        WriteLog(ss.str());

        for(int j = 0; j < vecPTs.size(); j++){
            auto pt = vecPTs[j];
            if(pt.x < 0.0)
                continue;
            float x = pt.x*scale;
            float y = pt.y*scale;
            cv::circle(img, cv::Point2f(x,y), 3, cv::Scalar(255, 0, 255, 255), -1);
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
}
