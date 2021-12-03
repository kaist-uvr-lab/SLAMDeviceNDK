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

    //std::map<int, cv::Mat> mapObjectInfos;
    cv::Mat ObjectInfo = cv::Mat::zeros(0,6,CV_32FC1);
    std::mutex mMutexObjectInfo;

    std::string strPath;
	int mnWidth, mnHeight;
	int mnSkipFrame;
	int mnFeature, mnLevel;
	float mfScale;

    std::ifstream inFile;
    char x[1000];

    std::ofstream ofile;
    std::string strLogFile;
    void SetPath(char* path) {
        strPath = std::string(path);
        std::stringstream ss;
        ss << strPath << "/debug.txt";
        strLogFile = strPath+"/debug.txt";

    }

    std::map<std::string, cv::Mat> UnityData;
    std::mutex mMutexUnityData;
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
    void SetInit(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, int nfeature, int nlevel, float fscale, int nSkip) {//char* vocName,

        ofile.open(strLogFile.c_str(), std::ios::trunc);
        ofile<<"start\n";
        ofile.close();

		mnWidth  = _w;
		mnHeight = _h;
        mnSkipFrame = nSkip;

        mnFeature = nfeature;
        mnLevel = nlevel;
        mfScale = fscale;

        pDetector = new EdgeSLAM::ORBDetector(nfeature,fscale,nlevel);
        pCamera = new EdgeSLAM::Camera(_w, _h, _fx, _fy, _cx, _cy, _d1, _d2, _d3, _d4);
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
    }

    void* imuAddr;
    void* accAddr;
    bool bIMU = false;
	void SetIMUAddress(void* addr, bool bimu){
        imuAddr = addr;
        bIMU = bimu;
	}

    int SetFrame(void* data, int id, double ts, float& t1, float& t2) {
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<"Frame start"<<std::endl;
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        cv::Mat gray;
        bool res = true;
        cv::Mat frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGBA);
        cv::flip(frame, frame,0);
        cv::cvtColor(frame, gray, cv::COLOR_RGBA2GRAY);//COLOR_BGRA2GRAY

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

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        auto du_total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float t_total = du_total / 1000.0;
        t1 = t_total;
        t2 = 0.0;
        ofile<<"SetFrame="<<t_total<<std::endl;
        ofile.close();
        return pCurrFrame->N;
    }
    void SetLocalMap(){

    }

    void SetReferenceFrame(int id) {
        ////reference frame
        //ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        //ofile<<"ReferenceFrame=start"<<std::endl;
        //std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
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
        for(ref = pRefFrame; ref; ref = ref->mpParent, nkf++){
            if(!ref || nkf >= 5){
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
        }

        ////delete ref frame
        while(ref){
            auto kf = ref;
            ref = ref->mpParent;
            kf->mpParent = nullptr;
            auto vpMPs = kf->mvpMapPoints;
            for(int i =0; i < kf->N; i++){
                auto pMP = vpMPs[i];
                if(!pMP || pMP->isBad()){
                    continue;
                }
                pMP->EraseObservation(kf);
            }

            if(!ref)
                break;
        }

        pMap->SetReferenceFrame(pRefFrame);
        pMap->SetLocalMap(pLocal);

        //std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        //auto du_total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        //float t_total = du_total / 1000.0;
        //ofile<<"SetReference="<<t_total<<std::endl;
        //ofile.close();
	}
    void AddObjectInfos(){
        std::unique_lock<std::mutex> lock(mMutexObjectInfo);
        cv::Mat tdata = GetDataFromUnity("ObjectDetection");
        int n = tdata.rows/24;
        ObjectInfo = cv::Mat(n, 6, CV_32FC1, tdata.data);
        //float* data = (float*)tdata.data;
        //ObjectInfo =
    }

	void AddContentInfo(int id, float x, float y, float z) {
		cv::Mat pos = (cv::Mat_<float>(3, 1) <<x, y, z);
		std::unique_lock<std::mutex> lock(mMutexContentInfo);
		mapContentInfos.insert(std::make_pair(id, pos));
	}

	bool Track(void* data) {
        //ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        //ofile<<"tracking start"<<std::endl;
		bool bTrack = false;
		int nMatch = -1;

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
		}

		cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
		cv::Mat t = cv::Mat::zeros(3, 1, CV_32FC1);
		if (bTrack) {
			pTracker->mTrackState = EdgeSLAM::TrackingState::Success;
			pMotionModel->update(pCurrFrame->GetPose());
			cv::Mat T = pCurrFrame->GetPose();
			R = T.rowRange(0, 3).colRange(0, 3).clone();
			t = T.col(3).rowRange(0, 3).clone();
		}
		else {
			pTracker->mTrackState = EdgeSLAM::TrackingState::Failed;
			pMotionModel->reset();
		}
		R.push_back(t.t());
		memcpy(data, R.data, sizeof(float) * 12);

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        auto du_total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float t_total = du_total / 1000.0;
        //ofile<<"Tracking="<<t_total<<std::endl;
        //ofile.close();
		return bTrack;
	}

    void WriteLog(char* data){
        std::string log(data);
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<log<<"\n";
        ofile.close();
    }

    bool VisualizeFrame(void* data){
        if (!pCurrFrame)
			return false;
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<"visualize start"<<std::endl;
        cv::Mat img = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        cv::cvtColor(img, img, cv::COLOR_BGRA2RGBA);
        cv::flip(img, img,0);

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
        bool res = false;
		if (pTracker->mTrackState == EdgeSLAM::TrackingState::Success) {

			for (int i = 0; i < pCurrFrame->N; i++) {
				auto pt = pCurrFrame->mvKeys[i].pt;
                pt.y = mnHeight -pt.y;
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
		//cv::flip(img, img, 0);

		cv::cvtColor(img, img, cv::COLOR_RGBA2BGRA);
		memcpy(data, img.data, sizeof(cv::Vec4b) * img.rows * img.cols);

        /*
        std::stringstream ss;
        ss<<strPath<<"/color.jpg";
        cv::imwrite(ss.str(), img);
        */
        ofile<<"visualize end"<<std::endl;
        ofile.close();
		return true;
    }
}
