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
	EdgeSLAM::ORBDetector* EdgeSLAM::Frame::detector;
	EdgeSLAM::ORBDetector* EdgeSLAM::RefFrame::detector;

	std::string EdgeSLAM::LocalMap::logFile;

    std::map<int, cv::Mat> mapContentInfos;
    std::mutex mMutexContentInfo;

    std::string strPath;
	int mnWidth, mnHeight;

    std::ifstream inFile;
    char x[1000];

    std::ofstream ofile;
    std::string strLogFile;
    void SetPath(char* path) {
        strPath = std::string(path);
        std::stringstream ss;
        ss << strPath << "/debug.txt";
        strLogFile = strPath+"/debug.txt";

        //freopen(strLogFile.c_str(), "w", stdout);
        //ofile.open(strLogFile.c_str(), std::ios::trunc);
        //ofile<<"start\n";
        //ofile.close();
    }
    void LoadVocabulary(){
        std::string strVoc = strPath+"/orbvoc.dbow3";//std::string(vocName);
        pVoc = new DBoW3::Vocabulary();
        pVoc->load(strVoc);
    }
    void SetInit(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, int nfeature, int nlevel, float fscale) {//char* vocName,

		mnWidth  = _w;
		mnHeight = _h;


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
        EdgeSLAM::RefFrame::detector = pDetector;
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
        EdgeSLAM::Frame::detector = pDetector;
        EdgeSLAM::RefFrame::detector = pDetector;
        EdgeSLAM::MapPoint::Detector = pDetector;

        EdgeSLAM::LocalMap::logFile = strLogFile;
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


    bool GrabImage(void* data, int id){
        cv::Mat gray;
        bool res = true;

        cv::Mat frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGBA);
        cv::flip(frame, frame,0);
        cv::cvtColor(frame, gray, cv::COLOR_RGBA2GRAY);//COLOR_BGRA2GRAY

        //pMap->AddImage(gray, id);
/*
        {
            std::unique_lock < std::mutex> lock(pMap->mMutexFrame);
            pMap->mapGrayImages[id] = gray.clone();
        }
*/

/*
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<data==frame.data<<std::endl;
        ofile.close();
*/

/*
        if (bDeviceCam) {
            cv::flip(gray, gray, 0);
            res = true;
        }else{

            res = true;
        }
*/

/*
        std::stringstream ss;
        ss<<strPath<<"/img/image_"<<id<<".jpg";
        imwrite(ss.str(), frame);
        ss.str("");
        ss<<strPath<<"/img/image_gray_"<<id<<".jpg";
        imwrite(ss.str(), gray);
*/
        return res;
    }

    int SetFrame(void* data, int id, double ts, float& t1, float& t2) {

        cv::Mat gray;
        bool res = true;

        cv::Mat frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGBA);
        cv::flip(frame, frame,0);
        cv::cvtColor(frame, gray, cv::COLOR_RGBA2GRAY);//COLOR_BGRA2GRAY

        /*
        cv::Mat gray = pMap->GetImage(id);
        {
            std::unique_lock < std::mutex> lock(pMap->mMutexFrame);
            gray = pMap->mapGrayImages[id].clone();
        }
        */

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        pPrevFrame = pCurrFrame;
        pCurrFrame = new EdgeSLAM::Frame(gray, pCamera, id);

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        //{
        //    std::unique_lock < std::mutex> lock(pMap->mMutexFrames);
        //    pMap->mmpFrames[id] = pCurrFrame;
        //}

        auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float t_test1 = du_test1 / 1000.0;
        t1 = t_test1;
        t2 = 0.0;

        return pCurrFrame->N;
    }
    void SetLocalMap(void* data1, int len1, void* data2, int len2, void* data3, int len3, void* data4, int len4){
        cv::Mat points(len1/12, 3, CV_32FC1, data1);
        cv::Mat descs(len2/32, 32, CV_8UC1, data2);
        cv::Mat scales(len3, 1, CV_8UC1, data3);
        cv::Mat angles(len4/4, 1, CV_32FC1, data4);
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<len1/12<<" "<<len2/32<<" "<<len3<<" "<,len4/4<<std::endl;
        ofile.close();
    }
    void SetReferenceFrame(int id, float* data) {

		float res = -1.0;
		cv::Mat tempGray = pMap->GetImage(id);;
		/*
		{
			std::unique_lock < std::mutex> lock(pMap->mMutexFrame);
			tempGray = pMap->mapGrayImages[id].clone();
		}
        */
		auto pRefFrame = new EdgeSLAM::RefFrame(pMap, tempGray, pCamera, data);

		std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pRefFrame->mDescriptors);
		pVoc->transform(vCurrentDesc, pRefFrame->mBowVec, pRefFrame->mFeatVec, 4);

		//int nKFs = pMap->mmpRefFrames.size();
		//if (nKFs == 0 ||pTracker->NeedNewKeyFrame(ref, pRefFrame, nKFs, pRefFrame->N)) {
		pMap->mmpRefFrames[pRefFrame->mnId] = pRefFrame;
		pRefFrame->UpdateMapPoints();
		pRefFrame->UpdateConnections();
		pTracker->mnLastKeyFrameId = id;
		pMap->SetReferenceFrame(pRefFrame);
		//update map points
		//udpate connection
		//addkeyframe in map
		//}
	}

	void ReleaseImage() {
		//frame.release();
	}
	/*
    void* GetResultAddr(bool& bres){

        if (!pCurrFrame){
            bres = false;
            return nullptr;
        }
        cv::Mat img = frame.clone();
        if (pTracker->mTrackState == EdgeSLAM::TrackingState::Success) {

            for (int i = 0; i < pCurrFrame->N; i++) {
                if (!pCurrFrame->mvpMapPoints[i] || pCurrFrame->mvbOutliers[i])
                    continue;
                auto pt = pCurrFrame->mvKeys[i].pt;
                pt.y = mnHeight -pt.y;
                cv::circle(img, pt, 2, cv::Scalar(255, 255, 0, 255), -1);
            }

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
        }
        bres = true;
        return img.data;
    }
    */
    /*
	bool GetMatchingImage(void* data) {

		if (!pCurrFrame)
			return false;
        cv::Mat img = frame.clone();
		if (pTracker->mTrackState == EdgeSLAM::TrackingState::Success) {

			for (int i = 0; i < pCurrFrame->N; i++) {
				if (!pCurrFrame->mvpMapPoints[i] || pCurrFrame->mvbOutliers[i])
					continue;
				auto pt = pCurrFrame->mvKeys[i].pt;
				pt.y = mnHeight -pt.y;
				cv::circle(img, pt, 2, cv::Scalar(255, 255, 0, 255), -1);
			}

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
		}
		cv::flip(img, img, 0);
		memcpy(data, img.data, sizeof(cv::Vec4b) * img.rows * img.cols);
		return true;

	}
	*/
	void AddContentInfo(int id, float x, float y, float z) {
		cv::Mat pos = (cv::Mat_<float>(3, 1) <<x, y, z);
		std::unique_lock<std::mutex> lock(mMutexContentInfo);
		mapContentInfos.insert(std::make_pair(id, pos));
	}

	bool Track(void* data) {

		bool bTrack = false;
		int nMatch = -1;

		if (pTracker->mTrackState == EdgeSLAM::TrackingState::NotEstimated || pTracker->mTrackState == EdgeSLAM::TrackingState::Failed) {
			EdgeSLAM::RefFrame* rf = pMap->GetReferenceFrame();

			if (rf) {
				std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pCurrFrame->mDescriptors);
				pVoc->transform(vCurrentDesc, pCurrFrame->mBowVec, pCurrFrame->mFeatVec, 4);

				nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame);
				bTrack = nMatch >= 10;

				if (bTrack) {
					pTracker->mnLastRelocFrameId = pCurrFrame->mnFrameID;
				}
			}
		}
		if (pTracker->mTrackState == EdgeSLAM::TrackingState::Success) {
			//pCurrFrame->check_replaced_map_points();
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
					std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pCurrFrame->mDescriptors);
					pVoc->transform(vCurrentDesc, pCurrFrame->mBowVec, pCurrFrame->mFeatVec, 4);

					nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame);
					bTrack = nMatch >= 10;
				}
			}
		}
		if (bTrack) {
			//local map
			nMatch = 4;
			nMatch = pTracker->TrackWithLocalMap(pCurrFrame, 100.0, 50.0);

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
		//printf("Track::end=%d : %d, %d\n", pCurrFrame->mnFrameID, nMatch, bTrack);

		return bTrack;
	}

    void WriteLog(char* data){
        std::string log(data);
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<log<<"\n";
        ofile.close();
    }

/*
	int SetFrameByPtr(void* addr, int w, int h, int id) {

		cv::Mat img = cv::Mat(h, w, CV_8UC4, addr);


		pPrevFrame = pCurrFrame;
		pCurrFrame = new EdgeSLAM::Frame(img, pCamera, id);
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

		{
			std::unique_lock < std::mutex> lock(pMap->mMutexFrames);
			pMap->mmpFrames[id] = pCurrFrame;
		}
		return pCurrFrame->N;
	}

	int SetFrameByFile(char* name, int id, double ts, float& t1, float& t2) {
		freopen("debug.txt", "a", stdout);
		printf("SetFrame::Start=%d\n",id);
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		std::string strfile(name);
		cv::Mat img = cv::imread(strfile);

		pPrevFrame = pCurrFrame;
		pCurrFrame = new EdgeSLAM::Frame(img, pCamera, id);
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

		{
			std::unique_lock < std::mutex> lock(pMap->mMutexFrames);
			pMap->mmpFrames[id] = pCurrFrame;
		}

		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		t1 = t_test1;
		t2 = 0.0;
		printf("SetFrame::End\n");
		return pCurrFrame->N;
	}


	int SetFrameByImage(unsigned char* raw, int len, int id, double ts, float& t1, float& t2) {
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		cv::Mat temp = cv::Mat(len, 1, CV_8UC1, raw);
		//cv::Mat temp = cv::Mat::zeros(len, 1, CV_8UC1, (void*)raw);
		//std::memcpy(temp.data, raw, len);
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);
		std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();

		pPrevFrame = pCurrFrame;
		pCurrFrame = new EdgeSLAM::Frame(img, pCamera, id);

		{
			std::unique_lock < std::mutex> lock(pMap->mMutexFrames);
			pMap->mmpFrames[id] = pCurrFrame;
		}

		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count();
		float t_test2 = du_test2 / 1000.0;

		t1 = t_test1;
		t2 = t_test2;

		return pCurrFrame->N;
	}
	 */







}
