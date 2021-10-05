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
	EdgeSLAM::Map* EdgeSLAM::RefFrame::MAP;

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
        EdgeSLAM::RefFrame::MAP = pMap;

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

        cv::Mat gray;
        bool res = true;

        cv::Mat frame = cv::Mat(mnHeight, mnWidth, CV_8UC4, data);
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2RGBA);
        cv::flip(frame, frame,0);
        cv::cvtColor(frame, gray, cv::COLOR_RGBA2GRAY);//COLOR_BGRA2GRAY

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        pPrevFrame = pCurrFrame;
        pCurrFrame = new EdgeSLAM::Frame(gray, pCamera, id);
        pCurrFrame->logfile = strLogFile;

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float t_test1 = du_test1 / 1000.0;
        t1 = t_test1;
        t2 = 0.0;

        return pCurrFrame->N;
    }
    void SetLocalMap(){

        EdgeSLAM::LocalMap* pLocal = new EdgeSLAM::LocalMap();
        ////reference frame
        cv::Mat localdesc = GetDataFromUnity("LocalMap");
        cv::Mat localscales = GetDataFromUnity("LocalMapScales");
        cv::Mat localangles = GetDataFromUnity("LocalMapAngles");
        cv::Mat localpoints = GetDataFromUnity("LocalMapPoints");
        cv::Mat localids = GetDataFromUnity("LocalMapPointIDs");
        cv::Mat tempObs = GetDataFromUnity("LocalMapPointObservation");
        ////reference frame

        //create map points
        int Nmp = localpoints.rows/12;
        int Ndesc = localdesc.rows/32;
        int Nscale = localscales.rows;
        int Nangle = localangles.rows/4;
        int Nid = localids.rows/4;

        cv::Mat tempPoints(Nmp, 3, CV_32FC1, localpoints.data);
        cv::Mat tempDescs(Ndesc, 32, CV_8UC1, localdesc.data);
        cv::Mat tempScales(Nscale, 1, CV_8UC1, localscales.data);
        cv::Mat tempAngles(Nangle, 1, CV_32FC1, localpoints.data);
        cv::Mat tempIDs(Nid, 1, CV_32SC1, localids.data);

        for(int i = 0; i < Nmp; i++)
        {
            int id = tempIDs.at<int>(i,0);
            EdgeSLAM::MapPoint* pMP = nullptr;
            if(pMap->CheckMapPoint(id))
            {
                pMP = pMap->GetMapPoint(id);
            }else{
                pMP = new EdgeSLAM::MapPoint(id, tempPoints.at<float>(i,0), tempPoints.at<float>(i,1), tempPoints.at<float>(i,2));
                pMap->AddMapPoint(id, pMP);
            }
            ////update map point
            int scale = tempScales.at<uchar>(i,0);
            cv::Mat pos = tempPoints.row(i).clone();
            float angle = tempAngles.at<float>(i,0);
            cv::Mat desc = tempDescs.row(i).clone();
            int obs = tempObs.at<uchar>(i,0);
            pMP->Update(pos.t(), desc, angle, scale, obs);

            //create track point
            auto pTP = new EdgeSLAM::TrackPoint();

            pLocal->mvpMapPoints.push_back(pMP);
            pLocal->mvpTrackPoints.push_back(pTP);

        }
        pMap->SetLocalMap(pLocal);

        //create map points

        //auto pLocalMap = new
        /*
        cv::Mat points(len1/12, 3, CV_32FC1, data1);
        cv::Mat descs(len2/32, 32, CV_8UC1, data2);
        cv::Mat scales(len3, 1, CV_8UC1, data3);
        cv::Mat angles(len4/4, 1, CV_32FC1, data4);
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<len1/12<<" "<<len2/32<<" "<<len3<<" "<<len4/4<<std::endl;
        ofile.close();
        */
        /*
        ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<Nmp<<" "<<Ndesc<<" "<<Nscale<<" "<<Nangle<<" "<<Nid<<"="<<localids.rows<<std::endl;
        ofile.close();
        */
    }

    void SetReferenceFrame() {
        ////id는 서버에 보낸 이미지의 id가 기록 됨

        ////reference frame
        cv::Mat f1 = GetDataFromUnity("ReferenceFrame");
        cv::Mat refdesc = GetDataFromUnity("ReferenceFrameDesc");
        cv::Mat desc(refdesc.rows/32,32,CV_8UC1, refdesc.data);
        auto pRefFrame = new EdgeSLAM::RefFrame(pCamera, desc, (float*)f1.data);
        std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pRefFrame->mDescriptors);
		pVoc->transform(vCurrentDesc, pRefFrame->mBowVec, pRefFrame->mFeatVec, 4);

        //pTracker->mnLastKeyFrameId = id;
		pMap->SetReferenceFrame(pRefFrame);

        /*
		float res = -.0;
		ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
        ofile<<pRefFrame->N<<std::endl;
        ofile.close();
        */

        /*
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
		*/
	}


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
				ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
                ofile<<"ref matching = "<<nMatch<<" "<<rf->mDescriptors.rows<<" "<<rf->mvpMapPoints.size()<<" "<<rf->N<<"\n";
                ofile.close();
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
					std::vector<cv::Mat> vCurrentDesc = Utils::toDescriptorVector(pCurrFrame->mDescriptors);
					pVoc->transform(vCurrentDesc, pCurrFrame->mBowVec, pCurrFrame->mFeatVec, 4);

					nMatch = pTracker->TrackWithReferenceFrame(rf, pCurrFrame);
					bTrack = nMatch >= 10;
				}
			}
		}
		if (bTrack) {
			//local map
			auto pLocal = pMap->GetLocalMap();
			nMatch = 4;
			nMatch = pTracker->TrackWithLocalMap(pCurrFrame, pLocal, 100.0, 50.0);

            ofile.open(strLogFile.c_str(), std::ios_base::out | std::ios_base::app);
            ofile<<"local map matching = "<<nMatch<<"\n";
            ofile.close();

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

}
