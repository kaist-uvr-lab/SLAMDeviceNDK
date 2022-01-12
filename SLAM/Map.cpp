#include "Map.h"
#include "MapPoint.h"
#include "LocalMap.h"
#include "RefFrame.h"

namespace EdgeSLAM {
	Map::Map():mpRefFrame(nullptr), mpLocalMap(nullptr){

	}
	Map::~Map() {

	}

	void Map::SetReferenceFrame(RefFrame* pRef){
		std::unique_lock<std::mutex> lock(mMutexRefFrame);
		mpRefFrame = pRef;
	}

	RefFrame* Map::GetReferenceFrame() {
		std::unique_lock<std::mutex> lock(mMutexRefFrame);
		return mpRefFrame;
	}

	void Map::AddImage(cv::Mat img, int id){
        std::unique_lock<std::mutex> lock(mMutexImages);
        mapImages[id] = img;
	}
    cv::Mat Map::GetImage(int id){
        std::unique_lock<std::mutex> lock(mMutexImages);
        cv::Mat res = mapImages[id].clone();
        mapImages.erase(id);
        return res;
    }

    void Map::SetLocalMap(LocalMap* pLocal){
        std::unique_lock<std::mutex> lock(mMutexLocalMap);
        if(mpLocalMap)
            delete mpLocalMap;
        mpLocalMap = pLocal;
    }
    void Map::GetLocalMap(LocalMap* pLocal){
        std::unique_lock<std::mutex> lock(mMutexLocalMap);
        pLocal->mvpMapPoints.reserve(mpLocalMap->mvpMapPoints.size());
        for(int i = 0; i < mpLocalMap->mvpMapPoints.size(); i++){
            auto pMPi = mpLocalMap->mvpMapPoints[i];
            if(pMPi && !pMPi->isBad())
                pLocal->mvpMapPoints.push_back(pMPi);
        }
        pLocal->mvpMapPoints  = std::vector<MapPoint*>  (mpLocalMap->mvpMapPoints.begin(),   mpLocalMap->mvpMapPoints.end());
    }
    bool Map::CheckMapPoint(int id){
        std::unique_lock<std::mutex> lock(mMutexMapPoints);
        return mmpMapPoints.count(id)>0;
    }
    void Map::AddMapPoint(int id, MapPoint* pMP){
        std::unique_lock<std::mutex> lock(mMutexMapPoints);
        mmpMapPoints[id]=pMP;
    }
    MapPoint* Map::GetMapPoint(int id){
        std::unique_lock<std::mutex> lock(mMutexMapPoints);
        return mmpMapPoints[id];
    }
    void Map::RemoveMapPoint(int id){
        std::unique_lock<std::mutex> lock(mMutexMapPoints);
        if (mmpMapPoints.count(id))
            mmpMapPoints.erase(id);
    }
}