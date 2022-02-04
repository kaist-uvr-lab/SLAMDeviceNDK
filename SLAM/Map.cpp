#include "Map.h"
#include "MapPoint.h"
#include "LocalMap.h"
#include "RefFrame.h"

namespace EdgeSLAM {
	Map::Map():mpRefFrame(nullptr), mpLocalMap(nullptr){

	}
	Map::~Map() {
        if(mpRefFrame)
            delete mpRefFrame;
        if(mpLocalMap)
            delete mpLocalMap;
        auto mapMPs = mapMapPoints.Get();
        for(auto iter = mapMPs.begin(), iend = mapMPs.end(); iter != iend; iter++)
            delete iter->second;
        mapMapPoints.Release();
	}

	void Map::SetReferenceFrame(RefFrame* pRef){
		std::unique_lock<std::mutex> lock(mMutexRefFrame);
		mpRefFrame = pRef;
	}

	RefFrame* Map::GetReferenceFrame() {
		std::unique_lock<std::mutex> lock(mMutexRefFrame);
		return mpRefFrame;
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

}