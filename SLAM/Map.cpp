#include "Map.h"
#include "LocalMap.h"
#include "RefFrame.h"

namespace EdgeSLAM {
	Map::Map():mpRefFrame(nullptr), mpLocalMap(nullptr){

	}
	Map::~Map() {

	}

	void Map::SetReferenceFrame(RefFrame* pRef){
		std::unique_lock<std::mutex> lock(mMutexRefFrame);
		if(mpRefFrame)
		    delete mpRefFrame;
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
    LocalMap* Map::GetLocalMap(){
        std::unique_lock<std::mutex> lock(mMutexLocalMap);
        return mpLocalMap;
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
}