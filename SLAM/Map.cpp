#include "Map.h"
#include "MapPoint.h"
#include "RefFrame.h"

namespace EdgeSLAM {
	Map::Map():mpRefFrame(nullptr){

	}
	Map::~Map() {
        if(mpRefFrame)
            delete mpRefFrame;
        auto mapMPs = MapPoints.Get();
        for(auto iter = mapMPs.begin(), iend = mapMPs.end(); iter != iend; iter++)
            delete iter->second;
        MapPoints.Release();
	}

	void Map::SetReferenceFrame(RefFrame* pRef){
		std::unique_lock<std::mutex> lock(mMutexRefFrame);
		mpRefFrame = pRef;
	}

	RefFrame* Map::GetReferenceFrame() {
		std::unique_lock<std::mutex> lock(mMutexRefFrame);
		return mpRefFrame;
	}

}