#include "LocalMap.h"
#include "Frame.h"
#include "RefFrame.h"
#include "MapPoint.h"

namespace EdgeSLAM {
	LocalMap::LocalMap() {}
	LocalMap::~LocalMap() {
	    for(int i = 0; i < mvpMapPoints.size(); i++){
	        mvpMapPoints[i] = nullptr;
	    }
	    std::vector<MapPoint*>().swap(mvpMapPoints);
	}
}