#include "LocalMap.h"
#include "Frame.h"
#include "RefFrame.h"
#include "MapPoint.h"

namespace EdgeSLAM {
	LocalMap::LocalMap() {}
	LocalMap::~LocalMap() {
	    std::vector<MapPoint*>().swap(mvpMapPoints);
	}
}