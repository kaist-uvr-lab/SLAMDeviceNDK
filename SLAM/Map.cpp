#include "Map.h"

namespace EdgeSLAM {
	Map::Map():mpRefFrame(nullptr){

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

	void Map::AddImage(cv::Mat gray, int id){
	    std::unique_lock<std::mutex> lock(mMutexFrame);
	    mapGrayImages.clear();
	    mapGrayImages.insert(std::make_pair(id, gray.clone()));
        //mapGrayImages[id] = gray.clone();
	}
    cv::Mat Map::GetImage(int id){
        std::unique_lock<std::mutex> lock(mMutexFrame);
        return mapGrayImages[id];
    }
}