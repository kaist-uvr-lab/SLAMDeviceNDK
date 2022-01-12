//
// Created by wiseuiux on 2021-12-29.
//

#include "GridFrame.h"
#include "GridCell.h"

namespace EdgeSLAM{
    GridFrame::GridFrame(){
        Init(10,10);
	}
	GridFrame::GridFrame(int r, int c){
        Init(r,c);
	}
	GridFrame::~GridFrame(){
		for (int i = 0, iend = mGrid.size(); i < iend; i++) {
			for (int j = 0, jend = mGrid[i].size(); j < jend; j++) {
				auto pGC = mGrid[i][j];
				if (!pGC)
					continue;
				//pGC->EraseObservation(this);
				//if (pGC->isBad())
				//	delete pGC;
			}
			std::vector<GridCell*>().swap(mGrid[i]);
		}
		std::vector<std::vector<GridCell*>>().swap(mGrid);
	}
    void GridFrame::Init(int row, int col){
        mGrid = std::vector<std::vector<GridCell*>>(row);
        for (int i = 0, iend = mGrid.size(); i < iend; i++)
            mGrid[i] = std::vector<GridCell*>(col, nullptr);
    }
    void GridFrame::Copy(GridFrame* p) {
        for (int i = 0, iend = this->mGrid.size(); i < iend; i++) {
            for (int j = 0, jend = this->mGrid[i].size(); j < jend; j++) {
                auto pCell = p->mGrid[i][j];
                this->mGrid[i][j] = pCell;
                pCell->AddObservation(this, i*iend + j);
            }
        }
    }

}
