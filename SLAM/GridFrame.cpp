//
// Created by wiseuiux on 2021-12-29.
//

#include "GridFrame.h"
#include "GridCell.h"

namespace EdgeSLAM{
    GridFrame::GridFrame(){
        mFlow = cv::Mat::zeros(10, 10, CV_32FC2);
        mOccupied = cv::Mat::zeros(10, 10, CV_8UC1);
	}
	GridFrame::GridFrame(int row, int col){
         mFlow = cv::Mat::zeros(row, col, CV_32FC2);
         mOccupied = cv::Mat::zeros(row, col, CV_8UC1);
	}
	GridFrame::~GridFrame(){
        mFlow.release();
        mOccupied.release();
        vecCells.Release();
	}
	/*
    void GridFrame::Init(int row, int col){
        mGrid = std::vector<std::vector<GridCell*>>(row);
        for (int i = 0, iend = mGrid.size(); i < iend; i++)
            mGrid[i] = std::vector<GridCell*>(col, nullptr);

        mFlow = cv::Mat::zeros(row, col, CV_32FC2);
    }

    void GridFrame::Copy(GridFrame* p) {
        for (int i = 0, iend = this->mGrid.size(); i < iend; i++) {
            for (int j = 0, jend = this->mGrid[i].size(); j < jend; j++) {
                auto pCell = p->mGrid[i][j];
                this->mGrid[i][j] = pCell;
                pCell->AddObservation(this, i*iend + j);
            }
        }
        this->mFlow = p->mFlow.clone();
    }
    */
}
