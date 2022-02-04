//
// Created by wiseuiux on 2021-12-29.
//

#ifndef EDGESLAMNDK_GRIDFRAME_H
#define EDGESLAMNDK_GRIDFRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>
#include "ConcurrentVector.h"

namespace EdgeSLAM {
    class GridCell;
    class GridFrame {
    public:
            GridFrame();
            GridFrame(int r, int c);
            virtual ~GridFrame();
        public:
            //void Init(int row, int col);
            //void Copy(GridFrame* p);
            //std::vector<std::vector<GridCell*>> mGrid;
            cv::Mat mFlow, mOccupied;
            ConcurrentVector<GridCell> vecCells;
    };
}


#endif //EDGESLAMNDK_GRIDFRAME_H
