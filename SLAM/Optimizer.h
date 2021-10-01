#ifndef UNITYLIBRARY_OPTIMIZER_H
#define UNITYLIBRARY_OPTIMIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"

namespace EdgeSLAM {
	class MapPoint;
	class Frame;

	class Optimizer {
	public:
		int static PoseOptimization(Frame* pFrame);
	};
}
#endif