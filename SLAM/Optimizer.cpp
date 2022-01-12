#include "Optimizer.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Utils.h"

namespace EdgeSLAM {
	int Optimizer::PoseOptimization(Frame *pFrame)
	{
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

		linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		int nInitialCorrespondences = 0;

		// Set Frame vertex
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Utils::toSE3Quat(pFrame->GetPose()));
		vSE3->setId(0);
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);

		// Set MapPoint vertices
		const int N = pFrame->N;

		std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
		std::vector<size_t> vnIndexEdgeMono;
		vpEdgesMono.reserve(N);
		vnIndexEdgeMono.reserve(N);

		const float deltaMono = sqrt(5.991);


		{
			//std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

			for (int i = 0; i<N; i++)
			{
				MapPoint* pMP = pFrame->mvpMapPoints[i];
				if (pMP && !pMP->isBad())
				{
					nInitialCorrespondences++;
					pFrame->mvbOutliers[i] = false;

					Eigen::Matrix<double, 2, 1> obs;
					const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
					obs << kpUn.pt.x, kpUn.pt.y;

					g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
					e->setMeasurement(obs);
					const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(deltaMono);

					e->fx = pFrame->fx;
					e->fy = pFrame->fy;
					e->cx = pFrame->cx;
					e->cy = pFrame->cy;
					cv::Mat Xw = pMP->GetWorldPos();
					e->Xw[0] = Xw.at<float>(0);
					e->Xw[1] = Xw.at<float>(1);
					e->Xw[2] = Xw.at<float>(2);

					optimizer.addEdge(e);

					vpEdgesMono.push_back(e);
					vnIndexEdgeMono.push_back(i);
				}

			}
		}


		if (nInitialCorrespondences<3)
			return 0;

		// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
		// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
		const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
		const int its[4] = { 10,10,10,10 };

		int nBad = 0;
		for (size_t it = 0; it<4; it++)
		{

			vSE3->setEstimate(Utils::toSE3Quat(pFrame->GetPose()));
			optimizer.initializeOptimization(0);
			optimizer.optimize(its[it]);

			nBad = 0;
			for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
			{
				g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

				const size_t idx = vnIndexEdgeMono[i];

				if (pFrame->mvbOutliers[idx])
				{
					e->computeError();
				}

				const float chi2 = e->chi2();

				if (chi2>chi2Mono[it])
				{
					pFrame->mvbOutliers[idx] = true;
					e->setLevel(1);
					nBad++;
				}
				else
				{
					pFrame->mvbOutliers[idx] = false;
					e->setLevel(0);
				}
				
				if (it == 2)
					e->setRobustKernel(0);
			}

			if (optimizer.edges().size()<10)
				break;
		}

		// Recover optimized pose and return number of inliers
		g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
		g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		cv::Mat pose = Utils::toCvMat(SE3quat_recov);
		pFrame->SetPose(pose);

		return nInitialCorrespondences - nBad;
	}
}