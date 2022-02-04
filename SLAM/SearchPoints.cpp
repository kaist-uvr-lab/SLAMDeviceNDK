#include "SearchPoints.h"
#include "Frame.h"
#include "MapPoint.h"
#include "RefFrame.h"
#include "ORBDetector.h"

namespace EdgeSLAM {
	
	const int SearchPoints::HISTO_LENGTH = 30;

    int SearchPoints::SearchFrameByProjection(RefFrame* ref, Frame* curr, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri){

        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        std::vector<int> rotHist[HISTO_LENGTH];

        const float factor = 1.0f / HISTO_LENGTH;
        cv::Mat Tcw = curr->GetPose();
        const cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = Tcw.rowRange(0, 3).col(3);

        for (int i = 0; i<ref->N; i++)
		{
			MapPoint* pMP = ref->mvpMapPoints[i];

			if (pMP && !pMP->isBad())
			{
				// Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw + tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0 / x3Dc.at<float>(2);

                if (invzc<0)
                    continue;

                float u = curr->fx*xc*invzc + curr->cx;
                float v = curr->fy*yc*invzc + curr->cy;

                if (u<curr->mnMinX || u>curr->mnMaxX)
                    continue;
                if (v<curr->mnMinY || v>curr->mnMaxY)
                    continue;

                int nLastOctave = ref->mvKeys[i].octave;

                // Search in a window. Size depends on scale
                float radius = thProjection*curr->mvScaleFactors[nLastOctave];

                std::vector<size_t> vIndices2;
                vIndices2 = curr->GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);
                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                float bestDist = 256;
                int bestIdx2 = -1;

                for (std::vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                {
                    const size_t i2 = *vit;
                    auto pMP2 = curr->mvpMapPoints[i2];
                    if (pMP2 && !pMP2->isBad() && pMP2->Observations() > 0)
                        continue;

                    const cv::Mat &d = curr->mDescriptors.row(i2);

                    const float dist = Detector->CalculateDescDistance(dMP, d);

                    if (dist<bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= thMaxDesc)
                {
                    curr->mvpMapPoints[bestIdx2] = pMP;
                    nmatches++;

                    if (bCheckOri)
                    {
                        float rot = ref->mvKeysUn[i].angle - curr->mvKeysUn[bestIdx2].angle;
                        if (rot<0.0)
                            rot += 360.0f;
                        int bin = round(rot*factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i<HISTO_LENGTH; i++)
			{
				if (i != ind1 && i != ind2 && i != ind3)
				{
					for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
					{
						curr->mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
						nmatches--;
					}
				}
			}
		}
		return nmatches;

    }

	int SearchPoints::SearchFrameByProjection(Frame* prev, Frame* curr, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri) {

        //std::ofstream ofile;
        //ofile.open(LogFile.c_str(), std::ios_base::out | std::ios_base::app);
        //ofile<<"SearchPoints::SearchFrameByProjection::a"<<std::endl;

		int nmatches = 0;

		// Rotation Histogram (to check rotation consistency)
		std::vector<int> rotHist[HISTO_LENGTH];
		/*
		for (int i = 0; i<HISTO_LENGTH; i++)
			rotHist[i].reserve(500);
		*/
		const float factor = 1.0f / HISTO_LENGTH;
		cv::Mat Tcw = curr->GetPose();
		//ofile<<"SearchPoints::SearchFrameByProjection::111"<<std::endl;
		cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
		//ofile<<"SearchPoints::SearchFrameByProjection::222"<<std::endl;
		cv::Mat tcw = Tcw.rowRange(0, 3).col(3);

        /*
        ofile<<"SearchPoints::SearchFrameByProjection::b"<<std::endl;
        std::stringstream ss;
        ss<<Rcw<<std::endl;
        ofile<<ss.str();
        ss.str("");
        ss<<tcw<<std::endl;
        ofile<<ss.str();
        ss.str("");
        ss<<Tcw.t()<<std::endl;
        ofile<<ss.str();
        */
		for (int i = 0; i<prev->N; i++)
		{
			MapPoint* pMP = prev->mvpMapPoints[i];

			if (pMP && !pMP->isBad())
			{
				if (!prev->mvbOutliers[i])
				{
					// Project
					cv::Mat x3Dw = pMP->GetWorldPos();
					cv::Mat x3Dc = Rcw*x3Dw + tcw;

					const float xc = x3Dc.at<float>(0);
					const float yc = x3Dc.at<float>(1);
					const float invzc = 1.0 / x3Dc.at<float>(2);

					if (invzc<0)
						continue;

					float u = curr->fx*xc*invzc + curr->cx;
					float v = curr->fy*yc*invzc + curr->cy;

					if (u<curr->mnMinX || u>curr->mnMaxX)
						continue;
					if (v<curr->mnMinY || v>curr->mnMaxY)
						continue;

					int nLastOctave = prev->mvKeys[i].octave;

					// Search in a window. Size depends on scale
					float radius = thProjection*curr->mvScaleFactors[nLastOctave];

					std::vector<size_t> vIndices2;
					vIndices2 = curr->GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);
					if (vIndices2.empty())
						continue;

					const cv::Mat dMP = pMP->GetDescriptor();

					float bestDist = 256;
					int bestIdx2 = -1;

					for (std::vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
					{
						const size_t i2 = *vit;
						auto pMP2 = curr->mvpMapPoints[i2];
						if (pMP2 && !pMP2->isBad() && pMP2->Observations()>0)
                            continue;

						const cv::Mat &d = curr->mDescriptors.row(i2);

						const float dist = Detector->CalculateDescDistance(dMP, d);

						if (dist<bestDist)
						{
							bestDist = dist;
							bestIdx2 = i2;
						}
					}

					if (bestDist <= thMaxDesc)
					{
						curr->mvpMapPoints[bestIdx2] = pMP;
						nmatches++;

						if (bCheckOri)
						{
							float rot = prev->mvKeysUn[i].angle - curr->mvKeysUn[bestIdx2].angle;
							if (rot<0.0)
								rot += 360.0f;
							int bin = round(rot*factor);
							if (bin == HISTO_LENGTH)
								bin = 0;
							assert(bin >= 0 && bin<HISTO_LENGTH);
							rotHist[bin].push_back(bestIdx2);
						}
					}
				}
			}
		}
        //ofile<<"SearchPoints::SearchFrameByProjection::c"<<std::endl;
		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i<HISTO_LENGTH; i++)
			{
				if (i != ind1 && i != ind2 && i != ind3)
				{
					for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
					{
						curr->mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
						nmatches--;
					}
				}
			}
		}
		//ofile<<"SearchPoints::SearchFrameByProjection::d"<<std::endl;
		//ofile.close();
		return nmatches;
	}

	int SearchPoints::SearchMapByProjection(Frame *F, const std::vector<MapPoint*> &vpMapPoints, float thMaxDesc, float thMinDesc, float thRadius, float thMatchRatio, bool bCheckOri)
	{

	    //std::ofstream ofile;
        //ofile.open(LogFile.c_str(), std::ios_base::out | std::ios_base::app);
        //ofile<<"SearchPoints::SearchMapByProjection::a"<<std::endl;

		int nmatches = 0;
		const bool bFactor = thRadius != 1.0;

		for (size_t iMP = 0; iMP<vpMapPoints.size(); iMP++)
		{
			MapPoint* pMP = vpMapPoints[iMP];
            if(!pMP || pMP->isBad())
                continue;
			if (!pMP->mbTrackInView) {
				continue;
			}
			const int &nPredictedLevel = pMP->mnTrackScaleLevel;

			// The size of the window will depend on the viewing direction
			float r = RadiusByViewingCos(pMP->mTrackViewCos);

			if (bFactor)
				r *= thRadius;

			const std::vector<size_t> vIndices =
				F->GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r*F->mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

			if (vIndices.empty()) {
				continue;
			}
			const cv::Mat MPdescriptor = pMP->GetDescriptor();

			float bestDist = 256;
			int bestLevel = -1;
			float bestDist2 = 256;
			int bestLevel2 = -1;
			int bestIdx = -1;

			// Get best and second matches with near keypoints
			for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
			{
				const size_t idx = *vit;
                auto pMP = F->mvpMapPoints[idx];
				if (pMP && !pMP->isBad() && pMP->Observations()>0) {
					continue;
				}
				const cv::Mat &d = F->mDescriptors.row(idx);

				const float dist = Detector->CalculateDescDistance(MPdescriptor, d);

				if (dist<bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestLevel2 = bestLevel;
					bestLevel = F->mvKeysUn[idx].octave;
					bestIdx = idx;
				}
				else if (dist<bestDist2)
				{
					bestLevel2 = F->mvKeysUn[idx].octave;
					bestDist2 = dist;
				}
			}
			// Apply ratio to second match (only if best and second are in the same scale level)
			if (bestDist <= thMaxDesc)
			{
				if (bestLevel == bestLevel2 && bestDist > thMatchRatio*bestDist2) {
					continue;
				}
				F->mvpMapPoints[bestIdx] = pMP;
				nmatches++;
			}
		}
        //ofile<<"SearchPoints::SearchMapByProjection::b"<<std::endl;
        //ofile.close();
		return nmatches;
	}

	void SearchPoints::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
	{
		int max1 = 0;
		int max2 = 0;
		int max3 = 0;

		for (int i = 0; i<L; i++)
		{
			const int s = histo[i].size();
			if (s>max1)
			{
				max3 = max2;
				max2 = max1;
				max1 = s;
				ind3 = ind2;
				ind2 = ind1;
				ind1 = i;
			}
			else if (s>max2)
			{
				max3 = max2;
				max2 = s;
				ind3 = ind2;
				ind2 = i;
			}
			else if (s>max3)
			{
				max3 = s;
				ind3 = i;
			}
		}

		if (max2<0.1f*(float)max1)
		{
			ind2 = -1;
			ind3 = -1;
		}
		else if (max3<0.1f*(float)max1)
		{
			ind3 = -1;
		}
	}
	float SearchPoints::RadiusByViewingCos(const float &viewCos)
	{
		if (viewCos>0.998)
			return 2.5;
		else
			return 4.0;
	}
}

