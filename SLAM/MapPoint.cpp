#include "MapPoint.h"
#include "RefFrame.h"
#include "Frame.h"
#include "ORBDetector.h"
#include "Map.h"
#include "ConcurrentMap.h"

namespace EdgeSLAM {

	MapPoint::MapPoint() {}
	MapPoint::MapPoint(int id, float _x, float _y, float _z): mnID(id), mpRefKF(nullptr), mbBad(false), mfMinDistance(0), mfMaxDistance(0), nObs(0){
		mWorldPos = cv::Mat::zeros(3, 1, CV_32FC1);
		mWorldPos.at<float>(0) = _x;
		mWorldPos.at<float>(1) = _y;
		mWorldPos.at<float>(2) = _z;
	}
	MapPoint::~MapPoint(){
	    std::map<RefFrame*, size_t>().swap(mObservations);

	}

    cv::Mat MapPoint::GetWorldPos()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return mWorldPos.clone();
	}

    cv::Mat MapPoint::GetDescriptor()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return mDescriptor.clone();
	}

    void MapPoint::UpdateNormalAndDepth()
    {
        std::map<RefFrame*, size_t> observations;
        RefFrame* pRefKF;
        cv::Mat Pos;
        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);

            observations = mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos.clone();
        }

        if (observations.empty())
            return;

        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
        int n = 0;
        for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            RefFrame* pKF = mit->first;
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali / cv::norm(normali);
            n++;
        }

        cv::Mat PC = Pos - pRefKF->GetCameraCenter();
        const float dist = cv::norm(PC);
        const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels;

        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            mfMaxDistance = dist*levelScaleFactor;
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
            mNormalVector = normal / n;
        }
    }

    float MapPoint::GetMinDistanceInvariance()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 0.8f*mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 1.2f*mfMaxDistance;
    }
    cv::Mat MapPoint::GetNormal()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }
    int MapPoint::PredictScale(const float &currentDist, Frame* pF)
    {
        float ratio;
        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
        if (nScale<0)
            nScale = 0;
        else if (nScale >= pF->mnScaleLevels)
            nScale = pF->mnScaleLevels - 1;

        return nScale;
    }
    int MapPoint::PredictScale(const float &currentDist, RefFrame* pF)
    {
        float ratio;
        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
        if (nScale<0)
            nScale = 0;
        else if (nScale >= pF->mnScaleLevels)
            nScale = pF->mnScaleLevels - 1;

        return nScale;
    }

	void MapPoint::SetWorldPos(float x, float y, float z)
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		mWorldPos.at<float>(0) = x;
		mWorldPos.at<float>(1) = y;
		mWorldPos.at<float>(2) = z;
	}

	void MapPoint::AddObservation(RefFrame* pKF, size_t idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		if (mObservations.count(pKF))
			return;
		mObservations[pKF] = idx;

		nObs++;
	}

	void MapPoint::EraseObservation(RefFrame* pKF)
	{
		bool bBad = false;
		{
			std::unique_lock<std::mutex> lock(mMutexFeatures);
			if (mObservations.count(pKF))
			{
				int idx = mObservations[pKF];

				nObs--;

				mObservations.erase(pKF);

				if (mpRefKF == pKF)
					mpRefKF = mObservations.begin()->first;

				// If only 2 observations or less, discard point
				if (nObs <= 0)
					bBad = true;
			}
		}
        if (bBad)
			SetBadFlag();
	}

	void MapPoint::SetBadFlag()
    {
        std::map<RefFrame*, size_t> obs;
        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;
            mObservations.clear();
        }
        for (std::map<RefFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
        {
            RefFrame* pKF = mit->first;
            pKF->EraseMapPointMatch(mit->second);
        }
        MAP->MapPoints.Erase(this->mnID);
    }
    bool MapPoint::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		std::unique_lock<std::mutex> lock2(mMutexPos);
		return mbBad;
	}
	std::map<RefFrame*, size_t> MapPoint::GetObservations()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mObservations;
	}
	int MapPoint::Observations()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return nObs;
	}

	void MapPoint::ComputeDistinctiveDescriptors()
	{
		// Retrieve all observed descriptors
		std::vector<cv::Mat> vDescriptors;

		std::map<RefFrame*, size_t> observations;

		{
			std::unique_lock<std::mutex> lock(mMutexFeatures);

			observations = mObservations;
		}

		if (observations.empty())
			return;

		vDescriptors.reserve(observations.size());

		for (std::map<RefFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			RefFrame* pKF = mit->first;

			//if (!pKF->isBad())
			vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
		}

		if (vDescriptors.empty())
			return;

		// Compute distances between them
		size_t N = vDescriptors.size();
		std::vector<std::vector<float> > Distances;
		Distances.resize(N, std::vector<float>(N, 0));
		for (size_t i = 0; i<N; i++)
		{
			Distances[i][i] = 0;
			for (size_t j = i + 1; j<N; j++)
			{
				float distij = Detector->CalculateDescDistance(vDescriptors[i], vDescriptors[j]);//ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
				Distances[i][j] = distij;
				Distances[j][i] = distij;
			}
		}


		// Take the descriptor with least median distance to the rest
		int BestMedian = INT_MAX;
		int BestIdx = 0;
		for (size_t i = 0; i<N; i++)
		{
			std::vector<int> vDists(Distances[i].begin(), Distances[i].end());
			sort(vDists.begin(), vDists.end());
			int median = vDists[0.5*(N - 1)];

			if (median<BestMedian)
			{
				BestMedian = median;
				BestIdx = i;
			}
		}

		{
			std::unique_lock<std::mutex> lock(mMutexFeatures);
			mDescriptor = vDescriptors[BestIdx].clone();
		}
	}
	

	bool MapPoint::IsInKeyFrame(RefFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return (mObservations.count(pKF));
	}

}