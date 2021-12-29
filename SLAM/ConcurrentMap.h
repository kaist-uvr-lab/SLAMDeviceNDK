#ifndef NEW_MAP_H
#define NEW_MAP_H
#pragma once

#include <mutex>

template <class T1, class T2>
class ConcurrentMap {
public:
	ConcurrentMap();
	virtual ~ConcurrentMap();
private:
	std::mutex mMutex;
	std::map<T1, T2> mMap;
public:
	size_t Count(T1 id);
	void Update(T1 id, T2 data);
	T2   Get(T1 id);
	std::map<T1, T2> Get();
	size_t Size();
	size_t Erase(T1 id);
	void Release();
};

template <class T1, class T2>
ConcurrentMap<T1, T2>::ConcurrentMap() {}
template <class T1, class T2>
ConcurrentMap<T1, T2>::~ConcurrentMap() {}

template <class T1, class T2>
size_t ConcurrentMap<T1, T2>::Count(T1 id) {
	std::unique_lock<std::mutex> lock(mMutex);
	return mMap.count(id);
	//return mMap.count(id) ? true : false;
}
template <class T1, class T2>
void ConcurrentMap<T1, T2>::Update(T1 id, T2 data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mMap[id] = data;
}
template <class T1, class T2>
T2   ConcurrentMap<T1, T2>::Get(T1 id) {
	std::unique_lock<std::mutex> lock(mMutex);
	return mMap[id];
}
template <class T1, class T2>
std::map<T1, T2> ConcurrentMap<T1, T2>::Get() {
	std::unique_lock<std::mutex> lock(mMutex);
	return std::map<T1, T2>(mMap.begin(), mMap.end());
}
template <class T1, class T2>
size_t ConcurrentMap<T1, T2>::Size() {
	std::unique_lock<std::mutex> lock(mMutex);
	return mMap.size();
}

template <class T1, class T2>
size_t ConcurrentMap<T1, T2>::Erase(T1 id) {
	std::unique_lock<std::mutex> lock(mMutex);
	return mMap.erase(id);
}

template <class T1, class T2>
void ConcurrentMap<T1, T2>::Release() {
	std::unique_lock<std::mutex> lock(mMutex);
	std::map<T1, T2>().swap(mMap);
}

#endif