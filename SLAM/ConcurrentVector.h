#ifndef CONCURRENT_VECTOR_H
#define CONCURRENT_VECTOR_H
#pragma once

#include <mutex>

template <class T>
class ConcurrentVector {
public:
	ConcurrentVector();
	virtual ~ConcurrentVector();
private:
	std::mutex mMutex;
	std::vector<T> mVector;
public:
	void Initialize(int N, T data);
	void set(std::vector<T> src);
	void set(ConcurrentVector<T> vec);
	std::vector<T> get();
	T get(int idx);
	void push_back(T data);
	void update(int idx,T data);
	size_t size();
	void Release();
	void Clear();
};

template <class T>
ConcurrentVector<T>::ConcurrentVector() {}
template <class T>
ConcurrentVector<T>::~ConcurrentVector() {}

template <class T>
void ConcurrentVector<T>::Initialize(int N, T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mVector = std::vector<T>(N, data);
}

template <class T>
void ConcurrentVector<T>::set(std::vector<T> src) {
	auto vec = std::vector<T>(src.begin(), src.end());
	std::unique_lock<std::mutex> lock(mMutex);
	mVector = vec;
}

template <class T>
void ConcurrentVector<T>::set(ConcurrentVector<T> vec) {
	std::vector<T> src = vec->get();
	std::unique_lock<std::mutex> lock(mMutex);
	mVector = std::vector<T>(src.begin(), src.end());
}

template <class T>
void ConcurrentVector<T>::push_back(T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mVector.push_back(data);
}

template <class T>
T ConcurrentVector<T>::get(int idx) {
	std::unique_lock<std::mutex> lock(mMutex);
	return mVector[idx];
}

template <class T>
std::vector<T> ConcurrentVector<T>::get(){
	std::vector<T> temp;
	{
		std::unique_lock<std::mutex> lock(mMutex);
		temp = mVector;
	}
	std::vector<T> res(temp.begin(), temp.end());
	return res;
}

template <class T>
void ConcurrentVector<T>::update(int idx, T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mVector[idx] = data;
}

template <class T>
size_t ConcurrentVector<T>::size() {
	std::unique_lock<std::mutex> lock(mMutex);
	return mVector.size();
}

template <class T>
void ConcurrentVector<T>::Release() {
	std::unique_lock<std::mutex> lock(mMutex);
	std::vector<T>().swap(mVector);
}
template <class T>
void ConcurrentVector<T>::Clear() {
	std::unique_lock<std::mutex> lock(mMutex);
	mVector.clear();
}
#endif