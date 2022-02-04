#ifndef EDGESLAMNDK_CONCURRENTDEQUE_H
#define EDGESLAMNDK_CONCURRENTDEQUE_H
#pragma once

#include <deque>
#include <mutex>

template <class T>
class ConcurrentDeque {
public:
	ConcurrentDeque();
	virtual ~ConcurrentDeque();
private:
	std::mutex mMutex;
	std::deque<T> mDeque;
public:

	void push_back(T data);
	void push_front(T data);
	void pop_front();
	void pop_back();
	T front();
    T back();

	std::vector<T> get();

	size_t size();
	void Release();
	void Clear();
};

template <class T>
ConcurrentDeque<T>::ConcurrentDeque() {
}
template <class T>
ConcurrentDeque<T>::~ConcurrentDeque() {}

template <class T>
void ConcurrentDeque<T>::push_back(T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mDeque.push_back(data);
}
template <class T>
void ConcurrentDeque<T>::push_front(T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mDeque.push_front(data);
}
template <class T>
void ConcurrentDeque<T>::pop_front() {
	std::unique_lock<std::mutex> lock(mMutex);
	mDeque.pop_front();
}
template <class T>
void ConcurrentDeque<T>::pop_back() {
	std::unique_lock<std::mutex> lock(mMutex);
	mDeque.pop_back();
}
template <class T>
T ConcurrentDeque<T>::front() {
	std::unique_lock<std::mutex> lock(mMutex);
	return mDeque.front();
}
template <class T>
T ConcurrentDeque<T>::back() {
	std::unique_lock<std::mutex> lock(mMutex);
	return mDeque.back();
}

template <class T>
std::vector<T> ConcurrentDeque<T>::get() {
	std::deque<T> temp;
	{
		std::unique_lock<std::mutex> lock(mMutex);
		temp = mDeque;
	}
	std::vector<T> res(temp.begin(), temp.end());
	return res;
}

template <class T>
size_t ConcurrentDeque<T>::size() {
	std::unique_lock<std::mutex> lock(mMutex);
	return mDeque.size();
}

template <class T>
void ConcurrentDeque<T>::Release() {
	std::unique_lock<std::mutex> lock(mMutex);
	std::deque<T>().swap(mDeque);
}

template <class T>
void ConcurrentDeque<T>::Clear() {
	std::unique_lock<std::mutex> lock(mMutex);
	mDeque.clear();
}

#endif //EDGESLAMNDK_CONCURRENTDEQUE_H
