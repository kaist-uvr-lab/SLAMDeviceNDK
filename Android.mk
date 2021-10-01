LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libedgeslam
LOCAL_C_INCLUDES := $(LOCAL_PATH)/Thirdparty $(LOCAL_PATH)/Thirdparty/DBoW3/src $(LOCAL_PATH)/Thirdparty/g2o
LOCAL_SRC_FILES :=	$(LOCAL_PATH)/SLAM/UnityLibrary.cpp\
	$(LOCAL_PATH)/SLAM/Camera.cpp\
	$(LOCAL_PATH)/SLAM/CameraPose.cpp\
	$(LOCAL_PATH)/SLAM/Frame.cpp\
	$(LOCAL_PATH)/SLAM/LocalMap.cpp\
	$(LOCAL_PATH)/SLAM/Map.cpp\
	$(LOCAL_PATH)/SLAM/MapPoint.cpp\
	$(LOCAL_PATH)/SLAM/MotionModel.cpp\
	$(LOCAL_PATH)/SLAM/Optimizer.cpp\
	$(LOCAL_PATH)/SLAM/ORBDetector.cpp\
	$(LOCAL_PATH)/SLAM/ORBExtractor.cpp\
	$(LOCAL_PATH)/SLAM/RefFrame.cpp\
	$(LOCAL_PATH)/SLAM/SearchPoints.cpp\
	$(LOCAL_PATH)/SLAM/Tracker.cpp

LOCAL_SHARED_LIBRARIES := g2o dbow3 opencv_java3
LOCAL_CPPFLAGS  += -std=c++11
LOCAL_LDLIBS := -llog

include $(BUILD_SHARED_LIBRARY)

$(call import-add-path, D:\UVR\EdgeSLAMNDK\libedgeslam\src\main\jni)
$(call import-module, Thirdparty)

