APP_OPTIM := debug release
APP_ABI := armeabi-v7a
APP_PLATFORM := android-24

APP_STL := gnustl_static
#APP_STL := c++_static
APP_CPPFLAGS := -frtti -fexceptions
APP_CPPFLAGS += -std=c++11
NDK_TOOLCHAIN_VERSION := clang

APP_BUILD_SCRIPT := Android.mk