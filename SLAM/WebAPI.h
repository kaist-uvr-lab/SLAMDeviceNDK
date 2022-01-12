#ifndef WEBAPI_H
#define WEBAPI_H
#pragma once

#include "happyhttp.h"
#include <cstring>
#include <sstream>

class WebAPI {
public:
	WebAPI(std::string a, int b);
	virtual ~WebAPI();

	const static char* headers[];
	std::string Send(std::string method, const unsigned char* input, int ndata);
	std::string Send(std::string method, std::string input);
protected:
	std::string ip;
	int port;
	std::stringstream datastream;
};
#endif