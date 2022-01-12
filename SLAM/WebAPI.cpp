#include "WebAPI.h"
#ifdef _WIN32
    #include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <netdb.h>	// for gethostbyname()
	#include <errno.h>
	#include <unistd.h>
	#include <sys/select.h>
#endif

WebAPI::WebAPI(std::string a, int b):ip(a), port(b), datastream(""){}
WebAPI::~WebAPI() {}

const char* WebAPI::headers[] = {
	"Connection", "close",
	"Content-type", "application/json",
	"Accept", "text/plain",
	0
};

void OnBegin(const happyhttp::Response* r, void* userdata)
{
	static_cast<std::stringstream*>(userdata)->str("");
}

void OnData(const happyhttp::Response* r, void* userdata, const unsigned char* data, int n)
{
	static_cast<std::stringstream*>(userdata)->write((const char*)data, n);
}

void OnComplete(const happyhttp::Response* r, void* userdata)
{
}

std::string WebAPI::Send(std::string method, std::string input) {
	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(OnBegin, OnData, OnComplete, (void*)&datastream);
	mpConnection->request("POST",
		method.c_str(),
		headers,
		(const unsigned char*)input.c_str(),
		strlen(input.c_str())
	);
	while (mpConnection->outstanding()){
		mpConnection->pump();
	}
	mpConnection->close();
	return datastream.str();
}

std::string WebAPI::Send(std::string method, const unsigned char* input, int ndata) {
	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(OnBegin, OnData, OnComplete, (void*)&datastream);
	mpConnection->request("POST",
		method.c_str(),
		headers,
		input,
		ndata
	);
	while (mpConnection->outstanding()) {
		mpConnection->pump();
	}
	mpConnection->close();
	return datastream.str();
}
