/*
 * file:                Debug.h
 * description:         Minimalist debug handling.
 * author:              (©) Julien Tierny <tierny@telecom-paristech.fr>.
 * date:                February 2011.
 */

#ifndef                 _DEBUG_H
#define                 _DEBUG_H

#include                <fstream>
#include                <iostream>
#include                <sstream>
#include                <string>
#include                <vector>

#ifdef _WIN32
  #include              <float.h>
  #include              <time.h>
  #include              <windows.h>
  #define               isnan(x)      _isnan(x)
#endif

#ifdef __unix__
#include                <unistd.h>
#include                <sys/time.h>
#endif

#ifdef __APPLE__
#include                <unistd.h>
#include                <sys/time.h>
#endif


using namespace std;

class Debug{

  public:

    Debug() { debugLevel_ = 2; };
    ~Debug(){};

    inline const int dMsg(ostream &stream, string msg, 
      const int &debugLevel) const{
      if(debugLevel_ >= debugLevel)
        stream << msg.data();
      return 0;
    }

    virtual inline const int setDebugLevel(const int &debugLevel){
      debugLevel_ = debugLevel;
      return 0;
    };

  protected:

    int                     debugLevel_;
};

class DebugTimer : public Debug{

  public:

    DebugTimer(){ 
      start_ = getTimeStamp();
    };

    DebugTimer(const DebugTimer &other){
      start_ = other.start_;
    }

    inline double getElapsedTime(){
    
      double end = getTimeStamp();
      return end - start_; 
    };

    inline double getTime(){ 
      return start_;
    };

    inline void reStart(){ 
      start_ = getTimeStamp();
    };

  protected:
   
    inline const double getTimeStamp(){

      #ifdef _WIN32
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);

        LARGE_INTEGER temp;
        QueryPerformanceCounter(&temp);

        return (double) temp.QuadPart/frequency.QuadPart;
      #endif

      #ifdef __APPLE__
        struct timeval stamp;
        gettimeofday(&stamp, NULL);
        return (stamp.tv_sec*1000000 + stamp.tv_usec)/1000000.0;
      #endif 

      #ifdef __unix__
        struct timeval stamp;
        gettimeofday(&stamp, NULL);
        return (stamp.tv_sec*1000000 + stamp.tv_usec)/1000000.0;
      #endif

      return 0;
    };

    double          start_;
};

class DebugMemory : public Debug{
  
  public:
    
    inline float getInstantUsage(){
#ifdef __linux__
      // horrible hack since getrusage() doesn't seem to work well under linux
      stringstream procFileName;
      procFileName << "/proc/" << getpid() << "/statm";
      
      ifstream procFile(procFileName.str().data(), ios::in);
      if(procFile){
        float memoryUsage;
        procFile >> memoryUsage;
        procFile.close();
        return memoryUsage/1024.0;
      }

#endif
      return 0;
    };
    
  protected:
};

#endif
