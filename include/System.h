/**
* This file is part of SDPL-SLAM.
*
* Copyright (C) 2024 Argyris Manetas, National Technical University of Athens
* For more information see <https://github.com/argyrissm/SDPL-SLAM>
*
**/

#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "Map.h"

namespace SDPL_SLAM
{

using namespace std;

class Map;
class Tracking;

class System
{
public:

    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2
    };

public:

    // Initialize the SLAM system.
    System(const string &strSettingsFile, const eSensor sensor);


    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackRGBD(const cv::Mat &im, cv::Mat &depthmap, const cv::Mat &flowmap, const cv::Mat &masksem,
                      const cv::Mat &mTcw_gt, const vector<vector<float> > &vObjPose_gt, const double &timestamp,
                      cv::Mat &imTraj, const int &nImage);

    void SaveResults(const string &filename);

private:

    // Input sensor
    eSensor mSensor;

    // Map structure.
    Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    Tracking* mpTracker;

};

}// namespace SDPL_SLAM

#endif // SYSTEM_H
