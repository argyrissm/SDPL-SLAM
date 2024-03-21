/**
* This file is part of SDPL-SLAM.
*
* Copyright (C) 2024 Argyris Manetas, National Technical University of Athens
* For more information see <https://github.com/argyrissm/SDPL-SLAM>
*
**/

#ifndef MAP_H
#define MAP_H

#include<opencv2/core/core.hpp>
#include <line_descriptor_custom.hpp>
#include <line_descriptor/descriptor_custom.hpp>
#include <set>


namespace SDPL_SLAM
{

class Map
{
public:
    Map();

    // ==========================================================
    // ============= output for evaluating results ==============

    // ==========================================================
    // ==========================================================

    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // <<<<<<<<<<<<<<<<<<<< output for graph structure >>>>>>>>>>>>>>>>>>>

    // static features and depths detected in image plane. (k*n)
    std::vector<std::vector<cv::KeyPoint> > vpFeatSta;
    std::vector<std::vector<float> > vfDepSta;
    std::vector<std::vector<cv::Mat> > vp3DPointSta;
    // index of temporal matching. (k-1)*n
    std::vector<std::vector<int> > vnAssoSta;
    // feature tracklets: pair.first = frameID; pair.second = featureID;
    std::vector<std::vector<std::pair<int, int> > > TrackletSta;
    std::vector<std::vector<std::pair<int, int>>> TrackletSta_line;
    //static features and depths detected in image plane for lines
    std::vector<std::vector<cv::line_descriptor::KeyLine> > vpFeatSta_line;
    std::vector<std::vector<std::pair<float, float> > > vfDepSta_line;
    std::vector<std::vector<std::pair<cv::Mat, cv::Mat>> > vp3DLineSta;
    std::vector<std::vector<cv::Mat>> vp3DLineStaPlucker;
    std::vector<std::vector<int>> vnAssoSta_line;

    // dynamic feature correspondences and depths detected in image plane. k*n
    std::vector<std::vector<cv::KeyPoint> > vpFeatDyn;
    std::vector<std::vector<float> > vfDepDyn;
    std::vector<std::vector<cv::Mat> > vp3DPointDyn;
    // index of temporal matching. (k-1)*n
    std::vector<std::vector<int> > vnAssoDyn;
    // label indicating which object the feature (3D point) belongs to. (k-1)*n
    std::vector<std::vector<int> > vnFeatLabel;
    // feature tracklets: pair.first = frameID; pair.second = featureID;
    std::vector<std::vector<std::pair<int, int> > > TrackletDyn;
    std::vector<std::vector<std::pair<int, int>>> TrackletDyn_line;
    std::vector<int> nObjID;
    std::vector<int> nObjID_line;

    //dynamic features and depths detected in image plane for lines
    std::vector<std::vector<cv::line_descriptor::KeyLine> > vpFeatDyn_line;
    std::vector<std::vector<std::pair<float, float> > > vfDepDyn_line;
    std::vector<std::vector<std::pair<cv::Mat, cv::Mat>> > vp3DLineDyn;
    std::vector<std::vector<cv::Mat>> vp3DLineDynPlucker;
    std::vector<std::vector<int>> vnAssoDyn_line;
    std::vector<std::vector<int>> vnFeatLabel_line;

    // absolute camera pose of each frame, starting from 1st frame. (k*1)
    std::vector<cv::Mat> vmCameraPose;
    std::vector<cv::Mat> vmCameraPose_RF;  // refine result
    std::vector<cv::Mat> vmCameraPose_GT;  // ground truth result
    // rigid motion of camera and dynamic points. (k-1)*m
    std::vector<std::vector<cv::Mat> > vmRigidCentre;  // ground truth object center
    //I think its the H motion of the objects from the paper
    std::vector<std::vector<cv::Mat> > vmRigidMotion;
    std::vector<std::vector<cv::Mat> > vmObjPosePre; // for new metric 26 Feb 2020
    std::vector<std::vector<cv::Mat> > vmRigidMotion_RF;  // refine result
    std::vector<std::vector<cv::Mat> > vmRigidMotion_GT;  // ground truth result
    std::vector<std::vector<float> > vfAllSpeed_GT; // camera and object speeds
    // rigid motion label in each frame (k-1)*m
    // 0 stands for camera motion; 1,...,l stands for rigid motions.
    std::vector<std::vector<int> > vnRMLabel; // tracking label
    std::vector<std::vector<int> > vnSMLabel; // semantic label
    std::vector<std::vector<int> > vnSMLabelGT;
    // object status (10 Jan 2020)
    std::vector<std::vector<bool> > vbObjStat;
    // object tracking times (10 Jan 2020)
    std::vector<std::vector<int> > vnObjTraTime;
    std::vector<int> nObjTraCount;
    std::vector<int> nObjTraCountGT;
    std::vector<int> nObjTraSemLab;

    // time analysis
    std::vector<float> fLBA_time;
    // (0) frame updating (1) camera estimation (2) object tracking (3) object estimation (4) map updating;
    std::vector<std::vector<float> > vfAll_time;


    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

protected:

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map
    int mnBigChangeIdx;

};

} //namespace SDPL_SLAM

#endif // MAP_H
