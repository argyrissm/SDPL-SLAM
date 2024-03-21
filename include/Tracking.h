/**
* This file is part of SDPL-SLAM.
*
* Copyright (C) 2024 Argyris Manetas, National Technical University of Athens
* For more information see <https://github.com/argyrissm/SDPL-SLAM>
*
**/


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

// eigen
#include <Eigen/Core>

#include"Map.h"
#include"Frame.h"
#include"ORBextractor.h"
#include "System.h"
#include"Lineextractor.h"

#include <mutex>

namespace SDPL_SLAM
{

using namespace std;

class Map;
class System;

class Tracking
{

    struct BirdEyeVizProperties {
        // Bird-eye visualization
        float birdeye_scale_factor_;
        int birdeye_far_plane_;
        int birdeye_left_plane_;
        int birdeye_right_plane_;

        BirdEyeVizProperties() {
            // Bird-eye visualization module params
            birdeye_scale_factor_ = 13.0;
            birdeye_far_plane_ = 50;
            birdeye_left_plane_ = -20;
            birdeye_right_plane_ = 20;
        }
    };

    struct LessPoint2f
    {
        bool operator()(const cv::Point2f& lhs, const cv::Point2f& rhs) const
        {
            return (lhs.x == rhs.x) ? (lhs.y < rhs.y) : (lhs.x < rhs.x);
        }
    };

public:

    Tracking(System* pSys, Map* pMap, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB, cv::Mat &imD, const cv::Mat &imFlow, const cv::Mat &maskSEM,
                          const cv::Mat &mTcw_gt, const vector<vector<float> > &vObjPose_gt, const double &timestamp,
                          cv::Mat &imTraj, const int &nImage);

    // Sparse Scene Flow Vector
    void GetSceneFlowObj();

    // Dynamic Object Tracking
    std::pair<std::vector<std::vector<int> >, std::vector<std::vector<int>>> DynObjTracking();

    // For flow display on 2d plane
    void DrawLine(cv::KeyPoint &keys, cv::Point2f &flow, cv::Mat &ref_image, const cv::Scalar &color,
                  int thickness=2, int line_type=1, const cv::Point2i &offset=cv::Point2i(0,0));
    void DrawTransparentSquare(cv::Point center, cv::Vec3b color, int radius, double alpha, cv::Mat &ref_image);
    void DrawGridBirdeye(double res_x, double res_z, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image);
    void DrawSparseFlowBirdeye(
        const std::vector<Eigen::Vector3d> &pts, const std::vector<Eigen::Vector3d> &vel,
        const cv::Mat &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image);
    void TransformPointToScaledFrustum(double &pose_x, double &pose_z, const BirdEyeVizProperties &viz_props);

    cv::Mat ObjPoseParsingKT(const std::vector<float> &vObjPose_gt);
    cv::Mat ObjPoseParsingOX(const std::vector<float> &vObjPose_gt);

    cv::Mat GetInitModelCam(const std::vector<int> &MatchId, const std::vector<int> &MatchId_Line, std::vector<int> &MatchId_sub, std::vector<int> &MatchId_sub_Line);
    cv::Mat GetInitModelObj(const std::vector<int> &ObjId, std::vector<int> &ObjId_sub, const int objid);

    void StackObjInfo(std::vector<cv::KeyPoint> &FeatDynObj, std::vector<float> &DepDynObj,
                      std::vector<int> &FeatLabObj);
    std::pair<std::vector<std::vector<std::pair<int, int>>>, std::vector<std::vector<std::pair<int, int>>>>  GetStaticTrack();
    std::vector<std::vector<std::pair<int, int> > > GetDynamicTrack();
    std::pair<std::vector<std::vector<std::pair<int, int> > >, std::vector<std::vector<std::pair<int, int> > >> GetDynamicTrackNew();
    std::vector<std::vector<int> > GetObjTrackTime(std::vector<std::vector<int> > &ObjTrackLab, std::vector<std::vector<int> > &ObjSemanticLab,
                                                   std::vector<std::vector<int> > &vnSMLabGT);

    void GetMetricError(const std::vector<cv::Mat> &CamPose, const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &ObjPosePre,
                        const std::vector<cv::Mat> &CamPose_gt, const std::vector<std::vector<cv::Mat> > &RigMot_gt,
                        const std::vector<std::vector<bool> > &ObjStat);
    void PlotMetricError(const std::vector<cv::Mat> &CamPose, const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &ObjPosePre,
                         const std::vector<cv::Mat> &CamPose_gt, const std::vector<std::vector<cv::Mat> > &RigMot_gt,
                         const std::vector<std::vector<bool> > &ObjStat);
    void GetVelocityError(const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &PointDyn,
                          const std::vector<std::vector<int> > &FeaLab, const std::vector<std::vector<int> > &RMLab,
                          const std::vector<std::vector<float> > &Velo_gt, const std::vector<std::vector<int> > &TmpMatch,
                          const std::vector<std::vector<bool> > &ObjStat);
    //Assigns to current frame the points and lines that will be used in the next iteration. 
    void RenewFrameInfo(const std::vector<int> &TM_sta, const std::vector<int> &TM_sta_line);

    void UpdateMask();


public:

    // Tracking states
    enum eTrackingState{
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Dataset Selection
    enum eDataState{
        OMD=1,
        KITTI=2,
        VirtualKITTI=3,
    };

    eDataState mTestData;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    cv::Mat mImGrayLast;  // ++++++ new added

    // new added (Nov 14 2019)
    cv::Mat mFlowMap, mFlowMapLast;
    cv::Mat mDepthMap;
    cv::Mat mSegMap, mSegMapLast;

    // transfer the ground truth to use identity matrix as origin
    cv::Mat mOriginInv;

    // Store temperal matching feature index
    bool bFrame2Frame,bFirstFrame;  // ++++++ new added
    std::vector<int> TemperalMatch, TemperalMatch_subset;  // ++++++ new added
    std::vector<cv::KeyPoint> mvKeysLastFrame, mvKeysCurrentFrame;  // ++++++ new added
    std::vector<cv::line_descriptor::KeyLine> mvKeysCurrentFrame_line;

    //for lines
    std::vector<cv::line_descriptor::KeyLine> mvKeysLastFrame_Line;
    std::vector<int> TemperalMatch_Line, TemperalMatch_subset_Line;

    std::vector<cv::KeyPoint> mvTmpObjKeys;
    std::vector<float> mvTmpObjDepth;
    std::vector<int> mvTmpSemObjLabel;
    std::vector<cv::Point2f> mvTmpObjFlowNext;
    std::vector<cv::KeyPoint> mvTmpObjCorres;

    std::vector<cv::line_descriptor::KeyLine> mvTmpObjKeys_line;
    std::vector<pair<float,float>> mvTmpObjDepth_line;
    std::vector<int> mvTmpSemObjLabel_line;
    std::vector<pair<cv::Point2f,cv::Point2f>> mvTmpObjFlowNext_line;
    std::vector<cv::line_descriptor::KeyLine> mvTmpObjCorres_line;

    //used to create 3d visualization
    cv::viz::Viz3d mWindow;

    // re-projection error
    std::vector<float> repro_e;

    // save current frame ID
    int f_id;

    // save the global Tracking ID
    int max_id;

    // save stop frame
    int StopFrame;

    // save optimization decision
    bool bLocalBatch;
    bool bGlobalBatch;
    // whether use joint optic-flow formulation
    bool bJoint;

    // Window Size and Overlapping Size for Local Batch Optimization
    int nWINDOW_SIZE, nOVERLAP_SIZE;

    // Max Tracking Points on Background and Object in each frame
    int nMaxTrackPointBG, nMaxTrackPointOBJ;

    // Scene Flow Magnitude and Distribution Threshold
    float fSFMgThres, fSFDsThres;

    // save timing values
    std::vector<float> all_timing;

    // use sampled feature or detected feature for background
    int nUseSampleFea;

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization
    void Initialization();

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    //Lines
    Lineextractor* mpLineextractorLeft, *mpLineextractorRight;

    // System
    System* mpSystem;

    //Map
    Map* mpMap;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    float mThDepth;
    float mThDepthObj;

    // The depth map scale factor.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame Info
    Frame mLastFrame;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;
};

} //namespace SDPL_SLAM

#endif // TRACKING_H
