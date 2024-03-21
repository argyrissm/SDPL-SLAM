/**
* This file is part of SDPL-SLAM.
*
* Copyright (C) 2024 Argyris Manetas, National Technical University of Athens
* For more information see <https://github.com/argyrissm/SDPL-SLAM>
*
**/

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "Frame.h"
#include "dependencies/g2o/g2o/types/types_six_dof_expmap.h"
#include <opencv2/core/eigen.hpp>
#include "Tracking.h"

namespace SDPL_SLAM
{

using namespace std;

class Optimizer
{
public:


    int static PoseOptimizationNew(Frame *pCurFrame, Frame *pLastFrame, vector<int> &TemperalMatch);
    int static PoseOptimizationNewWithLines(Frame *pCurFrame, Frame *pLastFrame, vector<int> &TemperalMatch, vector<int> &TemperalMatch_Line);

    int static PoseOptimizationFlow2Cam(Frame *pCurFrame, Frame *pLastFrame, vector<int> &TemperalMatch);
    int static PoseOptimizationFlow2CamWithLines(Frame *pCurFrame, Frame *pLastFrame, vector<int> &TemperalMatch, vector<int> &TemperalMatch_Line);

    cv::Mat static PoseOptimizationObjMot(Frame *pCurFrame, Frame *pLastFrame, const vector<int> &ObjId, std::vector<int> &InlierID);
    cv::Mat static PoseOptimizationObjMotWithLines(Frame *pCurFrame, Frame *pLastFrame, const vector<int> &ObjId, const vector<int> &ObjId_Line, std::vector<int> &InlierID, std::vector<int> &InlierID_Line);

    cv::Mat static PoseOptimizationFlow2(Frame *pCurFrame, Frame *pLastFrame, const vector<int> &ObjId, std::vector<int> &InlierID);
    cv::Mat static PoseOptimizationFlow2withLines(Frame *pCurFrame, Frame *pLastFrame, const vector<int> &ObjId, const vector<int> &ObjId_Line, std::vector<int> &InlierID, std::vector<int> &InlierID_Line);
    
    void static FullBatchOptimization(Map* pMap, const cv::Mat Calib_K);
    void static FullBatchOptimizationWithLines(Map* pMap, const cv::Mat Calib_K);
    void static PartialBatchOptimization(Map* pMap, const cv::Mat Calib_K, const int WINDOW_SIZE);
    void static PartialBatchOptimizationWithLines(Map* pMap, const cv::Mat Calib_K, const int WINDOW_SIZE);

    cv::Mat static Get3DinWorld(const cv::KeyPoint &Feats2d, const float &Dpts, const cv::Mat &Calib_K, const cv::Mat &CameraPose);
    std::pair<cv::Mat, cv::Mat> static Get3DinWorld_line(const cv::line_descriptor::KeyLine &Feat_line, const std::pair<float,float> &Dpts, const cv::Mat &Calib_K, const cv::Mat &CameraPose);

    cv::Mat static Get3DinCamera(const cv::KeyPoint &Feats2d, const float &Dpts, const cv::Mat &Calib_K);
    std::pair<cv::Mat, cv::Mat> static Get3DinCamera_line(const cv::line_descriptor::KeyLine &Feat_line, const std::pair<float,float> &Dpts, const cv::Mat &Calib_K);

};

} //namespace SDPL_SLAM

#endif // OPTIMIZER_H
