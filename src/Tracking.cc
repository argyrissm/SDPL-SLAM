/**
* This file is part of SDPL-SLAM.
*
* Copyright (C) 2024 Argyris Manetas, National Technical University of Athens
* For more information see <https://github.com/argyrissm/SDPL-SLAM>
*
**/


#include "Tracking.h"

#include <Eigen/Core>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <cvplot/cvplot.h>

#include"Converter.h"
#include"Map.h"
#include"Optimizer.h"

#include<iostream>
#include<string>
#include<stdio.h>
#include<math.h>
#include<time.h>

#include<mutex>
#include<unistd.h>

#include <numeric>
#include <algorithm>
#include <map>
#include <random>
#include <fstream>

using namespace std;

bool SortPairInt(const pair<int,int> &a,
              const pair<int,int> &b)
{
    return (a.second > b.second);
}

namespace SDPL_SLAM
{

Tracking::Tracking(System *pSys, Map *pMap, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mpSystem(pSys), mpMap(pMap)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    cout << endl << "Camera Parameters: " << endl << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    //Arguments for the creation of Lineextractor object
    int lsd_nfeatures, lsd_refine, extractor = 0, levels;
    float lsd_scale, scale;

    //Just to try the line extractor !!! It should not be hard coded
    lsd_nfeatures = 0; //800
    lsd_refine = LSD_REFINE_ADV;
    lsd_scale = 0.8;
    levels = 2;
    scale = 2.0;

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    
    mpLineextractorLeft = new Lineextractor(lsd_nfeatures, lsd_refine, lsd_scale, levels, scale, extractor);


    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "System Parameters: " << endl << endl;

    int DataCode = fSettings["ChooseData"];
    switch (DataCode)
    {
        case 1:
            mTestData = OMD;
            cout << "- tested dataset: OMD " << endl;
            break;
        case 2:
            mTestData = KITTI;
            cout << "- tested dataset: KITTI " << endl;
            break;
        case 3:
            mTestData = VirtualKITTI;
            cout << "- tested dataset: Virtual KITTI " << endl;
            break;
    }

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = (float)fSettings["ThDepthBG"];
        mThDepthObj = (float)fSettings["ThDepthOBJ"];
        cout << "- depth threshold (background/object): " << mThDepth << "/" << mThDepthObj << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        cout << "- depth map factor: " << mDepthMapFactor << endl;
    }

    nMaxTrackPointBG = fSettings["MaxTrackPointBG"];
    nMaxTrackPointOBJ = fSettings["MaxTrackPointOBJ"];
    cout << "- max tracking points: " << "(1) background: " << nMaxTrackPointBG << " (2) object: " << nMaxTrackPointOBJ << endl;

    fSFMgThres = fSettings["SFMgThres"];
    fSFDsThres = fSettings["SFDsThres"];
    cout << "- scene flow paras: " << "(1) magnitude: " << fSFMgThres << " (2) percentage: " << fSFDsThres << endl;

    nWINDOW_SIZE = fSettings["WINDOW_SIZE"];
    nOVERLAP_SIZE = fSettings["OVERLAP_SIZE"];
    cout << "- local batch paras: " << "(1) window: " << nWINDOW_SIZE << " (2) overlap: " << nOVERLAP_SIZE << endl;

    nUseSampleFea = fSettings["UseSampleFeature"];
    if (nUseSampleFea==1)
        cout << "- used sampled feature for background scene..." << endl;
    else
        cout << "- used detected feature for background scene..." << endl;
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, cv::Mat &imD, const cv::Mat &imFlow,
                                const cv::Mat &maskSEM, const cv::Mat &mTcw_gt, const vector<vector<float> > &vObjPose_gt,
                                const double &timestamp, cv::Mat &imTraj, const int &nImage)
{
    // initialize some paras
    StopFrame = nImage-1;
    bJoint = true;
    cv::RNG rng((unsigned)time(NULL));

    // Initialize Global ID
    if (mState==NO_IMAGES_YET)
        f_id = 0;

    mImGray = imRGB;
    std::cout << "Frame ID: " << f_id << std::endl;
    // preprocess depth  !!! important for kitti and oxford dataset
    for (int i = 0; i < imD.rows; i++)
    {
        for (int j = 0; j < imD.cols; j++)
        {
            if (imD.at<float>(i,j)<0)
                imD.at<float>(i,j)=0;
            else
            {
                if (mTestData==OMD)
                {
                    // --- for stereo depth map ---
                    //imD.at<float>(i,j) = mbf/(imD.at<float>(i,j)/mDepthMapFactor);
                    // --- for RGB-D depth map ---
                     imD.at<float>(i,j) = imD.at<float>(i,j)/mDepthMapFactor;
                }
                else if (mTestData==KITTI)
                {
                    // --- for stereo depth map ---
                    imD.at<float>(i,j) = mbf/(imD.at<float>(i,j)/mDepthMapFactor);
                    // --- for monocular depth map ---
                    // imD.at<float>(i,j) = imD.at<float>(i,j)/500.0;
                }
            }
        }
    }

    cv::Mat imDepth = imD;

    // Transfer color image to grey image
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // Save map in the tracking head (new added Nov 14 2019)
    mDepthMap = imD;
    mFlowMap = imFlow;
    mSegMap = maskSEM;

    // Initialize timing vector (Output)
    all_timing.resize(5,0);

    // (new added Nov 21 2019)
    if (mState!=NO_IMAGES_YET)
    {
        clock_t s_0, e_0;
        double mask_upd_time;
        s_0 = clock();
        // ****** Update Mask information *******
        UpdateMask();
        e_0 = clock();
        mask_upd_time = (double)(e_0-s_0)/CLOCKS_PER_SEC*1000;
        all_timing[0] = mask_upd_time;
        // cout << "mask updating time: " << mask_upd_time << endl;
    }

    //mCurrentFrame = Frame(mImGray,imDepth,imFlow,maskSEM,timestamp,mpORBextractorLeft,mK,mDistCoef,mbf,mThDepth,mThDepthObj,nUseSampleFea);
    mCurrentFrame = Frame(mImGray,imDepth,imFlow,maskSEM,timestamp,mpORBextractorLeft, mpLineextractorLeft, mK,mDistCoef,mbf,mThDepth,mThDepthObj,nUseSampleFea);
    // std::cout << "mCurrentFrame.mvInfiniteLinesCorr " << mCurrentFrame.mvInfiniteLinesCorr.size() << std::endl;
    // std::cout << "mCurrentFrame.mvStatKeysLineTmp " << mCurrentFrame.mvStatKeysLineTmp.size() << std::endl;
    // ---------------------------------------------------------------------------------------
    // +++++++++++++++++++++++++ For sampled features ++++++++++++++++++++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------

    if(mState!=NO_IMAGES_YET)
    {
        cout << "Update Current Frame From Last....." << endl;

        mCurrentFrame.mvStatKeys = mLastFrame.mvCorres;
        mCurrentFrame.N_s = mCurrentFrame.mvStatKeys.size();

        mCurrentFrame.mvStatKeys_Line = mLastFrame.mvCorresLine;
        //print mCurrentFrame.mvStatKeys_Line 
        // std::cout << "mCurrentFrame.mvStatKeys_Line for debugging " << std::endl;
        // for (int i = 0; i < mCurrentFrame.mvStatKeys_Line.size(); ++i)
        // {
        //     std::cout << "current line i " << i << "curr star point " << mCurrentFrame.mvStatKeys_Line[i].startPointX << " " << mCurrentFrame.mvStatKeys_Line[i].startPointY << "curr end point " << mCurrentFrame.mvStatKeys_Line[i].endPointX << " " << mCurrentFrame.mvStatKeys_Line[i].endPointY << std::endl;
        // }
        
        mCurrentFrame.N_sta_l = mCurrentFrame.mvStatKeys_Line.size();
        //print in file the number of lines for each frame
        // std::ofstream file;
        // file.open("./statistics/line_numbers.txt", std::ios_base::app);
        // file << "Frame " << f_id << " has " << mCurrentFrame.N_sta_l << " static lines" << std::endl;
        // file.close();
        // file.open("./statistics/points_numbers.txt", std::ios_base::app);
        // file << "Frame " << f_id << " has " << mCurrentFrame.N_s << " static points" << std::endl;
        // file.close();
        //std::cout << "Size of mvStatKeys_Line: " << mCurrentFrame.N_sta_l << std::endl;
        mCurrentFrame.mvStatInfiniteLines = mLastFrame.mvInfiniteLinesCorr;

        // //Draw inf line
        // for (int i = 0; i < mCurrentFrame.mvStatInfiniteLines.size(); i++)
        // {
        //     mCurrentFrame.plotINFlines(i);
        // }
        //N_sta_l is the same number with the number of infinite lines

        // Showing line correspondences
        // ---------------------------------------------------------------------------------------

        //std::cout << "Lines in last frame: " << mLastFrame.mvStatKeysLineTmp.size() << std::endl;
        //std::cout << "Lines in current frame: " << mCurrentFrame.mvStatKeys_Line.size() << std::endl;
        
        //Draw the lines on background
        // for (int i = 0; i < mCurrentFrame.N_sta_l; i = i + 6)
        // {
        //     int width = mLastFrame.imGray_.cols + mCurrentFrame.imGray_.cols;
        //     int height = mCurrentFrame.imGray_.rows;
        //     cv::Mat visualization(height, width, CV_8UC1, cv::Scalar(0,0,0));

        //     cv::Mat roi1 = visualization(cv::Rect(0, 0, mLastFrame.imGray_.cols, mLastFrame.imGray_.rows));
        //     mLastFrame.imGray_.copyTo(roi1);
        //     //std::cout << "Last frame image sizes " << mLastFrame.imGray_.cols << " " << mLastFrame.imGray_.rows << std::endl;
        //     cv::Mat roi2 = visualization(cv::Rect(mLastFrame.imGray_.cols, 0, mCurrentFrame.imGray_.cols, mCurrentFrame.imGray_.rows));
        //     mCurrentFrame.imGray_.copyTo(roi2);
        //     std::cout << "Showing line correspondences" << std::endl;
        //     cv::Point2f start1 = mLastFrame.mvStatKeysLineTmp[i].getStartPoint();
        //     cv::Point2f end1 = mLastFrame.mvStatKeysLineTmp[i].getEndPoint();
        //     cv::Point2f start2 = mCurrentFrame.mvStatKeys_Line[i].getStartPoint();
        //     cv::Point2f end2 = mCurrentFrame.mvStatKeys_Line[i].getEndPoint();
        //     start2.x += mLastFrame.imGray_.cols;
        //     end2.x += mLastFrame.imGray_.cols;
        //     cv::line(visualization, start1, start2, cv::Scalar(0, 255, 0), 1);
        //     cv::line(visualization, end1, end2, cv::Scalar(0, 255, 0), 1);
        //     cv::imshow("Correspondences", visualization);
        //     cv::waitKey(0);
        // }
        //Draw Line Correspondences on objects
        // for (int i = 0; i < mLastFrame.mvObjKeys_Linetmp.size(); i++)
        // {
        //     int width = mLastFrame.imGray_.cols + mCurrentFrame.imGray_.cols;
        //     int height = mCurrentFrame.imGray_.rows;
        //     cv::Mat visualization(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        //     cv::cvtColor(mLastFrame.imGray_, visualization(cv::Rect(0, 0, mLastFrame.imGray_.cols, mLastFrame.imGray_.rows)), cv::COLOR_GRAY2BGR);
        //     cv::cvtColor(mCurrentFrame.imGray_, visualization(cv::Rect(mLastFrame.imGray_.cols, 0, mCurrentFrame.imGray_.cols, mCurrentFrame.imGray_.rows)), cv::COLOR_GRAY2BGR);

        //     cv::Point2f start1 = mLastFrame.mvObjKeys_Linetmp[i].getStartPoint();
        //     cv::Point2f end1 = mLastFrame.mvObjKeys_Linetmp[i].getEndPoint();
        //     cv::Point2f start2 = mLastFrame.mvObjCorres_Line[i].getStartPoint();
        //     cv::Point2f end2 = mLastFrame.mvObjCorres_Line[i].getEndPoint();
        //     start2.x += mLastFrame.imGray_.cols;
        //     end2.x += mLastFrame.imGray_.cols;

        //     // Draw green lines on the visualization
        //     cv::line(visualization, start1, start2, cv::Scalar(0, 255, 0), 1);
        //     cv::line(visualization, end1, end2, cv::Scalar(0, 255, 0), 1);

        //     cv::imshow("Correspondences", visualization);
        //     cv::waitKey(0);
        // }
        // ---------------------------------------------------------------------------------------

        // assign the depth value to each keypoint
        mCurrentFrame.mvStatDepth = std::vector<float>(mCurrentFrame.N_s,-1);
        for(int i=0; i<mCurrentFrame.N_s; i++)
        {
            const cv::KeyPoint &kp = mCurrentFrame.mvStatKeys[i];

            const int v = kp.pt.y;
            const int u = kp.pt.x;

            if (u<(mImGray.cols-1) && u>0 && v<(mImGray.rows-1) && v>0)
            {
                float d = imDepth.at<float>(v,u); // be careful with the order  !!!

                if(d>0)
                    mCurrentFrame.mvStatDepth[i] = d;
            }

        }
        mCurrentFrame.mvStatDepth_Line = std::vector<std::pair<float, float>>(mCurrentFrame.N_sta_l, std::make_pair(-1,-1));
        //std::cout << "mCurrentFrame.N_sta_l " << mCurrentFrame.N_sta_l << std::endl;
        // for(int i=0; i<mCurrentFrame.N_sta_l; i++)
        // {
        //     std::cout << "i " << i << std::endl;
        //     const cv::line_descriptor::KeyLine &kl = mCurrentFrame.mvStatKeys_Line[i];
           
        // }
        //std::cout << "mCurrentFrame.mvStatKeys_Line.size() " << mCurrentFrame.mvStatKeys_Line.size() << std::endl;

        for(int i=0; i<mCurrentFrame.N_sta_l; i++)
        {   
            const cv::line_descriptor::KeyLine &kl = mCurrentFrame.mvStatKeys_Line[i];
            const int v_start = (kl.getStartPoint()).y;
            const int u_start = (kl.getStartPoint()).x;
            const int v_end = (kl.getEndPoint()).y;
            const int u_end = (kl.getEndPoint()).x; 
            //u for columns and v for rows
            //TODO: if it does not go in the loop some values are garbage
            if (u_start<(mImGray.cols-1) && u_start>0 && v_start<(mImGray.rows-1) && v_start>0 && u_end<(mImGray.cols-1) && u_end>0 && v_end<(mImGray.rows-1) && v_end>0)
            {
                float d_start = imDepth.at<float>(v_start, u_start); // be careful with the order  !!!
                float d_end = imDepth.at<float>(v_end, u_end); // be careful with the order  !!!
        
                if(d_start > 0 && d_end > 0)
                    mCurrentFrame.mvStatDepth_Line[i] = make_pair(d_start,d_end);
            }
        }
        //std::cout << "Size of mvStatDepth_Line is " << mCurrentFrame.mvStatDepth_Line.size() << std::endl; 
    
        //TO-DO the above for the lines

        // *********** Save object keypoints and depths ************

        // *** first assign current keypoints and depth to last frame
        // *** then assign last correspondences to current frame
        mvTmpObjKeys = mCurrentFrame.mvObjKeys;
        mvTmpObjDepth = mCurrentFrame.mvObjDepth;
        mvTmpSemObjLabel = mCurrentFrame.vSemObjLabel;
        mvTmpObjFlowNext = mCurrentFrame.mvObjFlowNext;
        mvTmpObjCorres = mCurrentFrame.mvObjCorres;

        mCurrentFrame.mvObjKeys = mLastFrame.mvObjCorres;
        mCurrentFrame.mvObjDepth.resize(mCurrentFrame.mvObjKeys.size(),-1);
        mCurrentFrame.vSemObjLabel.resize(mCurrentFrame.mvObjKeys.size(),-1);
        for (int i = 0; i < mCurrentFrame.mvObjKeys.size(); ++i)
        {
            const int u = mCurrentFrame.mvObjKeys[i].pt.x;
            const int v = mCurrentFrame.mvObjKeys[i].pt.y;
            if (u<(mImGray.cols-1) && u>0 && v<(mImGray.rows-1) && v>0 && imDepth.at<float>(v,u)<mThDepthObj && imDepth.at<float>(v,u)>0)
            {
                mCurrentFrame.mvObjDepth[i] = imDepth.at<float>(v,u);
                mCurrentFrame.vSemObjLabel[i] = maskSEM.at<int>(v,u);
            }
            else
            {
                mCurrentFrame.mvObjDepth[i] = 0.1;
                mCurrentFrame.vSemObjLabel[i] = 0;
            }
        }

        //For lines
        mvTmpObjKeys_line = mCurrentFrame.mvObjKeys_Line;
        mvTmpObjDepth_line = mCurrentFrame.mvObjDepth_line;
        mvTmpSemObjLabel_line = mCurrentFrame.vSemObjLabel_Line;
        mvTmpObjFlowNext_line = mCurrentFrame.mvObjFlowNext_Line;
        mvTmpObjCorres_line = mCurrentFrame.mvObjCorres_Line;

        mCurrentFrame.mvObjKeys_Line = mLastFrame.mvObjCorres_Line;
        mCurrentFrame.mvObjDepth_line.resize(mCurrentFrame.mvObjKeys_Line.size(),std::make_pair(-1,-1));
        mCurrentFrame.vSemObjLabel_Line.resize(mCurrentFrame.mvObjKeys_Line.size(),-1);
        //Calculating the depth of the new obj lines that are the ones corresponding to the previous frame
        for (int i = 0; i < mCurrentFrame.mvObjKeys_Line.size(); ++i)
        {
            const int u_start = (mCurrentFrame.mvObjKeys_Line[i].getStartPoint()).x;
            const int v_start = (mCurrentFrame.mvObjKeys_Line[i].getStartPoint()).y;
            const int u_end = (mCurrentFrame.mvObjKeys_Line[i].getEndPoint()).x;
            const int v_end = (mCurrentFrame.mvObjKeys_Line[i].getEndPoint()).y;
            if (u_start<(mImGray.cols-1) && u_start>0 && v_start<(mImGray.rows-1) && v_start>0 && u_end<(mImGray.cols-1) && u_end>0 && v_end<(mImGray.rows-1) && v_end>0 && imDepth.at<float>(v_start,u_start)<mThDepthObj && imDepth.at<float>(v_start,u_start)>0 && imDepth.at<float>(v_end,u_end)<mThDepthObj && imDepth.at<float>(v_end,u_end)>0)
            {
                mCurrentFrame.mvObjDepth_line[i] = make_pair(imDepth.at<float>(v_start,u_start),imDepth.at<float>(v_end,u_end));
                mCurrentFrame.vSemObjLabel_Line[i] = maskSEM.at<int>(v_start,u_start);
            }
            else
            {
                mCurrentFrame.mvObjDepth_line[i] = make_pair(0.1,0.1);
                mCurrentFrame.vSemObjLabel_Line[i] = 0;
            }
        }
        // **********************************************************
        // // show image
        // cv::Mat img_show;
        // cv::drawKeypoints(mImGray, mCurrentFrame.mvObjKeys, img_show, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
         //cv::imshow("Dense Feature Distribution 2", img_show);
         //cv::waitKey(0);
        //cout << "Update Current Frame, Done!" << endl;
    }

    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------

    // // Assign pose ground truth
    if (mState==NO_IMAGES_YET)
    {
        mCurrentFrame.mTcw_gt = Converter::toInvMatrix(mTcw_gt);
        mOriginInv = mTcw_gt;
    }
    else
    {
        mCurrentFrame.mTcw_gt = Converter::toInvMatrix(mTcw_gt)*mOriginInv;
    }


    // Assign object pose ground truth
    mCurrentFrame.nSemPosi_gt.resize(vObjPose_gt.size());
    mCurrentFrame.vObjPose_gt.resize(vObjPose_gt.size());
    for (int i = 0; i < vObjPose_gt.size(); ++i){
        // (1) label
        mCurrentFrame.nSemPosi_gt[i] = vObjPose_gt[i][1];
        // (2) pose
        if (mTestData==OMD)
            mCurrentFrame.vObjPose_gt[i] = ObjPoseParsingOX(vObjPose_gt[i]);
        else if (mTestData==KITTI)
            mCurrentFrame.vObjPose_gt[i] = ObjPoseParsingKT(vObjPose_gt[i]);
    }

    // Save temperal matches for visualization
    TemperalMatch = vector<int>(mCurrentFrame.N_s,-1);

    TemperalMatch_Line = vector<int>(mCurrentFrame.N_sta_l, -1);

    // Initialize object label
    mCurrentFrame.vObjLabel.resize(mCurrentFrame.mvObjKeys.size(),-2);
    mCurrentFrame.vObjLabel_Line.resize(mCurrentFrame.mvObjKeys_Line.size(),-2);
    // *** main ***
    cout << "Start Tracking ......" << endl;
    Track();
    cout << "End Tracking ......" << endl;
    // ************

    // Update Global ID
    f_id = f_id + 1;

    // ---------------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++++++ Display Information ++++++++++++++++++++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------------

    // // // ************** display label on the image ***************  // //
    if(timestamp!=0 && bFrame2Frame == true)
    {
        std::vector<cv::KeyPoint> KeyPoints_tmp(1);
        std::vector<cv::line_descriptor::KeyLine> KeyLines_tmp(1);
        // background features
        // for (int i = 0; i < mCurrentFrame.mvStatKeys.size(); i=i+1)
        // {
        //     KeyPoints_tmp[0] = mCurrentFrame.mvStatKeys[i];
        //     if(maskSEM.at<int>(KeyPoints_tmp[0].pt.y,KeyPoints_tmp[0].pt.x)!=0)
        //         continue;
        //     cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,0,255), 1); // red
        // }
        for (int i = 0; i < TemperalMatch_subset.size(); i=i+1)
        {
            if (TemperalMatch_subset[i]>=mCurrentFrame.mvStatKeys.size())
                continue;
            KeyPoints_tmp[0] = mCurrentFrame.mvStatKeys[TemperalMatch_subset[i]];
            if (KeyPoints_tmp[0].pt.x>=(mImGray.cols-1) || KeyPoints_tmp[0].pt.x<=0 || KeyPoints_tmp[0].pt.y>=(mImGray.rows-1) || KeyPoints_tmp[0].pt.y<=0)
                continue;
            //if(maskSEM.at<int>(KeyPoints_tmp[0].pt.y,KeyPoints_tmp[0].pt.x)!=0)
             //   continue;
            cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,0,255), 1); // red
        }

        for (int i = 0; i < TemperalMatch_Line.size(); i=i+1)
        {
            //std::cout << "TemperalMatch_subset_Line.size() " << TemperalMatch_Line.size() << std::endl;
            //std::cout << "i " << i << std::endl;
            //std::cout << "Tmperal Match_Line is " << TemperalMatch_Line[i] << std::endl;
            if (TemperalMatch_Line[i]>=mCurrentFrame.mvStatKeys_Line.size())
                continue;
            if (TemperalMatch_Line[i] == -1)
                continue;
            KeyLines_tmp[0] = mCurrentFrame.mvStatKeys_Line[TemperalMatch_Line[i]];
            if ((KeyLines_tmp[0].getStartPoint()).x >= (mImGray.cols-1) || (KeyLines_tmp[0].getStartPoint()).x <= 0 || (KeyLines_tmp[0].getStartPoint()).y >= (mImGray.rows-1) || (KeyLines_tmp[0].getStartPoint()).y <= 0 || (KeyLines_tmp[0].getEndPoint()).x <= 0 || (KeyLines_tmp[0].getEndPoint()).x >= (mImGray.cols-1) || (KeyLines_tmp[0].getEndPoint()).y <= 0 || (KeyLines_tmp[0].getEndPoint()).y >= (mImGray.rows-1))
                continue;
            
            cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(255, 0, 0), 2);
        }

        // static and dynamic objects line
        for (int i = 0; i < mCurrentFrame.vObjLabel_Line.size(); ++i)
        {
            if (mCurrentFrame.vObjLabel_Line[i]==-1 || mCurrentFrame.vObjLabel_Line[i]==-2)
                continue;
            int l = mCurrentFrame.vObjLabel_Line[i];
            if (l>25)
                l = l/2;
            // int l = mCurrentFrame.vSemObjLabel[i];
            // cout << "label: " << l << endl;
            KeyLines_tmp[0] = mCurrentFrame.mvObjKeys_Line[i];
            if ((KeyLines_tmp[0].getStartPoint()).x >= (mImGray.cols-1) || (KeyLines_tmp[0].getStartPoint()).x <= 0 || (KeyLines_tmp[0].getStartPoint()).y >= (mImGray.rows-1) || (KeyLines_tmp[0].getStartPoint()).y <= 0 || (KeyLines_tmp[0].getEndPoint()).x <= 0 || (KeyLines_tmp[0].getEndPoint()).x >= (mImGray.cols-1) || (KeyLines_tmp[0].getEndPoint()).y <= 0 || (KeyLines_tmp[0].getEndPoint()).y >= (mImGray.rows-1))
                continue;
            //check if nan
            if (std::isnan(KeyLines_tmp[0].getStartPoint().x) || std::isnan(KeyLines_tmp[0].getStartPoint().y) || std::isnan(KeyLines_tmp[0].getEndPoint().x) || std::isnan(KeyLines_tmp[0].getEndPoint().y))
                continue;
            //check that the line endpoints are not 0 by checking if they are less than a very small value
            if (KeyLines_tmp[0].getStartPoint().x < 1e-5 || KeyLines_tmp[0].getStartPoint().y < 1e-5 || KeyLines_tmp[0].getEndPoint().x < 1e-5 || KeyLines_tmp[0].getEndPoint().y < 1e-5)
                continue;
            // std::cout << "KeyLines_tmp[0].getStartPoint() " << KeyLines_tmp[0].getStartPoint() << std::endl;
            // std::cout << "KeyLines_tmp[0].getEndPoint() " << KeyLines_tmp[0].getEndPoint() << std::endl;

            switch (l)
            {
                case 0:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2); // red
                    break;
                case 1:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2); // 255, 165, 0
                    break;
                case 2:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 3:

                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2); // 255,255,0
                    break;
                case 4:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2); // 255,192,203
                    break;
                case 5:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 6:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 7:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 8:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 9:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 10:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 11:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 12:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 13:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2); // red
                    break;
                case 14:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 15:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 16:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 17:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 18:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 19:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 20:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 21:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 22:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 23:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 24:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 25:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
                case 41:
                    cv::line(imRGB, KeyLines_tmp[0].getStartPoint(), KeyLines_tmp[0].getEndPoint(), cv::Scalar(0, 255, 0), 2);
                    break;
            }
        }

        // static and dynamic objects
        // for (int i = 0; i < mCurrentFrame.vObjLabel.size(); ++i)
        // {
        //     if(mCurrentFrame.vObjLabel[i]==-1 || mCurrentFrame.vObjLabel[i]==-2)
        //         continue;
        //     int l = mCurrentFrame.vObjLabel[i];
        //     if (l>25)
        //         l = l/2;
        //     // int l = mCurrentFrame.vSemObjLabel[i];
        //     // cout << "label: " << l << endl;
        //     KeyPoints_tmp[0] = mCurrentFrame.mvObjKeys[i];
        //     if (KeyPoints_tmp[0].pt.x>=(mImGray.cols-1) || KeyPoints_tmp[0].pt.x<=0 || KeyPoints_tmp[0].pt.y>=(mImGray.rows-1) || KeyPoints_tmp[0].pt.y<=0)
        //         continue;
        //     switch (l)
        //     {
        //         case 0:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,0,255), 1); // red
        //             break;
        //         case 1:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 0, 128), 1); // 255, 165, 0
        //             break;
        //         case 2:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,255,0), 1);
        //             break;
        //         case 3:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0, 255, 0), 1); // 255,255,0
        //             break;
        //         case 4:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,0,0), 1); // 255,192,203
        //             break;
        //         case 5:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,255,255), 1);
        //             break;
        //         case 6:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 0, 128), 1);
        //             break;
        //         case 7:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,255,255), 1);
        //             break;
        //         case 8:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,228,196), 1);
        //             break;
        //         case 9:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(180, 105, 255), 1);
        //             break;
        //         case 10:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(165,42,42), 1);
        //             break;
        //         case 11:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(35, 142, 107), 1);
        //             break;
        //         case 12:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(45, 82, 160), 1);
        //             break;
        //         case 13:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,0,255), 1); // red
        //             break;
        //         case 14:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255, 165, 0), 1);
        //             break;
        //         case 15:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,255,0), 1);
        //             break;
        //         case 16:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,255,0), 1);
        //             break;
        //         case 17:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,192,203), 1);
        //             break;
        //         case 18:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0,255,255), 1);
        //             break;
        //         case 19:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 0, 128), 1);
        //             break;
        //         case 20:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,255,255), 1);
        //             break;
        //         case 21:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255,228,196), 1);
        //             break;
        //         case 22:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(180, 105, 255), 1);
        //             break;
        //         case 23:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(165,42,42), 1);
        //             break;
        //         case 24:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(35, 142, 107), 1);
        //             break;
        //         case 25:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(45, 82, 160), 1);
        //             break;
        //         case 41:
        //             cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(60, 20, 220), 1);
        //             break;
        //     }
        // }

        //cv::Mat mImBGR(mImGray.size(), CV_8UC3);
        //cvtColor(mImGray, mImBGR, CV_GRAY2RGB);
        if (mTestData == KITTI)
        {
        for (int i = 0; i < mCurrentFrame.vObjBoxID.size(); ++i)
        {
            if (mCurrentFrame.vSpeed[i].x==0)
                continue;
            // cout << "ID: " << mCurrentFrame.vObjBoxID[i] << endl;
            cv::Point pt1(vObjPose_gt[mCurrentFrame.vObjBoxID[i]][2], vObjPose_gt[mCurrentFrame.vObjBoxID[i]][3]);
            cv::Point pt2(vObjPose_gt[mCurrentFrame.vObjBoxID[i]][4], vObjPose_gt[mCurrentFrame.vObjBoxID[i]][5]);
            // cout << pt1.x << " " << pt1.y << " " << pt2.x << " " << pt2.y << endl;
            cv::rectangle(imRGB, pt1, pt2, cv::Scalar(0, 255, 0),2);
            // string sp_gt = std::to_string(mCurrentFrame.vSpeed[i].y);
            string sp_est = std::to_string(mCurrentFrame.vSpeed[i].x/36);
            // sp_gt.resize(5);
            sp_est.resize(5);
            // string output_gt = "GT:" + sp_gt + "km/h";
            string output_est = sp_est + "km/h";
            cv::putText(imRGB, output_est, cv::Point(pt1.x, pt1.y-10), cv::FONT_HERSHEY_DUPLEX, 0.9, CV_RGB(0,255,0), 2); // CV_RGB(255,140,0)
            // cv::putText(mImBGR, output_gt, cv::Point(pt1.x, pt1.y-32), cv::FONT_HERSHEY_DUPLEX, 0.7, CV_RGB(255, 0, 0), 2);
        }
        }
        cv::imshow("Static Background and Object Points", imRGB);
        //save images at a path
        cv::imwrite("feat.png",imRGB);
        //wait until i press something
        //cv::waitKey(0);
        
        // cv::imwrite("feat.png",imRGB);
        if (f_id<4)
            cv::waitKey(4);
        else
            cv::waitKey(4);

    }

    // ************** show bounding box with speed ***************
    if(timestamp!=0 && bFrame2Frame == true && mTestData==KITTI)
    {
        cv::Mat mImBGR(mImGray.size(), CV_8UC3);
        cvtColor(mImGray, mImBGR, CV_GRAY2RGB);
        for (int i = 0; i < mCurrentFrame.vObjBoxID.size(); ++i)
        {
            if (mCurrentFrame.vSpeed[i].x==0)
                continue;
            // cout << "ID: " << mCurrentFrame.vObjBoxID[i] << endl;
            cv::Point pt1(vObjPose_gt[mCurrentFrame.vObjBoxID[i]][2], vObjPose_gt[mCurrentFrame.vObjBoxID[i]][3]);
            cv::Point pt2(vObjPose_gt[mCurrentFrame.vObjBoxID[i]][4], vObjPose_gt[mCurrentFrame.vObjBoxID[i]][5]);
            // cout << pt1.x << " " << pt1.y << " " << pt2.x << " " << pt2.y << endl;
            cv::rectangle(mImBGR, pt1, pt2, cv::Scalar(0, 255, 0),2);
            // string sp_gt = std::to_string(mCurrentFrame.vSpeed[i].y);
            string sp_est = std::to_string(mCurrentFrame.vSpeed[i].x/36);
            // sp_gt.resize(5);
            sp_est.resize(5);
            // string output_gt = "GT:" + sp_gt + "km/h";
            string output_est = sp_est + "km/h";
            cv::putText(mImBGR, output_est, cv::Point(pt1.x, pt1.y-10), cv::FONT_HERSHEY_DUPLEX, 0.9, CV_RGB(0,255,0), 2); // CV_RGB(255,140,0)
            // cv::putText(mImBGR, output_gt, cv::Point(pt1.x, pt1.y-32), cv::FONT_HERSHEY_DUPLEX, 0.7, CV_RGB(255, 0, 0), 2);
        }
        cv::imshow("Object Speed", mImBGR);
        cv::waitKey(1);
    }

    // // ************** show trajectory results ***************
    if (mTestData==KITTI)
    {
        int sta_x = 300, sta_y = 100, radi = 1, thic = 2;  // (160/120/2/5)
        float scale = 6; // 6
        cv::Mat CamPos = Converter::toInvMatrix(mCurrentFrame.mTcw);
        int x = int(CamPos.at<float>(0,3)*scale) + sta_x;
        int y = int(CamPos.at<float>(2,3)*scale) + sta_y;
        // cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(255,0,0), thic);
        cv::rectangle(imTraj, cv::Point(x, y), cv::Point(x+5, y+5), cv::Scalar(0,0,255),1);
        cv::rectangle(imTraj, cv::Point(10, 30), cv::Point(550, 60), CV_RGB(0,0,0), CV_FILLED);
        cv::putText(imTraj, "Camera Trajectory (RED SQUARE)", cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 255), 1);
        char text[100];
        sprintf(text, "x = %02fm y = %02fm z = %02fm", CamPos.at<float>(0,3), CamPos.at<float>(1,3), CamPos.at<float>(2,3));
        cv::putText(imTraj, text, cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar::all(255), 1);
        cv::putText(imTraj, "Object Trajectories (COLORED CIRCLES)", cv::Point(10, 70), cv::FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 255), 1);

        for (int i = 0; i < mCurrentFrame.vObjCentre3D.size(); ++i)
        {
            if (mCurrentFrame.vObjCentre3D[i].at<float>(0,0)==0 && mCurrentFrame.vObjCentre3D[i].at<float>(0,2)==0)
                continue;
            int x = int(mCurrentFrame.vObjCentre3D[i].at<float>(0,0)*scale) + sta_x;
            int y = int(mCurrentFrame.vObjCentre3D[i].at<float>(0,2)*scale) + sta_y;
            // int l = mCurrentFrame.nSemPosition[i];
            int l = mCurrentFrame.nModLabel[i];
            switch (l)
            {
                case 1:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(128, 0, 128), thic); // orange
                    break;
                case 2:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0,255,255), thic); // green
                    break;
                case 3:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0, 255, 0), thic); // yellow
                    break;
                case 4:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0,0,255), thic); // pink
                    break;
                case 5:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(255,255,0), thic); // cyan (yellow green 47,255,173)
                    break;
                case 6:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(128, 0, 128), thic); // purple
                    break;
                case 7:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(255,255,255), thic);  // white
                    break;
                case 8:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(196,228,255), thic); // bisque
                    break;
                case 9:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(180, 105, 255), thic);  // blue
                    break;
                case 10:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(42,42,165), thic);  // brown
                    break;
                case 11:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(35, 142, 107), thic);
                    break;
                case 12:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(45, 82, 160), thic);
                    break;
                case 41:
                    cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(60, 20, 220), thic);
                    break;
            }
        }

        imshow( "Camera and Object Trajectories", imTraj);
        if (f_id<3)
            cv::waitKey(1);
        else
            cv::waitKey(1);
    }


    // if(timestamp!=0 && bFrame2Frame == true && mTestData==OMD)
    // {
    //     PlotMetricError(mpMap->vmCameraPose,mpMap->vmRigidMotion, mpMap->vmObjPosePre,
    //                    mpMap->vmCameraPose_GT,mpMap->vmRigidMotion_GT, mpMap->vbObjStat);
    // }


    // ************** display temperal matching ***************
    // if(timestamp!=0 && bFrame2Frame == true)
    // {
    //     std::vector<cv::KeyPoint> PreKeys, CurKeys;
    //     std::vector<cv::DMatch> TemperalMatches;
    //     int count =0;
    //     for(int iL=0; iL<mvKeysCurrentFrame.size(); iL=iL+50)
    //     {
    //         if(maskSEM.at<int>(mvKeysCurrentFrame[iL].pt.y,mvKeysCurrentFrame[iL].pt.x)!=0)
    //             continue;
    //         // if(TemperalMatch[iL]==-1)
    //         //     continue;
    //         // if(checkit[iL]==0)
    //         //     continue;
    //         // if(mCurrentFrame.vObjLabel[iL]<=0)
    //         //     continue;
    //         // if(cv::norm(mCurrentFrame.vFlow_3d[iL])<0.15)
    //         //     continue;
    //         PreKeys.push_back(mvKeysLastFrame[TemperalMatch[iL]]);
    //         CurKeys.push_back(mvKeysCurrentFrame[iL]);
    //         TemperalMatches.push_back(cv::DMatch(count,count,0));
    //         count = count + 1;
    //     }
    //     // cout << "temperal features numeber: " << count <<  endl;

    //     cv::Mat img_matches;
    //     drawMatches(mImGrayLast, PreKeys, mImGray, CurKeys,
    //                 TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
    //                 vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //     cv::resize(img_matches, img_matches, cv::Size(img_matches.cols/1.0, img_matches.rows/1.0));
    //     cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
    //     cv::imshow("temperal matches", img_matches);
    //     cv::waitKey(0);
    // }

    //display temperal matching for endpoints of lines    
    // if(timestamp!=0 && bFrame2Frame == true)
    // {
    //     std::vector<cv::KeyPoint> PreKeys, CurKeys;
    //     std::vector<cv::DMatch> TemperalMatches_line;
    //     int count = 0;
    //     for (int k =0; k < mvKeysLastFrame_Line.size(); ++k)
    //     {
    //         std::cout << "mvKeysLastFrame_Line" << mvKeysLastFrame_Line[k].getStartPoint().x << " " << mvKeysLastFrame_Line[k].getStartPoint().y << " " << mvKeysLastFrame_Line[k].getEndPoint().x << " " << mvKeysLastFrame_Line[k].getEndPoint().y << std::endl;
    //     }
    //     for (int iL=0; iL < mvKeysCurrentFrame_line.size(); iL = iL + 20)
    //     {
    //         if (maskSEM.at<int>(mvKeysCurrentFrame_line[iL].getStartPoint().y, mvKeysCurrentFrame_line[iL].getStartPoint().x) != 0 && maskSEM.at<int>(mvKeysCurrentFrame_line[iL].getEndPoint().y, mvKeysCurrentFrame_line[iL].getEndPoint().x) != 0)
    //             continue;
    //         double epsilon = 1e-10;
    //         if (TemperalMatch_Line[iL] == -1)
    //         {
    //             //this means that this line has been found for the first time
    //             continue;
    //         }
    //         std::cout << "mvKeysLastFrame_Line is " << mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getStartPoint().x << " " <<  mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getStartPoint().y << " " << mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getEndPoint().x << " " << mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getEndPoint().y << std::endl;
    //         if ((mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getStartPoint().x < epsilon && mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getStartPoint().y < epsilon) || ( mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getEndPoint().x < epsilon && mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getEndPoint().y < epsilon))
    //         {
    //             std::cout << "A point with only zeros is found" << std::endl;
    //             std::cout << "Size of mvKeysLastFrame_line is " << mvKeysLastFrame_Line.size() << std::endl;
    //             std::cout << "iL is " << iL << std::endl;
    //             std::cout << "TemperalMatch_Line[iL] is " << TemperalMatch_Line[iL] << std::endl;
    //         }
    //         //std::cout << "Points in temperal matching " << std::endl;
    //         cv::Point2f point = mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getStartPoint();
    //         PreKeys.push_back(cv::KeyPoint(point.x, point.y, 1));
    //         //std::cout << "point.x " << point.x << " point.y " << point.y << std::endl;
    //         point = mvKeysLastFrame_Line[TemperalMatch_Line[iL]].getEndPoint();
    //         //std::cout << "point.x " << point.x << " point.y " << point.y << std::endl;
    //         PreKeys.push_back(cv::KeyPoint(point.x, point.y, 1));
    //         point = mvKeysCurrentFrame_line[iL].getStartPoint();
    //         //std::cout << "point.x " << point.x << " point.y " << point.y << std::endl;
    //         CurKeys.push_back(cv::KeyPoint(point.x, point.y, 1));
    //         point = mvKeysCurrentFrame_line[iL].getEndPoint();
    //         //std::cout << "point.x " << point.x << " point.y " << point.y << std::endl;
    //         CurKeys.push_back(cv::KeyPoint(point.x, point.y, 1));

    //         TemperalMatches_line.push_back(cv::DMatch(count, count, 0));
    //         TemperalMatches_line.push_back(cv::DMatch(count+1, count+1, 0));
    //         count = count + 2;
    //     }

    //     cv::Mat img_matches;
    //     drawMatches(mImGrayLast, PreKeys, mImGray, CurKeys,
    //                 TemperalMatches_line, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
    //                 vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //             cv::resize(img_matches, img_matches, cv::Size(img_matches.cols/1.0, img_matches.rows/1.0));
    //     cv::namedWindow("temperal matches lines", cv::WINDOW_NORMAL);
    //     cv::imshow("temperal matches lines", img_matches);
    //     //cv::waitKey(0);


    // }


    // ---------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------

    mImGrayLast = mImGray;
    TemperalMatch.clear();
    mSegMapLast = mSegMap;   // new added Nov 21 2019
    mFlowMapLast = mFlowMap; // new added Nov 21 2019

    return mCurrentFrame.mTcw.clone();
}


void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
        mState = NOT_INITIALIZED;

    mLastProcessedState=mState;


    if(mState==NOT_INITIALIZED)
    {
        bFirstFrame = true;
        bFrame2Frame = false;

        if(mSensor==System::RGBD)
            Initialization();

        if(mState!=OK)
            return;
    }
    else
    {
        bFrame2Frame = true;

        cout << "--------------------------------------------" << endl;
        cout << "..........Dealing with Camera Pose.........." << endl;
        cout << "--------------------------------------------" << endl;

        // // *********** Update TemperalMatch ***********
        for (int i = 0; i < mCurrentFrame.N_s; ++i){
            TemperalMatch[i] = i;
        }

        for (int i = 0; i < mCurrentFrame.N_sta_l; ++i)
        {
            TemperalMatch_Line[i] = i;
        }
        // // ********************************************
        int temp_counter = 0;
        //check that stat line is valid
        for (int i = 0; i < mCurrentFrame.N_sta_l; ++i)
        {
            
            float start_x = mCurrentFrame.mvStatKeys_Line[i].startPointX;
            float start_y = mCurrentFrame.mvStatKeys_Line[i].startPointY;
            float end_x = mCurrentFrame.mvStatKeys_Line[i].endPointX;
            float end_y = mCurrentFrame.mvStatKeys_Line[i].endPointY;
            //depth discontinuity
            float depth_start = mDepthMap.at<float>(start_y, start_x);
            float depth_end = mDepthMap.at<float>(end_y, end_x);
            float mid_x = (start_x + end_x)/2;
            float mid_y = (start_y + end_y)/2;
            float depth_mid = mDepthMap.at<float>(mid_y, mid_x);
            float depth_expected = (depth_start + depth_end)/2;
            float D_threshold = 10.0;
            float line_length  = sqrt( pow((start_x - end_x),2) + pow((start_y - end_y),2) );
            D_threshold = D_threshold * (line_length / 1000);
            if (fabs(depth_mid - depth_expected) > D_threshold)
            {
                //std::cout << "depth discontinuity " << std::endl;
                //remove the line
                TemperalMatch_Line.erase(TemperalMatch_Line.begin() + i - temp_counter);
                temp_counter = temp_counter + 1;
                continue;
            }
            if (mSegMap.at<int>(start_y, start_x) != 0 || mSegMap.at<int>(end_y, end_x) != 0)
            {
                //remove the line
                TemperalMatch_Line.erase(TemperalMatch_Line.begin() + i - temp_counter);
                temp_counter = temp_counter + 1;
                continue;
            }
        }

        //write in file number of lines after removing depth discontinuity
        // std::ofstream file;
        // file.open("./statistics/line_with_valid_depth_numbers.txt", std::ios_base::app);
        // file << "Frame " << f_id << " Number of lines after removing depth discontinuity: " << TemperalMatch_Line.size() << std::endl;
        // file.close();


        clock_t s_1_1, s_1_2, e_1_1, e_1_2;
        double cam_pos_time;
        s_1_1 = clock();
        // Get initial estimate using P3P plus RanSac
        cv::Mat iniTcw = GetInitModelCam(TemperalMatch, TemperalMatch_Line, TemperalMatch_subset, TemperalMatch_subset_Line);
        e_1_1 = clock();


        s_1_2 = clock();
        // cout << "the ground truth pose: " << endl << mCurrentFrame.mTcw_gt << endl;
        // cout << "initial pose: " << endl << iniTcw << endl;
        // // compute the pose with new matching
        mCurrentFrame.SetPose(iniTcw);
        #define USE_LINE
        if (bJoint)
        {
            #if defined(USE_LINE)
                std::cout << "pose optimization with lines" << std::endl;
                Optimizer::PoseOptimizationFlow2CamWithLines(&mCurrentFrame, &mLastFrame, TemperalMatch_subset, TemperalMatch_Line);
            #else
                Optimizer::PoseOptimizationFlow2Cam(&mCurrentFrame, &mLastFrame, TemperalMatch_subset);
            #endif
        }
        else
        {
            #if defined(USE_LINE)
                std::cout << "pose optimization with lines" << std::endl;
                Optimizer::PoseOptimizationNewWithLines(&mCurrentFrame, &mLastFrame, TemperalMatch_subset, TemperalMatch_Line);
            #else
                std::cout << "pose optimization without lines" << std::endl;
                Optimizer::PoseOptimizationNew(&mCurrentFrame, &mLastFrame, TemperalMatch_subset);
            #endif
        }
        int inl_counter, inl_counter_line;
        inl_counter = 0;
        inl_counter_line = 0;
        for (int t = 0; t < TemperalMatch_subset.size(); ++t)
        {
            if (TemperalMatch_subset[t] != -1)
            {
                inl_counter = inl_counter + 1;
            }
        }
        for (int t = 0; t < TemperalMatch_Line.size(); ++t)
        {
            if (TemperalMatch_Line[t] != -1)
            {
                inl_counter_line = inl_counter_line + 1;
            }
        }
        // file.open("./statistics/inlier_stat_number.txt", std::ios_base::app);
        // file << "Frame " << f_id << " Number of inliers after pose optimization: " << inl_counter << std::endl;
        // file.close();
        // file.open("./statistics/inlier_stat_line_number.txt", std::ios_base::app);
        // file << "Frame " << f_id << " Number of inliers lines after pose optimization " << inl_counter_line << std::endl;
        // file.close();

        //std::cout << "TemperalMatch_Line size after pose optimization " << TemperalMatch_Line.size() << std::endl;
        // for (int g = 0; g < TemperalMatch_Line.size(); ++g)
        // {
        //     std::cout << "TemperalMatch_Line[g] "  << TemperalMatch_Line[g] << std::endl;
        // }
        cout << "pose after update: " << endl << mCurrentFrame.mTcw << endl;
        e_1_2 = clock();
        cam_pos_time = (double)(e_1_1-s_1_1)/CLOCKS_PER_SEC*1000 + (double)(e_1_2-s_1_2)/CLOCKS_PER_SEC*1000;
        all_timing[1] = cam_pos_time;
        // cout << "camera pose estimation time: " << cam_pos_time << endl;

        // Update motion model
        if(!mLastFrame.mTcw.empty())
        {
            cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
            mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
            mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
            mVelocity = mCurrentFrame.mTcw*LastTwc;
        }

        // ----------- compute camera pose error ----------

        // cv::Mat Tcw_est_inv = Converter::toInvMatrix(mCurrentFrame.mTcw);
        // cv::Mat RePoEr_cam = Tcw_est_inv*mCurrentFrame.mTcw_gt;
        // cout << "error matrix: " << endl << RePoEr_cam << endl;
        cv::Mat T_lc_inv = mCurrentFrame.mTcw*Converter::toInvMatrix(mLastFrame.mTcw);
        cv::Mat T_lc_gt = mLastFrame.mTcw_gt*Converter::toInvMatrix(mCurrentFrame.mTcw_gt);
        cv::Mat RePoEr_cam = T_lc_inv*T_lc_gt;

        float t_rpe_cam = std::sqrt( RePoEr_cam.at<float>(0,3)*RePoEr_cam.at<float>(0,3) + RePoEr_cam.at<float>(1,3)*RePoEr_cam.at<float>(1,3) + RePoEr_cam.at<float>(2,3)*RePoEr_cam.at<float>(2,3) );
        float trace_rpe_cam = 0;
        for (int i = 0; i < 3; ++i)
        {
            if (RePoEr_cam.at<float>(i,i)>1.0)
                trace_rpe_cam = trace_rpe_cam + 1.0-(RePoEr_cam.at<float>(i,i)-1.0);
            else
                trace_rpe_cam = trace_rpe_cam + RePoEr_cam.at<float>(i,i);
        }
        cout << std::fixed << std::setprecision(6);
        float r_rpe_cam = acos( (trace_rpe_cam -1.0)/2.0 )*180.0/3.1415926;

        cout << "the relative pose error of estimated camera pose, " << "t: " << t_rpe_cam <<  " R: " << r_rpe_cam << endl;

        //write it also to a file
        // std::string filePath = "./Results/RelativePoseError.txt";
        // std::ofstream outputFile(filePath, std::ios::app);

        // if (!outputFile.is_open())
        // {
        //     std::cerr << "Error opening the file" << std::endl;
        //     exit(EXIT_FAILURE);
        // }

        // outputFile << "the relative pose error of estimated camera pose, " << "t: " << t_rpe_cam <<  " R: " << r_rpe_cam << endl;
        // outputFile.close();

        // // **** show the picked points ****
        // std::vector<cv::KeyPoint> PickKeys;
        // for (int j = 0; j < TemperalMatch_subset.size(); ++j){
        //     PickKeys.push_back(mCurrentFrame.mvStatKeys[TemperalMatch_subset[j]]);
        // }
        // cv::drawKeypoints(mImGray, PickKeys, mImGray, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        // cv::imshow("KeyPoints and Grid on Background", mImGray);
        // cv::waitKey(0);

        // -------------------------------------------------

        cout << "--------------------------------------------" << endl;
        cout << "..........Dealing with Objects Now.........." << endl;
        cout << "--------------------------------------------" << endl;

        // // ====== compute sparse scene flow to the found matches =======
        GetSceneFlowObj();
        //TODO add to scene flow lines (?)
        // // ---------------------------------------------------------------------------------------
        // // ++++++++++++++++++++++++++++++++ Dynamic Object Tracking ++++++++++++++++++++++++++++++
        // // ---------------------------------------------------------------------------------------

        cout << "Object Tracking ......" << endl;
        std::pair<std::vector<std::vector<int> >, std::vector<std::vector<int>>> ObjIdNew_pair = DynObjTracking();
        std::vector<std::vector<int>> ObjIdNew = ObjIdNew_pair.first;
        std::vector<std::vector<int>> ObjIdNew_Line = ObjIdNew_pair.second;
        cout << "Object Tracking, Done!" << endl;

        // // ---------------------------------------------------------------------------------------
        // // ++++++++++++++++++++++++++++++ Object Motion Estimation +++++++++++++++++++++++++++++++
        // // ---------------------------------------------------------------------------------------

        clock_t s_3_1, s_3_2, e_3_1, e_3_2;
        double obj_mot_time = 0, t_con = 0;

        mCurrentFrame.bObjStat.resize(ObjIdNew.size(),true);
        mCurrentFrame.vObjMod.resize(ObjIdNew.size());
        mCurrentFrame.vObjPosePre.resize(ObjIdNew.size());
        mCurrentFrame.vObjMod_gt.resize(ObjIdNew.size());
        mCurrentFrame.vObjSpeed_gt.resize(ObjIdNew.size());
        mCurrentFrame.vSpeed.resize(ObjIdNew.size());
        mCurrentFrame.vObjBoxID.resize(ObjIdNew.size());
        mCurrentFrame.vObjCentre3D.resize(ObjIdNew.size());
        mCurrentFrame.vnObjID.resize(ObjIdNew.size());
        mCurrentFrame.vnObjID_line.resize(ObjIdNew_Line.size());
        mCurrentFrame.vnObjInlierID.resize(ObjIdNew.size());
        mCurrentFrame.vnObjInlierID_line.resize(ObjIdNew_Line.size());
        repro_e.resize(ObjIdNew.size(),0.0);
        cv::Mat Last_Twc_gt = Converter::toInvMatrix(mLastFrame.mTcw_gt);
        cv::Mat Curr_Twc_gt = Converter::toInvMatrix(mCurrentFrame.mTcw_gt);
        // main loop
        int dyn_point_counter, dyn_line_counter, inlier_dyn_point_counter, inlier_dyn_line_counter;
        dyn_point_counter = 0;
        dyn_line_counter = 0;
        inlier_dyn_point_counter = 0;
        inlier_dyn_line_counter = 0;
        for (int i = 0; i < ObjIdNew.size(); ++i)
        {
            cout << endl << "Processing Object No.[" << mCurrentFrame.nModLabel[i] << "]:" << endl;
            // Get the ground truth object motion
            cv::Mat L_p, L_c, L_w_p, L_w_c, H_p_c, H_p_c_body;
            bool bCheckGT1 = false, bCheckGT2 = false;
            for (int k = 0; k < mLastFrame.nSemPosi_gt.size(); ++k)
            {
                if (mLastFrame.nSemPosi_gt[k]==mCurrentFrame.nSemPosition[i]){
                    cout << "it is " << mLastFrame.nSemPosi_gt[k] << "!" << endl;
                    if (mTestData==OMD)
                    {
                        L_w_p = mLastFrame.vObjPose_gt[k];
                    }
                    else if (mTestData==KITTI)
                    {
                        L_p = mLastFrame.vObjPose_gt[k];
                        // cout << "what is L_p: " << endl << L_p << endl;
                        L_w_p = Last_Twc_gt*L_p;
                        // cout << "what is L_w_p: " << endl << L_w_p << endl;
                    }
                    bCheckGT1 = true;
                    break;
                }
            }
            for (int k = 0; k < mCurrentFrame.nSemPosi_gt.size(); ++k)
            {
                if (mCurrentFrame.nSemPosi_gt[k]==mCurrentFrame.nSemPosition[i]){
                    cout << "it is " << mCurrentFrame.nSemPosi_gt[k] << "!" << endl;
                    if (mTestData==OMD)
                    {
                        L_w_c = mCurrentFrame.vObjPose_gt[k];
                    }
                    else if (mTestData==KITTI)
                    {
                        L_c = mCurrentFrame.vObjPose_gt[k];
                        // cout << "what is L_c: " << endl << L_c << endl;
                        L_w_c = Curr_Twc_gt*L_c;
                        // cout << "what is L_w_c: " << endl << L_w_c << endl;
                    }
                    mCurrentFrame.vObjBoxID[i] = k;
                    bCheckGT2 = true;
                    break;
                }
            }

            if (!bCheckGT1 || !bCheckGT2)
            {
                cout << "Found a detected object with no ground truth motion! ! !" << endl;
                mCurrentFrame.bObjStat[i] = false;
                mCurrentFrame.vObjMod_gt[i] = cv::Mat::eye(4,4, CV_32F);
                mCurrentFrame.vObjMod[i] = cv::Mat::eye(4,4, CV_32F);
                mCurrentFrame.vObjCentre3D[i] = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
                mCurrentFrame.vObjSpeed_gt[i] = 0.0;
                mCurrentFrame.vnObjInlierID[i] = ObjIdNew[i];
                mCurrentFrame.vnObjInlierID_line[i] = ObjIdNew_Line[i];
                continue;
            }

            cv::Mat L_w_p_inv = Converter::toInvMatrix(L_w_p);
            cv::Mat L_w_c_inv = Converter::toInvMatrix(L_w_c);
            H_p_c = L_w_c*L_w_p_inv;
            H_p_c_body = L_w_p_inv*L_w_c; // for new metric (26 Feb 2020).
            mCurrentFrame.vObjMod_gt[i] = H_p_c_body;
            // mCurrentFrame.vObjCentre3D[i] = L_w_p.rowRange(0,3).col(3);
            mCurrentFrame.vObjPosePre[i] = L_w_p; // for new metric (26 Feb 2020).

            // cout << "ground truth motion of object No. " << mCurrentFrame.nSemPosition[i] << " :" << endl;
            // cout << H_p_c << endl;

            // ***************************************************************************************

            cv::Mat ObjCentre3D_pre = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
            for (int j = 0; j < ObjIdNew[i].size(); ++j)
            {
                // save object centroid in current frame
                cv::Mat x3D_p = mLastFrame.UnprojectStereoObject(ObjIdNew[i][j],0);
                ObjCentre3D_pre = ObjCentre3D_pre + x3D_p;

            }
            ObjCentre3D_pre = ObjCentre3D_pre/ObjIdNew[i].size();
            mCurrentFrame.vObjCentre3D[i] = ObjCentre3D_pre;


            s_3_1 = clock();

            // ******* Get initial model and inlier set using P3P RanSac ********
            std::vector<int> ObjIdTest = ObjIdNew[i];
            mCurrentFrame.vnObjID[i] = ObjIdTest;
            mCurrentFrame.vnObjID_line[i] = ObjIdNew_Line[i];
            std::vector<int> ObjIdTest_in;
            for (int j = 0; j < ObjIdTest.size(); ++j)
            {
                if (ObjIdTest[j]!=-1)
                {
                    dyn_point_counter = dyn_point_counter + 1;
                }
            }
            for (int j = 0; j < ObjIdNew_Line[i].size(); ++j)
            {
                if (ObjIdNew_Line[i][j]!=-1)
                {
                    dyn_line_counter = dyn_line_counter + 1;
                }
            }
            mCurrentFrame.mInitModel = GetInitModelObj(ObjIdTest,ObjIdTest_in,i);
            // cv::Mat H_tmp = Converter::toInvMatrix(mCurrentFrame.mTcw_gt)*mCurrentFrame.mInitModel;
            // cout << "Initial motion estimation: " << endl << H_tmp << endl;
            e_3_1 = clock();

            if (ObjIdTest_in.size()<50)
            {
                cout << "Object Initialization Fail! ! !" << endl;
                mCurrentFrame.bObjStat[i] = false;
                mCurrentFrame.vObjMod_gt[i] = cv::Mat::eye(4,4, CV_32F);
                mCurrentFrame.vObjMod[i] = cv::Mat::eye(4,4, CV_32F);
                mCurrentFrame.vObjCentre3D[i] = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
                mCurrentFrame.vObjSpeed_gt[i] = 0.0;
                mCurrentFrame.vSpeed[i] = cv::Point2f(0.f, 0.f);
                mCurrentFrame.vnObjInlierID[i] = ObjIdTest_in;
                mCurrentFrame.vnObjInlierID_line[i] = ObjIdNew_Line[i];
                continue;
            }

            // cout << "number of pick points: " << ObjIdTest_in.size() << "/" << ObjIdTest.size() << "/" << mCurrentFrame.mvObjKeys.size() << endl;

            // // **** show the picked points ****
            // std::vector<cv::KeyPoint> PickKeys;
            // for (int j = 0; j < ObjIdTest_in.size(); ++j){
            //     // PickKeys.push_back(mCurrentFrame.mvStatKeys[ObjIdTest[j]]);
            //     PickKeys.push_back(mCurrentFrame.mvObjKeys[ObjIdTest_in[j]]);
            // }
            // cv::drawKeypoints(mImGray, PickKeys, mImGray, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
            // cv::imshow("KeyPoints and Grid on Vehicle", mImGray);
            // cv::waitKey(0);

            // // // // image show the matching on each object
            // std::vector<cv::KeyPoint> PreKeys, CurKeys;
            // std::vector<cv::DMatch> TMes;
            // for (int j = 0; j < ObjIdTest.size(); ++j)
            // {
            //     // save key points for visualization
            //     PreKeys.push_back(mLastFrame.mvObjKeys[ObjIdTest[j]]);
            //     CurKeys.push_back(mCurrentFrame.mvObjKeys[ObjIdTest[j]]);
            //     TMes.push_back(cv::DMatch(count,count,0));
            //     count = count + 1;
            // }
            // cout << "count count: " << count << endl;
            // cv::Mat img_matches;
            // drawMatches(mImGrayLast, PreKeys, mImGray, CurKeys,
            //             TMes, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
            //             vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            // cv::resize(img_matches, img_matches, cv::Size(img_matches.cols/1.0, img_matches.rows/1.0));
            // cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
            // cv::imshow("temperal matches", img_matches);
            // cv::waitKey(0);

            // ***************************************************************************************

            s_3_2 = clock();
            // // save object motion and label
            std::vector<int> InlierID;
            std::vector<int> InlierID_Line;
            if (bJoint)
            {
                cv::Mat Obj_X_tmp;
                #if defined(USE_LINE)
                    std::cout << "Using lines for object optimization " << std::endl;
                    Obj_X_tmp = Optimizer::PoseOptimizationFlow2withLines(&mCurrentFrame,&mLastFrame,ObjIdTest_in,ObjIdNew_Line[i],InlierID, InlierID_Line);
                #else
                    Obj_X_tmp = Optimizer::PoseOptimizationFlow2(&mCurrentFrame,&mLastFrame,ObjIdTest_in,InlierID);
                #endif
                mCurrentFrame.vObjMod[i] = Converter::toInvMatrix(mCurrentFrame.mTcw)*Obj_X_tmp;
                
            }
            else
            {
                #if defined(USE_LINE)
                    std::cout << "Using lines for object optimization " << std::endl;
                    mCurrentFrame.vObjMod[i] = Optimizer::PoseOptimizationObjMotWithLines(&mCurrentFrame,&mLastFrame,ObjIdTest_in,ObjIdNew_Line[i],InlierID, InlierID_Line);
                #else
                    mCurrentFrame.vObjMod[i] = Optimizer::PoseOptimizationObjMot(&mCurrentFrame,&mLastFrame,ObjIdTest_in,InlierID);
                #endif
            }
            e_3_2 = clock();
            t_con = t_con + 1;
            obj_mot_time = obj_mot_time + (double)(e_3_1-s_3_1)/CLOCKS_PER_SEC*1000 + (double)(e_3_2-s_3_2)/CLOCKS_PER_SEC*1000;

            inlier_dyn_point_counter = inlier_dyn_point_counter + InlierID.size();
            inlier_dyn_line_counter = inlier_dyn_line_counter + InlierID_Line.size();

            mCurrentFrame.vnObjInlierID[i] = InlierID;
            mCurrentFrame.vnObjInlierID_line[i] = InlierID_Line;
            // cout << "computed motion of object No. " << mCurrentFrame.nSemPosition[i] << " :" << endl;
            // cout << mCurrentFrame.vObjMod[i] << endl;

            // ***********************************************************************************************

            // // ***** get the ground truth object speed here ***** (use version 1 here)
            cv::Mat sp_gt_v, sp_gt_v2;
            sp_gt_v = H_p_c.rowRange(0,3).col(3) - (cv::Mat::eye(3,3,CV_32F)-H_p_c.rowRange(0,3).colRange(0,3))*ObjCentre3D_pre; // L_w_p.rowRange(0,3).col(3) or ObjCentre3D_pre
            sp_gt_v2 = L_w_p.rowRange(0,3).col(3) - L_w_c.rowRange(0,3).col(3);
            float sp_gt_norm = std::sqrt( sp_gt_v.at<float>(0)*sp_gt_v.at<float>(0) + sp_gt_v.at<float>(1)*sp_gt_v.at<float>(1) + sp_gt_v.at<float>(2)*sp_gt_v.at<float>(2) )*36;
            // float sp_gt_norm2 = std::sqrt( sp_gt_v2.at<float>(0)*sp_gt_v2.at<float>(0) + sp_gt_v2.at<float>(1)*sp_gt_v2.at<float>(1) + sp_gt_v2.at<float>(2)*sp_gt_v2.at<float>(2) )*36;
            mCurrentFrame.vObjSpeed_gt[i] = sp_gt_norm;

            // // ***** calculate the estimated object speed *****
            cv::Mat sp_est_v;
            sp_est_v = mCurrentFrame.vObjMod[i].rowRange(0,3).col(3) - (cv::Mat::eye(3,3,CV_32F)-mCurrentFrame.vObjMod[i].rowRange(0,3).colRange(0,3))*ObjCentre3D_pre;
            float sp_est_norm = std::sqrt( sp_est_v.at<float>(0)*sp_est_v.at<float>(0) + sp_est_v.at<float>(1)*sp_est_v.at<float>(1) + sp_est_v.at<float>(2)*sp_est_v.at<float>(2) )*36;

            cout << "estimated and ground truth object speed: " << sp_est_norm << "km/h " << sp_gt_norm << "km/h " << endl;

            mCurrentFrame.vSpeed[i].x = sp_est_norm*36;
            mCurrentFrame.vSpeed[i].y = sp_gt_norm*36;


            // // ************** calculate the relative pose error *****************

            // (1) metric
            // cv::Mat ObjMot_inv = Converter::toInvMatrix(mCurrentFrame.vObjMod[i]);
            // cv::Mat RePoEr = ObjMot_inv*H_p_c;

            // (2) metric
            // cv::Mat L_w_c_est = mCurrentFrame.vObjMod[i]*L_w_p;
            // cv::Mat L_w_c_est_inv = Converter::toInvMatrix(L_w_c_est);
            // cv::Mat RePoEr = L_w_c_est_inv*L_w_c;

            // (3) metric
            // cv::Mat H_p_c_body_est = L_w_p_inv*mCurrentFrame.vObjMod[i]*L_w_p;
            cv::Mat H_p_c_body_est = L_w_p_inv*mCurrentFrame.vObjMod[i]*L_w_p;
            cv::Mat RePoEr = Converter::toInvMatrix(H_p_c_body_est)*H_p_c_body;

            // (4) metric
            // cv::Mat H_p_c_body = L_w_p_inv*L_w_c;
            // cv::Mat H_p_c_body_est_inv = Converter::toInvMatrix(mCurrentFrame.vObjMod[i]);
            // cv::Mat RePoEr = H_p_c_body_est_inv*H_p_c_body;

            float t_rpe = std::sqrt( RePoEr.at<float>(0,3)*RePoEr.at<float>(0,3) + RePoEr.at<float>(1,3)*RePoEr.at<float>(1,3) + RePoEr.at<float>(2,3)*RePoEr.at<float>(2,3) );
            float trace_rpe = 0;
            for (int i = 0; i < 3; ++i)
            {
                if (RePoEr.at<float>(i,i)>1.0)
                     trace_rpe = trace_rpe + 1.0-(RePoEr.at<float>(i,i)-1.0);
                else
                    trace_rpe = trace_rpe + RePoEr.at<float>(i,i);
            }
            float r_rpe = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.1415926;
            cout << "the relative pose error of the object, " << "t: " << t_rpe <<  " R: " << r_rpe << endl;

            // *****************************************************************************
        }

        //write in file number of dynamic points and lines
        // file.open("./statistics/dynamic_point_numbers.txt", std::ios_base::app);
        // file << "Frame " << f_id << " Number of dynamic points: " << dyn_point_counter << std::endl;
        // file.close();
        // file.open("./statistics/dynamic_line_numbers.txt", std::ios_base::app);
        // file << "Frame " << f_id << " Number of dynamic lines: " << dyn_line_counter << std::endl;
        // file.close();
        // file.open("./statistics/inlier_dynamic_point_numbers.txt", std::ios_base::app);
        // file << "Frame " << f_id << " Number of inliers after object motion estimation: " << inlier_dyn_point_counter << std::endl;
        // file.close();
        // file.open("./statistics/inlier_dynamic_line_numbers.txt", std::ios_base::app);
        // file << "Frame " << f_id << " Number of inliers after object motion estimation with lines: " << inlier_dyn_line_counter << std::endl;
        // file.close();

        if (t_con!=0)
        {
            obj_mot_time = obj_mot_time/t_con;
            all_timing[3] = obj_mot_time;
            // cout << "object motion estimation time: " << obj_mot_time << endl;
        }
        else
            all_timing[3] = 0;

        // ****** Renew Current frame information *******

        clock_t s_4, e_4;
        double map_upd_time;
        s_4 = clock();
        RenewFrameInfo(TemperalMatch_subset, TemperalMatch_Line);
        e_4 = clock();
        map_upd_time = (double)(e_4-s_4)/CLOCKS_PER_SEC*1000;
        all_timing[4] = map_upd_time;
        cout << "map updating time: " << map_upd_time/ObjIdNew.size() << endl;

        // **********************************************

        // Save timing analysis to the map
        mpMap->vfAll_time.push_back(all_timing);

        cout << "Assign To Lastframe ......" << endl;

        // // ====== Update from current to last frames ======
        mvKeysLastFrame = mLastFrame.mvStatKeys;  // new added (1st Dec)  mvStatKeys <-> mvKeys
        mvKeysCurrentFrame = mCurrentFrame.mvStatKeys; // new added (12th Sep)
        mvKeysLastFrame_Line = mLastFrame.mvStatKeys_Line;
        mvKeysCurrentFrame_line = mCurrentFrame.mvStatKeys_Line;

        mLastFrame = Frame(mCurrentFrame);  // this is very important!!!
        mLastFrame.mvObjKeys = mCurrentFrame.mvObjKeys;  // new added Nov 19 2019
        mLastFrame.mvObjDepth = mCurrentFrame.mvObjDepth;  // new added Nov 19 2019
        //TODO: Should this here be the tmp of the current frame? I think this one is for the previous frame
        mLastFrame.vSemObjLabel = mCurrentFrame.vSemObjLabel; // new added Nov 19 2019
        
        //delete mLastFrame.mvStatDepth_Line;
        //mLastFrame.mvStatDepth_Line = nullptr;
        mLastFrame.mvStatKeys = mCurrentFrame.mvStatKeysTmp; // new added Jul 30 2019
        mLastFrame.mvStatDepth = mCurrentFrame.mvStatDepthTmp;  // new added Jul 30 2019
        mLastFrame.mvStatKeys_Line = mCurrentFrame.mvStatKeysLineTmp; 
        mLastFrame.mvStatDepth_Line = mCurrentFrame.mvStatDepthLineTmp;

        //TO-DO the above for lines
        
        // // ================================================

        cout << "Assign To Lastframe, Done!" << endl;

        // **********************************************************
        // ********* save some stuffs for graph structure. **********
        // **********************************************************

        cout << "Save Graph Structure ......" << endl;



        // (1) detected static features, corresponding depth and associations
        mpMap->vpFeatSta.push_back(mCurrentFrame.mvStatKeysTmp);
        mpMap->vfDepSta.push_back(mCurrentFrame.mvStatDepthTmp);
        mpMap->vp3DPointSta.push_back(mCurrentFrame.mvStat3DPointTmp);  // (new added Dec 12 2019)
        mpMap->vnAssoSta.push_back(mCurrentFrame.nStaInlierID);         // (new added Nov 14 2019)

        // (2) detected dynamic object features, corresponding depth and associations
        mpMap->vpFeatDyn.push_back(mCurrentFrame.mvObjKeys);           
        mpMap->vfDepDyn.push_back(mCurrentFrame.mvObjDepth);           
        mpMap->vp3DPointDyn.push_back(mCurrentFrame.mvObj3DPoint);     
        mpMap->vnAssoDyn.push_back(mCurrentFrame.nDynInlierID);        
        mpMap->vnFeatLabel.push_back(mCurrentFrame.vObjLabel);         

        // (3) detected static line features, corresponding depth and associations
        mpMap->vpFeatSta_line.push_back(mCurrentFrame.mvStatKeysLineTmp);
        mpMap->vfDepSta_line.push_back(mCurrentFrame.mvStatDepthLineTmp);
        mpMap->vp3DLineSta.push_back(mCurrentFrame.mvStat3DLineTmp);  
        std::vector<cv::Mat> mv3DLineStaPluck = mCurrentFrame.CalculatePlucker(mCurrentFrame.mvStat3DLineTmp);
        mpMap->vp3DLineStaPlucker.push_back(mv3DLineStaPluck);
        mpMap->vnAssoSta_line.push_back(mCurrentFrame.nStaInlierID_line);         // (new added Nov 14 2019)

        // (4) detected dynamic object line features, corresponding depth and associations
        mpMap->vpFeatDyn_line.push_back(mCurrentFrame.mvObjKeys_Line);
        mpMap->vfDepDyn_line.push_back(mCurrentFrame.mvObjDepth_line);
        mpMap->vp3DLineDyn.push_back(mCurrentFrame.mvObj3DLine);
        
        //CREATE 3D PLOT OF THE LINES
        // mWindow = cv::viz::Viz3d("3D Map");
        // mWindow.showWidget("Map Widget", cv::viz::WCoordinateSystem());

        // for (int i = 0; i < mpMap->vp3DLineSta.size(); ++i)
        // {
        //     for (int j=0; j < mpMap->vp3DLineSta[i].size(); ++j)
        //     {
        //         cv::Point3f point1(mpMap->vp3DLineSta[i][j].first.at<float>(0), -mpMap->vp3DLineSta[i][j].first.at<float>(1), mpMap->vp3DLineSta[i][j].first.at<float>(2));
        //         cv::Point3f point2(mpMap->vp3DLineSta[i][j].second.at<float>(0), -mpMap->vp3DLineSta[i][j].second.at<float>(1), mpMap->vp3DLineSta[i][j].second.at<float>(2));
                
        //         cv::viz::WLine line_widget(point1, point2, cv::viz::Color::green());
        //         mWindow.showWidget("line"+std::to_string(i)+std::to_string(j), line_widget);
        //     }
        // }
        // for (int i = 0; i < mCurrentFrame.mvStat3DLineTmp.size(); ++i)
        // {
        //     cv::Point3f point1(mCurrentFrame.mvStat3DLineTmp[i].first.at<float>(0), mCurrentFrame.mvStat3DLineTmp[i].first.at<float>(1), mCurrentFrame.mvStat3DLineTmp[i].first.at<float>(2));
        //     cv::Point3f point2(mCurrentFrame.mvStat3DLineTmp[i].second.at<float>(0), mCurrentFrame.mvStat3DLineTmp[i].second.at<float>(1), mCurrentFrame.mvStat3DLineTmp[i].second.at<float>(2));
            
        //     cv::viz::WLine line_widget(point1, point2, cv::viz::Color::green());
        //     mWindow.showWidget("line"+std::to_string(i), line_widget);
        // }

        // while(!mWindow.wasStopped())
        // {
        //     mWindow.spinOnce(1, true);
        //     int key = cv::waitKey(1);
        //     if(key == 'q' || key == 'Q')  // Press 'q' or 'Q' to continue
        //         break;
        // }

        // for (int i=0; i < mCurrentFrame.mvObj3DLine.size(); ++i)
        // {
        //     std::cout << "mCurrentFrame.mvObj3DLine[i] startpoint: " << mCurrentFrame.mvObj3DLine[i].first << " endpoint " << mCurrentFrame.mvObj3DLine[i].second << std::endl;
        // }
        std::vector<cv::Mat> mv3DLineDynPluck = mCurrentFrame.CalculatePlucker(mCurrentFrame.mvObj3DLine);
        // for (int i=0; i < mCurrentFrame.mvObj3DLine.size(); ++i)
        // {
        //     std::cout << "mCurrentFramePlucker n" << mCurrentFrame.CalculatePlucker(mCurrentFrame.mvObj3DLine)[i] << std::endl;
        // }
        for (int i=0; i < mCurrentFrame.mvObj3DLine.size(); ++i)
        {
            if (cv::countNonZero(mCurrentFrame.CalculatePlucker(mCurrentFrame.mvObj3DLine)[i]) == 0)
            {
                std::cout << "Plucker is zero" << std::endl;
                std::cout << "mCurrentFramePlucker n" << mCurrentFrame.CalculatePlucker(mCurrentFrame.mvObj3DLine)[i] << std::endl;
                std::cout << "mCurrentFrame.mvObj3DLine[i] startpoint: " << mCurrentFrame.mvObj3DLine[i].first << " endpoint " << mCurrentFrame.mvObj3DLine[i].second << std::endl;
                std::cout << "mCurrentFrame.mvObjKeys_Line[i] startpoint: " << mCurrentFrame.mvObjKeys_Line[i].getStartPoint() << " endpoint " << mCurrentFrame.mvObjKeys_Line[i].getEndPoint() << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        mpMap->vp3DLineDynPlucker.push_back(mv3DLineDynPluck);
        mpMap->vnAssoDyn_line.push_back(mCurrentFrame.nDynInlierID_line);
        mpMap->vnFeatLabel_line.push_back(mCurrentFrame.vObjLabel_Line);
        if (f_id==StopFrame || bLocalBatch)
        {
            std::pair<std::vector<std::vector<std::pair<int, int>>>, std::vector<std::vector<std::pair<int, int>>>> TrackletStaAll;
            TrackletStaAll = GetStaticTrack();
            // (3) save static feature tracklets
            mpMap->TrackletSta = TrackletStaAll.first;
            mpMap->TrackletSta_line = TrackletStaAll.second;
            // (4) save dynamic feature tracklets
            std::pair<std::vector<std::vector<std::pair<int, int> > >, std::vector<std::vector<std::pair<int, int> > >> TrackletDynAll;
            TrackletDynAll = GetDynamicTrackNew();
            mpMap->TrackletDyn = TrackletDynAll.first; 
            mpMap->TrackletDyn_line = TrackletDynAll.second;
        }

        // (5) camera pose
        cv::Mat CameraPoseTmp = Converter::toInvMatrix(mCurrentFrame.mTcw);
        mpMap->vmCameraPose.push_back(CameraPoseTmp);
        mpMap->vmCameraPose_RF.push_back(CameraPoseTmp);
        // (6) Rigid motions and label, including camera (label=0) and objects (label>0)
        std::vector<cv::Mat> Mot_Tmp, ObjPose_Tmp;
        std::vector<int> Mot_Lab_Tmp, Sem_Lab_Tmp;
        std::vector<bool> Obj_Stat_Tmp;
        // (6.1) Save Camera Motion and Label
        cv::Mat CameraMotionTmp = Converter::toInvMatrix(mVelocity);
        Mot_Tmp.push_back(CameraMotionTmp);
        ObjPose_Tmp.push_back(CameraMotionTmp);
        Mot_Lab_Tmp.push_back(0);
        Sem_Lab_Tmp.push_back(0);
        Obj_Stat_Tmp.push_back(true);
        // (6.2) Save Object Motions and Label
        for (int i = 0; i < mCurrentFrame.vObjMod.size(); ++i)
        {
            if (!mCurrentFrame.bObjStat[i])
                continue;
            Obj_Stat_Tmp.push_back(mCurrentFrame.bObjStat[i]);
            Mot_Tmp.push_back(mCurrentFrame.vObjMod[i]);
            ObjPose_Tmp.push_back(mCurrentFrame.vObjPosePre[i]);
            Mot_Lab_Tmp.push_back(mCurrentFrame.nModLabel[i]);
            Sem_Lab_Tmp.push_back(mCurrentFrame.nSemPosition[i]);
        }
        // (6.3) Save to The Map
        mpMap->vmRigidMotion.push_back(Mot_Tmp);
        mpMap->vmObjPosePre.push_back(ObjPose_Tmp);
        mpMap->vmRigidMotion_RF.push_back(Mot_Tmp);
        mpMap->vnRMLabel.push_back(Mot_Lab_Tmp);
        mpMap->vnSMLabel.push_back(Sem_Lab_Tmp);
        mpMap->vbObjStat.push_back(Obj_Stat_Tmp);

        // (6.4) Count the tracking times of each unique object
        if (max_id>1)
            mpMap->vnObjTraTime = GetObjTrackTime(mpMap->vnRMLabel,mpMap->vnSMLabel, mpMap->vnSMLabelGT);

        // ---------------------------- Ground Truth --------------------------------

        // (7) Ground Truth Camera Pose
        cv::Mat CameraPoseTmpGT = Converter::toInvMatrix(mCurrentFrame.mTcw_gt);
        mpMap->vmCameraPose_GT.push_back(CameraPoseTmpGT);

        // (8) Ground Truth Rigid Motions
        std::vector<cv::Mat> Mot_Tmp_gt;
        // (8.1) Save Camera Motion
        cv::Mat CameraMotionTmp_gt = mLastFrame.mTcw_gt*Converter::toInvMatrix(mCurrentFrame.mTcw_gt);
        Mot_Tmp_gt.push_back(CameraMotionTmp_gt);
        // (8.2) Save Object Motions
        for (int i = 0; i < mCurrentFrame.vObjMod_gt.size(); ++i)
        {
            if (!mCurrentFrame.bObjStat[i])
                continue;
            Mot_Tmp_gt.push_back(mCurrentFrame.vObjMod_gt[i]);
        }
        // (8.3) Save to The Map
        mpMap->vmRigidMotion_GT.push_back(Mot_Tmp_gt);

        // (9) Ground Truth Camera and Object Speeds
        std::vector<float> Speed_Tmp_gt;
        // (9.1) Save Camera Speed
        Speed_Tmp_gt.push_back(1.0);
        // (9.2) Save Object Motions
        for (int i = 0; i < mCurrentFrame.vObjSpeed_gt.size(); ++i)
        {
            if (!mCurrentFrame.bObjStat[i])
                continue;
            Speed_Tmp_gt.push_back(mCurrentFrame.vObjSpeed_gt[i]);
        }
        // (9.3) Save to The Map
        mpMap->vfAllSpeed_GT.push_back(Speed_Tmp_gt);

        // (10) Computed Camera and Object Speeds
        std::vector<cv::Mat> Centre_Tmp;
        // (10.1) Save Camera Speed
        cv::Mat CameraCentre = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
        Centre_Tmp.push_back(CameraCentre);
        // (10.2) Save Object Motions
        for (int i = 0; i < mCurrentFrame.vObjCentre3D.size(); ++i)
        {
            if (!mCurrentFrame.bObjStat[i])
                continue;
            Centre_Tmp.push_back(mCurrentFrame.vObjCentre3D[i]);
        }
        // (10.3) Save to The Map
        mpMap->vmRigidCentre.push_back(Centre_Tmp);

        cout << "Save Graph Structure, Done!" << endl;
    }

    // =================================================================================================
    // ============== Partial batch optimize on all the measurements (local optimization) ==============
    // =================================================================================================

    bLocalBatch = true;
    if ( (f_id-nOVERLAP_SIZE+1)%(nWINDOW_SIZE-nOVERLAP_SIZE)==0 && f_id>=nWINDOW_SIZE-1 && bLocalBatch)
    {
        cout << "-------------------------------------------" << endl;
        cout << "! ! ! ! Partial Batch Optimization ! ! ! ! " << endl;
        cout << "-------------------------------------------" << endl;
        clock_t s_5, e_5;
        double loc_ba_time;
        s_5 = clock();
        // Get Partial Batch Optimization
        Optimizer::PartialBatchOptimizationWithLines(mpMap,mK,nWINDOW_SIZE);
        //Optimizer::PartialBatchOptimization(mpMap,mK,nWINDOW_SIZE);

        e_5 = clock();
        loc_ba_time = (double)(e_5-s_5)/CLOCKS_PER_SEC*1000;
        mpMap->fLBA_time.push_back(loc_ba_time);
        // cout << "local optimization time: " << loc_ba_time << endl;
    }

    // =================================================================================================
    // ============== Full batch optimize on all the measurements (global optimization) ================
    // =================================================================================================

    bGlobalBatch = true;
    if (f_id==StopFrame) // bFrame2Frame f_id>=2
    {
        // Metric Error BEFORE Optimization
        GetMetricError(mpMap->vmCameraPose,mpMap->vmRigidMotion, mpMap->vmObjPosePre,
                       mpMap->vmCameraPose_GT,mpMap->vmRigidMotion_GT, mpMap->vbObjStat);
        // GetVelocityError(mpMap->vmRigidMotion, mpMap->vp3DPointDyn, mpMap->vnFeatLabel,
        //                  mpMap->vnRMLabel, mpMap->vfAllSpeed_GT, mpMap->vnAssoDyn, mpMap->vbObjStat);

        //export mpMap->vp3DLineSta
        // std::ofstream output_file_;
        
        // output_file_.open("static_lines_in_map.txt", std::ios::app);
        // for (int i = 0; i < mpMap->vp3DLineSta.size(); ++i)
        // {
        //     for (int j = 0; j < mpMap->vp3DLineSta[i].size(); ++j)
        //     {
        //         output_file_ << "Line " << j << std::endl;
        //         output_file_ << "Start Point: " << mpMap->vp3DLineSta[i][j].first.at<float>(0) << " " << mpMap->vp3DLineSta[i][j].first.at<float>(1) << " " << mpMap->vp3DLineSta[i][j].first.at<float>(2) << std::endl;
        //         output_file_ << "End Point: " << mpMap->vp3DLineSta[i][j].second.at<float>(0) << " " << mpMap->vp3DLineSta[i][j].second.at<float>(1) << " " << mpMap->vp3DLineSta[i][j].second.at<float>(2) << std::endl;
        //     }
        // }
        // output_file_.close();

        //         output_file_.open("static_points_in_map.txt", std::ios::app);
        // for (int i = 0; i < mpMap->vp3DPointSta.size(); ++i)
        // {
        //     for (int j = 0; j < mpMap->vp3DPointSta[i].size(); ++j)
        //     {
        //         output_file_ << "Point " << j << std::endl;
        //         output_file_ << "Point: " << mpMap->vp3DPointSta[i][j].at<float>(0) << " " << mpMap->vp3DPointSta[i][j].at<float>(1) << " " << mpMap->vp3DPointSta[i][j].at<float>(2) << std::endl;
        //     }
        // }
        // output_file_.close();
        

        //write to a file with this format:
        //tx, ty, tz, qx, qy, qz, qw
        //of the camera poses
        //std::ofstream output_file_;
        //output_file_.open("camera_poses_in_map.txt", std::ios::out);
        // for (int i = 0; i < mpMap->vmCameraPose.size(); ++i)
        // {
        //     cv::Mat pose = mpMap->vmCameraPose[i];
        //     //calculate inverse
        //     //pose = Converter::toInvMatrix(pose);
        //     cv::Mat R = pose.rowRange(0,3).colRange(0,3);
        //     cv::Mat t = pose.rowRange(0,3).col(3);
        //     std::vector<float> q = Converter::toQuaternion(R);
        //     output_file_ << t.at<float>(0) << " " << -t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
        // }
        //output_file_.close();

        if (bGlobalBatch && mTestData==KITTI)
        {
            // Get Full Batch Optimization
            Optimizer::FullBatchOptimizationWithLines(mpMap,mK);
            
            
            // Metric Error AFTER Optimization
            GetMetricError(mpMap->vmCameraPose_RF,mpMap->vmRigidMotion_RF, mpMap->vmObjPosePre,
                           mpMap->vmCameraPose_GT,mpMap->vmRigidMotion_GT, mpMap->vbObjStat);
            // GetVelocityError(mpMap->vmRigidMotion_RF, mpMap->vp3DPointDyn, mpMap->vnFeatLabel,
            //                  mpMap->vnRMLabel, mpMap->vfAllSpeed_GT, mpMap->vnAssoDyn, mpMap->vbObjStat);
        }
    }

    mState = OK;
}


void Tracking::Initialization()
{
    cout << "Initialization ......" << endl;

    // initialize the 3d points
    {
        //TO-DO for lines
        // static
        std::vector<cv::Mat> mv3DPointTmp;
        for (int i = 0; i < mCurrentFrame.mvStatKeysTmp.size(); ++i)
        {
            mv3DPointTmp.push_back(Optimizer::Get3DinCamera(mCurrentFrame.mvStatKeysTmp[i], mCurrentFrame.mvStatDepthTmp[i], mK));
        }
        std::vector<std::pair<cv::Mat, cv::Mat>> mv3DLineTmp;
        for (int i = 0; i < mCurrentFrame.mvStatKeysLineTmp.size(); ++i)
        {
            mv3DLineTmp.push_back(Optimizer::Get3DinCamera_line(mCurrentFrame.mvStatKeysLineTmp[i], mCurrentFrame.mvStatDepthLineTmp[i], mK));
        }
        mCurrentFrame.mvStat3DPointTmp = mv3DPointTmp;
        mCurrentFrame.mvStat3DLineTmp = mv3DLineTmp;
        // dynamic
        std::vector<cv::Mat> mvObj3DPointTmp;
        for (int i = 0; i < mCurrentFrame.mvObjKeys.size(); ++i)
        {
            mvObj3DPointTmp.push_back(Optimizer::Get3DinCamera(mCurrentFrame.mvObjKeys[i], mCurrentFrame.mvObjDepth[i], mK));
        }
        std::vector<std::pair<cv::Mat, cv::Mat>> mvObj3DLineTmp;
        for (int i = 0; i < mCurrentFrame.mvObjKeys_Line.size(); ++i)
        {
            std::cout << "i is " << i << std::endl;
            std::cout << "mCurrentFrame.mvObjDepth_line.size() " << mCurrentFrame.mvObjDepth_line.size() << std::endl;
            std::cout << "mCurrentFrame.mvObjKeys_Line.size() " << mCurrentFrame.mvObjKeys_Line.size() << std::endl;
            mvObj3DLineTmp.push_back(Optimizer::Get3DinCamera_line(mCurrentFrame.mvObjKeys_Line[i], mCurrentFrame.mvObjDepth_line[i], mK));
        }
        mCurrentFrame.mvObj3DPoint = mvObj3DPointTmp;
        mCurrentFrame.mvObj3DLine = mvObj3DLineTmp;
        // cout << "see the size 1: " << mCurrentFrame.mvStatKeysTmp.size() << " " << mCurrentFrame.mvSift3DPoint.size() << endl;
        // cout << "see the size 2: " << mCurrentFrame.mvObjKeys.size() << " " << mCurrentFrame.mvObj3DPoint.size() << endl;
    }

    // (1) save detected static features and corresponding depth
    mpMap->vpFeatSta.push_back(mCurrentFrame.mvStatKeysTmp);  // modified Nov 14 2019
    mpMap->vfDepSta.push_back(mCurrentFrame.mvStatDepthTmp);  // modified Nov 14 2019
    mpMap->vp3DPointSta.push_back(mCurrentFrame.mvStat3DPointTmp);  // modified Dec 17 2019

    //for lines
    mpMap->vpFeatSta_line.push_back(mCurrentFrame.mvStatKeysLineTmp);
    mpMap->vfDepSta_line.push_back(mCurrentFrame.mvStatDepthLineTmp);
    mpMap->vp3DLineSta.push_back(mCurrentFrame.mvStat3DLineTmp);
    //Push the plucker representation
    std::vector<cv::Mat> mv3DLineStaPluck = mCurrentFrame.CalculatePlucker(mCurrentFrame.mvStat3DLineTmp);
    mpMap->vp3DLineStaPlucker.push_back(mv3DLineStaPluck);
    // (2) save detected dynamic object features and corresponding depth
    mpMap->vpFeatDyn.push_back(mCurrentFrame.mvObjKeys);  // modified Nov 19 2019
    mpMap->vfDepDyn.push_back(mCurrentFrame.mvObjDepth);  // modified Nov 19 2019
    mpMap->vp3DPointDyn.push_back(mCurrentFrame.mvObj3DPoint);  // modified Dec 17 2019

    //for lines // TODO check if it needs the tmp one below No it does not because there is no previous image (it doesn't enter at the point where the current frame becomes the correspondences of the last)
    mpMap->vpFeatDyn_line.push_back(mCurrentFrame.mvObjKeys_Line);
    mpMap->vfDepDyn_line.push_back(mCurrentFrame.mvObjDepth_line);
    mpMap->vp3DLineDyn.push_back(mCurrentFrame.mvObj3DLine);
    std::vector<cv::Mat> mv3DLineDynPluck = mCurrentFrame.CalculatePlucker(mCurrentFrame.mvObj3DLine);
    mpMap->vp3DLineDynPlucker.push_back(mv3DLineDynPluck);
    // (3) save camera pose
    mpMap->vmCameraPose.push_back(cv::Mat::eye(4,4,CV_32F));
    mpMap->vmCameraPose_RF.push_back(cv::Mat::eye(4,4,CV_32F));
    mpMap->vmCameraPose_GT.push_back(cv::Mat::eye(4,4,CV_32F));

    // cout << "mCurrentFrame.N: " << mCurrentFrame.N << endl;

    // Set Frame pose to the origin
    mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));  // +++++  new added +++++
    mCurrentFrame.mTcw_gt = cv::Mat::eye(4,4,CV_32F);
    // mCurrentFrame.mTcw_gt = Converter::toInvMatrix(mOriginInv)*mCurrentFrame.mTcw_gt;
    // cout << "mTcw_gt: " << mCurrentFrame.mTcw_gt << endl;
    // bFirstFrame = false;
    // cout << "current pose: " << endl << mCurrentFrame.mTcw_gt << endl;
    // cout << "current pose inverse: " << endl << mOriginInv << endl;

    mLastFrame = Frame(mCurrentFrame);  //  important !!!
    mLastFrame.mvObjKeys = mCurrentFrame.mvObjKeys; // new added Jul 30 2019
    mLastFrame.mvObjDepth = mCurrentFrame.mvObjDepth;  // new added Jul 30 2019
    mLastFrame.vSemObjLabel = mCurrentFrame.vSemObjLabel; // new added Aug 2 2019

    mLastFrame.mvStatKeys = mCurrentFrame.mvStatKeysTmp; // new added Jul 30 2019
    mLastFrame.mvStatDepth = mCurrentFrame.mvStatDepthTmp;  // new added Jul 30 2019
    mLastFrame.N_s = mCurrentFrame.N_s_tmp; // new added Nov 14 2019
    mvKeysLastFrame = mLastFrame.mvStatKeys; // +++ new added +++

    //Delete the current last frame allocations and replace with the new frame
    //delete mLastFrame.mvStatDepth_Line;
    //mLastFrame.mvStatDepth_Line = nullptr;
    mLastFrame.mvStatDepth_Line = mCurrentFrame.mvStatDepthLineTmp;
    mLastFrame.N_sta_l= mCurrentFrame.N_s_line_tmp;
    mLastFrame.mvStatKeys_Line = mCurrentFrame.mvStatKeysLineTmp;
    mvKeysLastFrame_Line = mLastFrame.mvStatKeys_Line;
    mState=OK;

    cout << "Initialization, Done!" << endl;
}

void Tracking::GetSceneFlowObj()
{
    // // Threshold // //
    // int max_dist = 90, max_lat = 30;
    // double fps = 10, max_velocity_ms = 40;
    // double max_depth = 30;

    // Initialization
    int N = mCurrentFrame.mvObjKeys.size();
    mCurrentFrame.vFlow_3d.resize(N);
    // mCurrentFrame.vFlow_2d.resize(N);

    std::vector<Eigen::Vector3d> pts_p3d(N,Eigen::Vector3d(-1,-1,-1)), pts_vel(N,Eigen::Vector3d(-1,-1,-1));

    const cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = mCurrentFrame.mTcw.rowRange(0,3).col(3);

    // Main loop
    for (int i = 0; i < N; ++i)
    {
        // // filter
        // if(mCurrentFrame.mvObjDepth[i]>max_depth  || mLastFrame.mvObjDepth[i]>max_depth)
        // {
        //     mCurrentFrame.vObjLabel[i]=-1;
        //     continue;
        // }
        if (mCurrentFrame.vSemObjLabel[i]<=0 || mLastFrame.vSemObjLabel[i]<=0)
        {
            mCurrentFrame.vObjLabel[i]=-1;
            continue;
        }

        // get the 3d flow
        cv::Mat x3D_p = mLastFrame.UnprojectStereoObject(i,0);
        cv::Mat x3D_c = mCurrentFrame.UnprojectStereoObject(i,0);

        pts_p3d[i] << x3D_p.at<float>(0), x3D_p.at<float>(1), x3D_p.at<float>(2);

        // cout << "3d points: " << x3D_p << " " << x3D_c << endl;

        cv::Point3f flow3d;
        flow3d.x = x3D_c.at<float>(0) - x3D_p.at<float>(0);
        flow3d.y = x3D_c.at<float>(1) - x3D_p.at<float>(1);
        flow3d.z = x3D_c.at<float>(2) - x3D_p.at<float>(2);

        pts_vel[i] << flow3d.x, flow3d.y, flow3d.z;

        // cout << "3d points: " << mCurrentFrame.vFlow_3d[i] << endl;

        // // threshold the velocity
        // if(cv::norm(flow3d)*fps > max_velocity_ms)
        // {
        //     mCurrentFrame.vObjLabel[i]=-1;
        //     continue;
        // }

        mCurrentFrame.vFlow_3d[i] = flow3d;

        // // get the 2D re-projection error vector
        // // (1) transfer 3d from world to current frame.
        // cv::Mat x3D_pc = Rcw*x3D_p+tcw;
        // // (2) project 3d into current image plane
        // float xc = x3D_pc.at<float>(0);
        // float yc = x3D_pc.at<float>(1);
        // float invzc = 1.0/x3D_pc.at<float>(2);
        // float u = mCurrentFrame.fx*xc*invzc+mCurrentFrame.cx;
        // float v = mCurrentFrame.fy*yc*invzc+mCurrentFrame.cy;

        // mCurrentFrame.vFlow_2d[i].x = mCurrentFrame.mvObjKeys[i].pt.x - u;
        // mCurrentFrame.vFlow_2d[i].y = mCurrentFrame.mvObjKeys[i].pt.y - v;

        // // cout << "2d errors: " << mCurrentFrame.vFlow_2d[i] << endl;

    }

    // // // ===== show scene flow from bird eye view =====
    // cv::Mat img_sparse_flow_3d;
    // BirdEyeVizProperties viz_props;
    // viz_props.birdeye_scale_factor_ = 20.0;
    // viz_props.birdeye_left_plane_ = -15.0;
    // viz_props.birdeye_right_plane_ = 15.0;
    // viz_props.birdeye_far_plane_ = 30.0;

    // Tracking::DrawSparseFlowBirdeye(pts_p3d, pts_vel, Converter::toInvMatrix(mLastFrame.mTcw), viz_props, img_sparse_flow_3d);
    // cv::imshow("SparseFlowBirdeye", img_sparse_flow_3d*255);
    // cv::waitKey(0);
}

std::pair<std::vector<std::vector<int> >, std::vector<std::vector<int>>> Tracking::DynObjTracking()
{
    clock_t s_2, e_2;
    double obj_tra_time;
    s_2 = clock();

    //Printing the contents of vSemObjLabel
    // for (int i = 0; i < mCurrentFrame.vSemObjLabel.size(); ++i)
    //     cout << mCurrentFrame.vSemObjLabel[i] << " ";
    // cout << endl;
    // for (int i = 0; i < mCurrentFrame.vObjLabel.size(); ++i)
    //     cout << mCurrentFrame.vObjLabel[i] << " ";
    // cout << endl;
    // Find the unique labels in semantic label
    auto UniLab = mCurrentFrame.vSemObjLabel;
    //for lines
    auto UniLab_Line = mCurrentFrame.vSemObjLabel_Line;
    std::sort(UniLab.begin(), UniLab.end());
    UniLab.erase(std::unique( UniLab.begin(), UniLab.end() ), UniLab.end() );

    std::sort(UniLab_Line.begin(), UniLab_Line.end());
    UniLab_Line.erase(std::unique( UniLab_Line.begin(), UniLab_Line.end() ), UniLab_Line.end() );

    cout << "Unique Semantic Label: ";
    for (int i = 0; i < UniLab.size(); ++i)
        cout  << UniLab[i] << " ";
    cout << endl;

    cout << "Unique Semantic Label for lines: ";
    for (int i = 0; i < UniLab_Line.size(); ++i)
        cout  << UniLab_Line[i] << " ";
    cout << endl;
    // Collect the predicted labels and semantic labels in vector
    //std::vector<std::pair<std::vector<int>, std::vector<int>>> Posi(max(UniLab.size(), UniLab_Line.size()));

    std::vector<std::pair<std::vector<int>, std::vector<int>>> Posi(UniLab.size());

    for (int k = 0; k < UniLab.size(); ++k) 
    {
        int current_lab = UniLab[k];
        int current_lab_line;
        auto it = std::find(UniLab_Line.begin(), UniLab_Line.end(), current_lab);
        current_lab_line = *it;
        for (int i = 0; i < mCurrentFrame.vSemObjLabel.size(); ++i)
        {
            if (mCurrentFrame.vObjLabel[i]==-1)
                continue;
            if (mCurrentFrame.vSemObjLabel[i] == UniLab[k]){
                Posi[k].first.push_back(i);
            }
        }
        if (it != UniLab_Line.end())
        {
            for (int i=0; i < mCurrentFrame.vSemObjLabel_Line.size(); ++i)
            {
                if (mCurrentFrame.vObjLabel_Line[i]==-1)
                    continue;
                if (mCurrentFrame.vSemObjLabel_Line[i] == UniLab[k]){
                    Posi[k].second.push_back(i);
                }	
            }
        }
    }

    //indices to loop through UniLab and UniLab_Line
    // int k = 0, l = 0;

    // while (k < UniLab.size() && l < UniLab_Line.size())
    // {   
    //     //check if UniLab[k] == UniLab_Line[l]
    //     if (UniLab[k] == UniLab_Line[l])
    //     {
    //         std::cout << "UniLab is the same with UniLab_Line " << UniLab[k] << std::endl;
    //         // for (int i = 0; i < mCurrentFrame.vSemObjLabel.size(); ++i)
    //         //     cout << mCurrentFrame.vSemObjLabel[i] << " ";
    //         // cout << endl;
    //         for (int i = 0; i < mCurrentFrame.vSemObjLabel.size(); ++i)
    //         {
    //             // skip outliers
    //             if (mCurrentFrame.vObjLabel[i]==-1)
    //             {
    //                 std::cout << "discarding" << std::endl;
    //                 continue;
    //             }
    //             //std::cout << "i " << i << std::endl; 
    //             //std::cout << "mCurrentFrame.vSemObjLabel[i] " << mCurrentFrame.vSemObjLabel[i] << std::endl;
    //             //std::cout << "UniLab[k] " << UniLab[k] << std::endl;
    //             // save object label
    //             if(mCurrentFrame.vSemObjLabel[i]==UniLab[k]){
    //                 Posi[max(k,l)].first.push_back(i);
    //             }
    //         }
    //         for (int i = 0; i < mCurrentFrame.vSemObjLabel_Line.size(); ++i)
    //         {
    //             // skip outliers
    //             if (mCurrentFrame.vObjLabel_Line[i]==-1)
    //                 continue;

    //             // save object label
    //             if(mCurrentFrame.vSemObjLabel_Line[i]==UniLab_Line[l]){
    //                 Posi[max(k,l)].second.push_back(i);
    //             }
    //         }
    //         k++;
    //         l++;
    //     }
    //     else if (UniLab[k] < UniLab_Line[l])
    //     {
    //         for (int i = 0; i < mCurrentFrame.vSemObjLabel.size(); ++i)
    //         {
    //             // skip outliers
    //             if (mCurrentFrame.vObjLabel[i]==-1)
    //                 continue;

    //             // save object label
    //             if(mCurrentFrame.vSemObjLabel[i]==UniLab[k]){
    //                 Posi[max(k, l)].first.push_back(i);
    //             }
    //         }
    //         k++;
    //     }
    //     else
    //     {
    //         for (int i = 0; i < mCurrentFrame.vSemObjLabel_Line.size(); ++i)
    //         {
    //             // skip outliers
    //             if (mCurrentFrame.vObjLabel_Line[i]==-1)
    //                 continue;

    //             // save object label
    //             if(mCurrentFrame.vSemObjLabel_Line[i]==UniLab_Line[l]){
    //                 Posi[max(k, l)].second.push_back(i);
    //             }
    //         }
    //         l++;
    //     }

    // }

    // int idx = max(k, l);

    // //for any remaining 
    // if (k < UniLab.size())
    // {
    //     for (int i = k; i < UniLab.size(); ++i)
    //     {
    //         for (int j = 0; j < mCurrentFrame.vSemObjLabel.size(); ++j)
    //         {
    //             // skip outliers
    //             if (mCurrentFrame.vObjLabel[j]==-1)
    //                 continue;

    //             // save object label
    //             if(mCurrentFrame.vSemObjLabel[j]==UniLab[i]){
    //                 Posi[idx].first.push_back(j);
    //             }
    //         }
    //         idx++;
    //     }
    // }
    // else if (l < UniLab_Line.size())
    // {
    //     for (int i = l; i < UniLab_Line.size(); ++i)
    //     {
    //         for (int j = 0; j < mCurrentFrame.vSemObjLabel_Line.size(); ++j)
    //         {
    //             // skip outliers
    //             if (mCurrentFrame.vObjLabel_Line[j]==-1)
    //                 continue;

    //             // save object label
    //             if(mCurrentFrame.vSemObjLabel_Line[j]==UniLab_Line[i]){
    //                 Posi[idx].second.push_back(j);
    //             }
    //         }
    //         idx++;
    //     }
    // }

    // std::cout << "Posi size: " << Posi.size() << std::endl;
    // std::cout << "Posi[0] size: " << Posi[0].first.size() << std::endl;
    // std::cout << "Posi[0] second size: " << Posi[0].second.size() << std::endl;
    // std::cout << "Posi[1] size: " << Posi[1].first.size() << std::endl;
    // std::cout << "Posi[1] second size: " << Posi[1].second.size() << std::endl;
    // for (int i = 0; i < mCurrentFrame.vSemObjLabel.size(); ++i)
    // {
    //     // skip outliers
    //     if (mCurrentFrame.vObjLabel[i]==-1)
    //         continue;

    //     // save object label
    //     for (int j = 0; j < UniLab.size(); ++j)
    //     {
    //         if(mCurrentFrame.vSemObjLabel[i]==UniLab[j]){
    //             Posi[j].push_back(i);
    //             break;
    //         }
    //     }
    // }
    // std::cout << "Test 1" << std::endl;
    // std::vector<std::vector<int> > Posi_Line(UniLab.size());
    // for (int i = 0; i < mCurrentFrame.vSemObjLabel_Line.size(); ++i)
    // {
    //     // skip outliers
    //     if (mCurrentFrame.vObjLabel_Line[i]==-1)
    //         continue;

    //     // save object label
    //     for (int j = 0; j < UniLab_Line.size(); ++j)
    //     {
    //         if(mCurrentFrame.vSemObjLabel_Line[i]==UniLab_Line[j]){
    //             Posi_Line[j].push_back(i);
    //             break;
    //         }
    //     }
    // }
    //std::cout << "Test 2" << std::endl;
    // // Save objects only from Posi() -> ObjId()
    std::vector<std::vector<int> > ObjId;
    std::vector<std::vector<int> > ObjId_Line;
    std::vector<int> sem_posi; // semantic label position for the objects
    std::vector<int> sem_posi_line; // semantic label position for the objects
    int shrin_thr_row=0, shrin_thr_col=0;
    if (mTestData==KITTI)
    {
        shrin_thr_row = 25;
        shrin_thr_col = 50;
    }

    //check if unilab and unilab_line are the same. If they are not print a message
    // if (UniLab.size() != UniLab_Line.size())
    // {
    //     cout << "UniLab and UniLab_Line are not the same size. This is a problem" << endl;
    // }
    
    // int ind1(0), ind2(0);
    // while(ind1 < Posi.size() || ind2 < Posi_Line.size())
    // {
    //     bool flag1 = false;
    //     bool flag2 = false;
    //     if (ind1 < Posi.size())
    //         flag1 = true;
    //     if (ind2 < Posi_Line.size())
    //         flag2 =true;
    //     if (flag1 == true && flag2 == true)
    //     {
    //         if (UniLab[ind1] == UniLab_Line[ind2]){
    //             // shrink the image to get rid of object parts on the boundary
    //             float count = 0, count_thres=0.5;
    //             for (int j = 0; j < Posi[ind1].size(); ++j)
    //             {
    //                 const float u = mCurrentFrame.mvObjKeys[Posi[ind1][j]].pt.x;
    //                 const float v = mCurrentFrame.mvObjKeys[Posi[ind1][j]].pt.y;
    //                 if ( v<shrin_thr_row || v>(mImGray.rows-shrin_thr_row) || u<shrin_thr_col || u>(mImGray.cols-shrin_thr_col) )
    //                     count = count + 1;
    //             }
    //             for (int j = 0; j < Posi_Line[ind2].size(); ++j) 
    //             {
    //                 const float u_start = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].startPointX;
    //                 const float v_start = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].startPointY;
    //                 const float u_end = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].endPointX;
    //                 const float v_end = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].endPointY;
    //                 if ( v_start<shrin_thr_row || v_start>(mImGray.rows-shrin_thr_row) || u_start<shrin_thr_col || u_start>(mImGray.cols-shrin_thr_col) || u_end<shrin_thr_col || u_end>(mImGray.cols-shrin_thr_col) || v_end<shrin_thr_row || v_end>(mImGray.rows-shrin_thr_row)) {
    //                     count = count + 1;
    //                 }
    //             }
    //             if (count/(Posi[ind1].size() + Posi_Line[ind2].size())>count_thres)
    //             {
    //                 // cout << "Most part of this object is on the image boundary......" << endl;
    //                 for (int k = 0; k < Posi[ind1].size(); ++k)
    //                 {
    //                     mCurrentFrame.vObjLabel[Posi[ind1][k]] = -1;
    //                     mCurrentFrame.vObjLabel_Line[Posi_Line[ind2][k]] = -1;
    //                 }
    //                 continue;
    //             }
    //             else
    //             {
    //                 ObjId.push_back(Posi[ind1]);
    //                 sem_posi.push_back(UniLab[ind1]);
    //                 ObjId_Line.push_back(Posi_Line[ind2]);
    //                 sem_posi_line.push_back(UniLab_Line[ind2]);
    //             }
    //             ind1++;
    //             ind2++;
    //         }
    //         else if (UniLab[ind1] < UniLab_Line[ind2])
    //         {
    //             // shrink the image to get rid of object parts on the boundary
    //             float count = 0, count_thres=0.5;
    //             for (int j = 0; j < Posi[ind1].size(); ++j)
    //             {
    //                 const float u = mCurrentFrame.mvObjKeys[Posi[ind1][j]].pt.x;
    //                 const float v = mCurrentFrame.mvObjKeys[Posi[ind1][j]].pt.y;
    //                 if ( v<shrin_thr_row || v>(mImGray.rows-shrin_thr_row) || u<shrin_thr_col || u>(mImGray.cols-shrin_thr_col) )
    //                     count = count + 1;
    //             }
    //             if (count/Posi[ind1].size()>count_thres)
    //             {
    //                 // cout << "Most part of this object is on the image boundary......" << endl;
    //                 for (int k = 0; k < Posi[ind1].size(); ++k)
    //                     mCurrentFrame.vObjLabel[Posi[ind1][k]] = -1;
    //                 continue;
    //             }
    //             else
    //             {
    //                 ObjId.push_back(Posi[ind1]);
    //                 sem_posi.push_back(UniLab[ind1]);
    //             }
    //             ind1++;
    //         }
    //         else 
    //         {
    //             float count = 0, count_thres=0.5;

    //             for (int j = 0; j < Posi_Line[ind2].size(); ++j) 
    //             {
    //             const float u_start = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].startPointX;
    //             const float v_start = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].startPointY;
    //             const float u_end = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].endPointX;
    //             const float v_end = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].endPointY;
    //             if ( v_start<shrin_thr_row || v_start>(mImGray.rows-shrin_thr_row) || u_start<shrin_thr_col || u_start>(mImGray.cols-shrin_thr_col) || u_end<shrin_thr_col || u_end>(mImGray.cols-shrin_thr_col) || v_end<shrin_thr_row || v_end>(mImGray.rows-shrin_thr_row)) {
    //                 count = count + 1;
    //             }
    //             }
    //             if (count/Posi_Line[ind2].size()>count_thres)
    //             {
    //                 // cout << "Most part of this object is on the image boundary......" << endl;
    //                 for (int k = 0; k < Posi_Line[ind2].size(); ++k)
    //                     mCurrentFrame.vObjLabel_Line[Posi_Line[ind2][k]] = -1;
    //                 continue;
                    
    //             }
    //             else
    //             {
    //                 ObjId_Line.push_back(Posi_Line[ind2]);
    //                 sem_posi_line.push_back(UniLab_Line[ind2]);
    //             }
    //             ind2++;
    //         }
    //     }
    //     else if (flag1 == true && flag2 == false)
    //     {
    //         // shrink the image to get rid of object parts on the boundary
    //         float count = 0, count_thres=0.5;
    //         for (int j = 0; j < Posi[ind1].size(); ++j)
    //         {
    //             const float u = mCurrentFrame.mvObjKeys[Posi[ind1][j]].pt.x;
    //             const float v = mCurrentFrame.mvObjKeys[Posi[ind1][j]].pt.y;
    //             if ( v<shrin_thr_row || v>(mImGray.rows-shrin_thr_row) || u<shrin_thr_col || u>(mImGray.cols-shrin_thr_col) )
    //                 count = count + 1;
    //         }
    //         if (count/Posi[ind1].size()>count_thres)
    //         {
    //             // cout << "Most part of this object is on the image boundary......" << endl;
    //             for (int k = 0; k < Posi[ind1].size(); ++k)
    //                 mCurrentFrame.vObjLabel[Posi[ind1][k]] = -1;
    //             continue;
    //         }
    //         else
    //         {
    //             ObjId.push_back(Posi[ind1]);
    //             sem_posi.push_back(UniLab[ind1]);
    //         }
    //         ind1++;
    //     }
    //     else if (flag1 == false && flag2 == true)
    //     {
    //         float count = 0, count_thres=0.5;

    //         for (int j = 0; j < Posi_Line[ind2].size(); ++j) 
    //         {
    //         const float u_start = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].startPointX;
    //         const float v_start = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].startPointY;
    //         const float u_end = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].endPointX;
    //         const float v_end = mCurrentFrame.mvObjKeys_Line[Posi_Line[ind2][j]].endPointY;
    //         if ( v_start<shrin_thr_row || v_start>(mImGray.rows-shrin_thr_row) || u_start<shrin_thr_col || u_start>(mImGray.cols-shrin_thr_col) || u_end<shrin_thr_col || u_end>(mImGray.cols-shrin_thr_col) || v_end<shrin_thr_row || v_end>(mImGray.rows-shrin_thr_row)) {
    //             count = count + 1;
    //         }
    //         }
    //         if (count/Posi_Line[ind2].size()>count_thres)
    //         {
    //             // cout << "Most part of this object is on the image boundary......" << endl;
    //             for (int k = 0; k < Posi_Line[ind2].size(); ++k)
    //                 mCurrentFrame.vObjLabel_Line[Posi_Line[ind2][k]] = -1;
    //             continue;
    //         }
    //         else
    //         {
    //             ObjId_Line.push_back(Posi_Line[ind2]);
    //             sem_posi_line.push_back(UniLab_Line[ind2]);
    //         }
    //         ind2++;
    //     }

    // }
    


    for (int i = 0; i < Posi.size(); ++i)
    {
        // shrink the image to get rid of object parts on the boundary
        float count = 0, count_thres=0.5;
        for (int j = 0; j < Posi[i].first.size(); ++j)
        {
            const float u = mCurrentFrame.mvObjKeys[Posi[i].first[j]].pt.x;
            const float v = mCurrentFrame.mvObjKeys[Posi[i].first[j]].pt.y;
            if ( v<shrin_thr_row || v>(mImGray.rows-shrin_thr_row) || u<shrin_thr_col || u>(mImGray.cols-shrin_thr_col) )
                count = count + 1;
        }

        //for lines
        //std::cout << "i = " << i << std::endl;

        for (int j = 0; j < Posi[i].second.size(); ++j) 
        {
            const float u_start = mCurrentFrame.mvObjKeys_Line[Posi[i].second[j]].startPointX;
            const float v_start = mCurrentFrame.mvObjKeys_Line[Posi[i].second[j]].startPointY;
            const float u_end = mCurrentFrame.mvObjKeys_Line[Posi[i].second[j]].endPointX;
            const float v_end = mCurrentFrame.mvObjKeys_Line[Posi[i].second[j]].endPointY;
            if ( v_start<shrin_thr_row || v_start>(mImGray.rows-shrin_thr_row) || u_start<shrin_thr_col || u_start>(mImGray.cols-shrin_thr_col) || u_end<shrin_thr_col || u_end>(mImGray.cols-shrin_thr_col) || v_end<shrin_thr_row || v_end>(mImGray.rows-shrin_thr_row)) {
                count = count + 1;
            }
        }
        //std::cout << "Test3 " << std::endl;
        if (count/(Posi[i].first.size() + Posi[i].second.size())>count_thres)
        {
            // cout << "Most part of this object is on the image boundary......" << endl;
            for (int k = 0; k < Posi[i].first.size(); ++k) 
            {
                mCurrentFrame.vObjLabel[Posi[i].first[k]] = -1;
            }
            for (int k=0; k < Posi[i].second.size(); ++k)
            {
                mCurrentFrame.vObjLabel_Line[Posi[i].second[k]] = -1;
                continue;
            }
        }
        else
        {
            ObjId.push_back(Posi[i].first);
            sem_posi.push_back(UniLab[i]);
            //I comment out the below because now we do not pay attention to objects who have only lines
            //sem_posi_line.push_back(UniLab_Line[i]);
            ObjId_Line.push_back(Posi[i].second);
        }
    }

    // // Check scene flow distribution of each object and keep the dynamic object
    std::vector<std::vector<int> > ObjIdNew, ObjIdNew_Line;
    std::vector<int> SemPosNew, obj_dis_tres(sem_posi.size(),0);
    for (int i = 0; i < ObjId.size(); ++i)
    {

        float obj_center_depth = 0, sf_min=100, sf_max=0, sf_mean=0, sf_count=0;
        std::vector<int> sf_range(10,0);
        for (int j = 0; j < ObjId[i].size(); ++j)
        {
            obj_center_depth = obj_center_depth + mCurrentFrame.mvObjDepth[ObjId[i][j]];
            // const float sf_norm = cv::norm(mCurrentFrame.vFlow_3d[ObjId[i][j]]);
            float sf_norm = std::sqrt(mCurrentFrame.vFlow_3d[ObjId[i][j]].x*mCurrentFrame.vFlow_3d[ObjId[i][j]].x + mCurrentFrame.vFlow_3d[ObjId[i][j]].z*mCurrentFrame.vFlow_3d[ObjId[i][j]].z);
            if (sf_norm<fSFMgThres)
                sf_count = sf_count+1;
            if(sf_norm<sf_min)
                sf_min = sf_norm;
            if(sf_norm>sf_max)
                sf_max = sf_norm;
            sf_mean = sf_mean + sf_norm;
            {
                if (0.0<=sf_norm && sf_norm<0.05)
                    sf_range[0] = sf_range[0] + 1;
                else if (0.05<=sf_norm && sf_norm<0.1)
                    sf_range[1] = sf_range[1] + 1;
                else if (0.1<=sf_norm && sf_norm<0.2)
                    sf_range[2] = sf_range[2] + 1;
                else if (0.2<=sf_norm && sf_norm<0.4)
                    sf_range[3] = sf_range[3] + 1;
                else if (0.4<=sf_norm && sf_norm<0.8)
                    sf_range[4] = sf_range[4] + 1;
                else if (0.8<=sf_norm && sf_norm<1.6)
                    sf_range[5] = sf_range[5] + 1;
                else if (1.6<=sf_norm && sf_norm<3.2)
                    sf_range[6] = sf_range[6] + 1;
                else if (3.2<=sf_norm && sf_norm<6.4)
                    sf_range[7] = sf_range[7] + 1;
                else if (6.4<=sf_norm && sf_norm<12.8)
                    sf_range[8] = sf_range[8] + 1;
                else if (12.8<=sf_norm && sf_norm<25.6)
                    sf_range[9] = sf_range[9] + 1;
            }
        }

        // cout << "scene flow distribution:"  << endl;
        // for (int j = 0; j < sf_range.size(); ++j)
        //     cout << sf_range[j] << " ";
        // cout << endl;

        if (sf_count/ObjId[i].size()>fSFDsThres)
        {
            // label this object as static background
            for (int k = 0; k < ObjId[i].size(); ++k)
                mCurrentFrame.vObjLabel[ObjId[i][k]] = 0;
            continue;
        }
        else if (obj_center_depth/ObjId[i].size()>mThDepthObj || ObjId[i].size()<150)
        {
            obj_dis_tres[i]=-1;
            // cout << "object " << sem_posi[i] <<" is too far away or too small! " << obj_center_depth/ObjId[i].size() << endl;
            // label this object as far away object
            for (int k = 0; k < ObjId[i].size(); ++k)
                mCurrentFrame.vObjLabel[ObjId[i][k]] = -1;
            continue;
        }
        else
        {
            // cout << "get new objects!" << endl;
            ObjIdNew.push_back(ObjId[i]);
            SemPosNew.push_back(sem_posi[i]);
            ObjIdNew_Line.push_back(ObjId_Line[i]);
        }
    }

    // add ground truth tracks
    std::vector<int> nSemPosi_gt_tmp = mCurrentFrame.nSemPosi_gt;
    for (int i = 0; i < sem_posi.size(); ++i)
    {
        for (int j = 0; j < nSemPosi_gt_tmp.size(); ++j)
        {
            if (sem_posi[i]==nSemPosi_gt_tmp[j] && obj_dis_tres[i]==-1)
            {
                nSemPosi_gt_tmp[j]=-1;
            }
        }
    }

    mpMap->vnSMLabelGT.push_back(nSemPosi_gt_tmp);


    // // *** show the points on object ***
    // for (int i = 0; i < ObjIdNew.size(); ++i)
    // {
    //     // **** show the picked points ****
    //     std::vector<cv::KeyPoint> PickKeys;
    //     for (int j = 0; j < ObjIdNew[i].size(); ++j){
    //         PickKeys.push_back(mCurrentFrame.mvObjKeys[ObjIdNew[i][j]]);
    //     }
    //     cv::drawKeypoints(mImGray, PickKeys, mImGray, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    //     cv::imshow("KeyPoints and Grid on Background", mImGray);
    //     cv::waitKey(0);
    // }

    // Relabel the objects that associate with the objects in last frame

    // initialize global object id
    if (f_id==1)
        max_id = 1;

    // save current label id
    std::vector<int> LabId(ObjIdNew.size());
    //ObjIdNew and ObjIdNew_Line have the same size
    for (int i = 0; i < ObjIdNew.size(); ++i)
    {
        // save semantic labels in last frame
        std::vector<int> Lb_last;
        for (int k = 0; k < ObjIdNew[i].size(); ++k)
            Lb_last.push_back(mLastFrame.vSemObjLabel[ObjIdNew[i][k]]);
        for (int k = 0; k < ObjIdNew_Line[i].size(); ++k)
            Lb_last.push_back(mLastFrame.vSemObjLabel_Line[ObjIdNew_Line[i][k]]);
        // find label that appears most in Lb_last()
        // (1) count duplicates
        std::map<int, int> dups;
        for(int k : Lb_last)
            ++dups[k];
        // (2) and sort them by descending order
        std::vector<std::pair<int, int> > sorted;
        for (auto k : dups)
            sorted.push_back(std::make_pair(k.first,k.second));
        std::sort(sorted.begin(), sorted.end(), SortPairInt);

        // label the object in current frame
        int New_lab = sorted[0].first;
        // cout << " what is in the new label: " << New_lab << endl;
        if (max_id==1)
        {
            LabId[i] = max_id;
            for (int k = 0; k < ObjIdNew[i].size(); ++k)
                mCurrentFrame.vObjLabel[ObjIdNew[i][k]] = max_id;
            for (int k = 0; k < ObjIdNew_Line[i].size(); ++k)
                mCurrentFrame.vObjLabel_Line[ObjIdNew_Line[i][k]] = max_id;
            max_id = max_id + 1;
        }
        else
        {
            bool exist = false;
            for (int k = 0; k < mLastFrame.nSemPosition.size(); ++k)
            {
                if (mLastFrame.nSemPosition[k]==New_lab && mLastFrame.bObjStat[k])
                {
                    LabId[i] = mLastFrame.nModLabel[k];
                    for (int k = 0; k < ObjIdNew[i].size(); ++k)
                        mCurrentFrame.vObjLabel[ObjIdNew[i][k]] = LabId[i];
                    for (int k = 0; k < ObjIdNew_Line[i].size(); ++k)
                        mCurrentFrame.vObjLabel_Line[ObjIdNew_Line[i][k]] = LabId[i];
                    exist = true;
                    break;
                }
            }
            if (exist==false)
            {
                LabId[i] = max_id;
                for (int k = 0; k < ObjIdNew[i].size(); ++k)
                    mCurrentFrame.vObjLabel[ObjIdNew[i][k]] = max_id;
                for (int k = 0; k < ObjIdNew_Line[i].size(); ++k)
                    mCurrentFrame.vObjLabel_Line[ObjIdNew_Line[i][k]] = max_id;
                max_id = max_id + 1;
            }
        }

    }

    // // assign the model label in current frame
    mCurrentFrame.nModLabel = LabId;
    mCurrentFrame.nSemPosition = SemPosNew;
    

    e_2 = clock();
    obj_tra_time = (double)(e_2-s_2)/CLOCKS_PER_SEC*1000;
    all_timing[2] = obj_tra_time;
    // cout << "dynamic object tracking time: " << obj_tra_time << endl;

    cout << "Current Max_id: ("<< max_id << ") motion label: ";
    for (int i = 0; i < LabId.size(); ++i)
        cout <<  LabId[i] << " ";


    //std::cout << "ObjIdNew: " << ObjIdNew.size() << " ObjIdNew_Line: " << ObjIdNew_Line.size() << std::endl;
    //print ObjIdNew
    // for (int i = 0; i < ObjIdNew.size(); ++i)
    // {
    //     cout << "ObjIdNew: ";
    //     for (int j = 0; j < ObjIdNew[i].size(); ++j)
    //         cout << ObjIdNew[i][j] << " ";
    //     cout << endl;
    // }

    // //print ObjIdNew_Line
    // for (int i = 0; i < ObjIdNew_Line.size(); ++i)
    // {
    //     cout << "ObjIdNew_Line: ";
    //     for (int j = 0; j < ObjIdNew_Line[i].size(); ++j)
    //         cout << ObjIdNew_Line[i][j] << " ";
    //     cout << endl;
    // }

    std::pair<std::vector<std::vector<int> >, std::vector<std::vector<int> > > ObjIdNew_pair;
    ObjIdNew_pair.first = ObjIdNew;
    ObjIdNew_pair.second = ObjIdNew_Line;

    return ObjIdNew_pair;
}

cv::Mat Tracking::GetInitModelCam(const std::vector<int> &MatchId, const std::vector<int> &MatchId_Line, std::vector<int> &MatchId_sub, std::vector<int> &MatchId_sub_Line)
{
    cv::Mat Mod = cv::Mat::eye(4,4,CV_32F);
    int N = MatchId.size();

    // construct input
    std::vector<cv::Point2f> cur_2d(N);
    std::vector<cv::Point3f> pre_3d(N);
    for (int i = 0; i < N; ++i)
    {
        cv::Point2f tmp_2d;
        tmp_2d.x = mCurrentFrame.mvStatKeys[MatchId[i]].pt.x;
        tmp_2d.y = mCurrentFrame.mvStatKeys[MatchId[i]].pt.y;
        cur_2d[i] = tmp_2d;
        cv::Point3f tmp_3d;
        cv::Mat x3D_p = mLastFrame.UnprojectStereoStat(MatchId[i],0);
        tmp_3d.x = x3D_p.at<float>(0);
        tmp_3d.y = x3D_p.at<float>(1);
        tmp_3d.z = x3D_p.at<float>(2);
        pre_3d[i] = tmp_3d;
    }

    // camera matrix & distortion coefficients
    cv::Mat camera_mat(3, 3, CV_64FC1);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 4, CV_64FC1);
    camera_mat.at<double>(0, 0) = mK.at<float>(0,0);
    camera_mat.at<double>(1, 1) = mK.at<float>(1,1);
    camera_mat.at<double>(0, 2) = mK.at<float>(0,2);
    camera_mat.at<double>(1, 2) = mK.at<float>(1,2);
    camera_mat.at<double>(2, 2) = 1.0;

    // output
    cv::Mat Rvec(3, 1, CV_64FC1);
    cv::Mat Tvec(3, 1, CV_64FC1);
    cv::Mat d(3, 3, CV_64FC1);
    cv::Mat inliers;

    // solve
    int iter_num = 500;
    double reprojectionError = 0.4, confidence = 0.98; // 0.5 0.3
    cv::solvePnPRansac(pre_3d, cur_2d, camera_mat, distCoeffs, Rvec, Tvec, false,
               iter_num, reprojectionError, confidence, inliers, cv::SOLVEPNP_AP3P); // AP3P EPNP P3P ITERATIVE DLS

    cv::Rodrigues(Rvec, d);

    // assign the result to current pose
    Mod.at<float>(0,0) = d.at<double>(0,0); Mod.at<float>(0,1) = d.at<double>(0,1); Mod.at<float>(0,2) = d.at<double>(0,2); Mod.at<float>(0,3) = Tvec.at<double>(0,0);
    Mod.at<float>(1,0) = d.at<double>(1,0); Mod.at<float>(1,1) = d.at<double>(1,1); Mod.at<float>(1,2) = d.at<double>(1,2); Mod.at<float>(1,3) = Tvec.at<double>(1,0);
    Mod.at<float>(2,0) = d.at<double>(2,0); Mod.at<float>(2,1) = d.at<double>(2,1); Mod.at<float>(2,2) = d.at<double>(2,2); Mod.at<float>(2,3) = Tvec.at<double>(2,0);


    // calculate the re-projection error
    std::vector<int> MM_inlier;
    cv::Mat MotionModel;
    if (mVelocity.empty())
        MotionModel = cv::Mat::eye(4,4,CV_32F)*mLastFrame.mTcw;
    else
        MotionModel = mVelocity*mLastFrame.mTcw;
    for (int i = 0; i < N; ++i)
    {
        const cv::Mat x3D  = (cv::Mat_<float>(3,1) << pre_3d[i].x, pre_3d[i].y, pre_3d[i].z);
        const cv::Mat x3D_c = MotionModel.rowRange(0,3).colRange(0,3)*x3D+MotionModel.rowRange(0,3).col(3);

        const float xc = x3D_c.at<float>(0);
        const float yc = x3D_c.at<float>(1);
        const float invzc = 1.0/x3D_c.at<float>(2);
        const float u = mCurrentFrame.fx*xc*invzc+mCurrentFrame.cx;
        const float v = mCurrentFrame.fy*yc*invzc+mCurrentFrame.cy;
        const float u_ = cur_2d[i].x - u;
        const float v_ = cur_2d[i].y - v;
        const float Rpe = std::sqrt(u_*u_ + v_*v_);
        if (Rpe<reprojectionError){
            MM_inlier.push_back(i);
        }
    }

    // cout << "Inlier Compare: " << "(1)AP3P RANSAC: " << inliers.rows << " (2)Motion Model: " << MM_inlier.size() << endl;

    cv::Mat output;

    if (inliers.rows>MM_inlier.size())
    {
        // save the inliers IDs
        output = Mod;
        MatchId_sub.resize(inliers.rows);
        for (int i = 0; i < MatchId_sub.size(); ++i){
            MatchId_sub[i] = MatchId[inliers.at<int>(i)];
        }
        // cout << "(Camera) AP3P+RanSac inliers/total number: " << inliers.rows << "/" << MatchId.size() << endl;
    }
    else
    {
        output = MotionModel;
        MatchId_sub.resize(MM_inlier.size());
        for (int i = 0; i < MatchId_sub.size(); ++i){
            MatchId_sub[i] = MatchId[MM_inlier[i]];
        }
        // cout << "(Camera) Motion Model inliers/total number: " << MM_inlier.size() << "/" << MatchId.size() << endl;
    }

    return output;
}

cv::Mat Tracking::GetInitModelObj(const std::vector<int> &ObjId, std::vector<int> &ObjId_sub, const int objid)
{
    cv::Mat Mod = cv::Mat::eye(4,4,CV_32F);
    int N = ObjId.size();

    // construct input
    std::vector<cv::Point2f> cur_2d(N);
    std::vector<cv::Point3f> pre_3d(N);
    for (int i = 0; i < N; ++i)
    {
        cv::Point2f tmp_2d;
        tmp_2d.x = mCurrentFrame.mvObjKeys[ObjId[i]].pt.x;
        tmp_2d.y = mCurrentFrame.mvObjKeys[ObjId[i]].pt.y;
        cur_2d[i] = tmp_2d;
        cv::Point3f tmp_3d;
        cv::Mat x3D_p = mLastFrame.UnprojectStereoObject(ObjId[i],0);
        tmp_3d.x = x3D_p.at<float>(0);
        tmp_3d.y = x3D_p.at<float>(1);
        tmp_3d.z = x3D_p.at<float>(2);
        pre_3d[i] = tmp_3d;
    }

    // camera matrix & distortion coefficients
    cv::Mat camera_mat(3, 3, CV_64FC1);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 4, CV_64FC1);
    camera_mat.at<double>(0, 0) = mK.at<float>(0,0);
    camera_mat.at<double>(1, 1) = mK.at<float>(1,1);
    camera_mat.at<double>(0, 2) = mK.at<float>(0,2);
    camera_mat.at<double>(1, 2) = mK.at<float>(1,2);
    camera_mat.at<double>(2, 2) = 1.0;

    // output
    cv::Mat Rvec(3, 1, CV_64FC1);
    cv::Mat Tvec(3, 1, CV_64FC1);
    cv::Mat d(3, 3, CV_64FC1);
    cv::Mat inliers;

    // solve
    int iter_num = 500;
    double reprojectionError = 0.4, confidence = 0.98; // 0.3 0.5 1.0
    cv::solvePnPRansac(pre_3d, cur_2d, camera_mat, distCoeffs, Rvec, Tvec, false,
               iter_num, reprojectionError, confidence, inliers, cv::SOLVEPNP_AP3P); // AP3P EPNP P3P ITERATIVE DLS

    cv::Rodrigues(Rvec, d);

    // assign the result to current pose
    Mod.at<float>(0,0) = d.at<double>(0,0); Mod.at<float>(0,1) = d.at<double>(0,1); Mod.at<float>(0,2) = d.at<double>(0,2); Mod.at<float>(0,3) = Tvec.at<double>(0,0);
    Mod.at<float>(1,0) = d.at<double>(1,0); Mod.at<float>(1,1) = d.at<double>(1,1); Mod.at<float>(1,2) = d.at<double>(1,2); Mod.at<float>(1,3) = Tvec.at<double>(1,0);
    Mod.at<float>(2,0) = d.at<double>(2,0); Mod.at<float>(2,1) = d.at<double>(2,1); Mod.at<float>(2,2) = d.at<double>(2,2); Mod.at<float>(2,3) = Tvec.at<double>(2,0);

    // ******* Generate Motion Model if it does exist from previous frame *******
    int CurObjLab = mCurrentFrame.nModLabel[objid];
    int PreObjID = -1;
    for (int i = 0; i < mLastFrame.nModLabel.size(); ++i)
    {
        if (mLastFrame.nModLabel[i]==CurObjLab)
        {
            PreObjID = i;
            break;
        }
    }

    cv::Mat MotionModel, output;
    std::vector<int> ObjId_tmp(N,-1); // new added Nov 19, 2019
    if (PreObjID!=-1)
    {
        // calculate the re-projection error
        std::vector<int> MM_inlier;
        MotionModel = mCurrentFrame.mTcw*mLastFrame.vObjMod[PreObjID];
        for (int i = 0; i < N; ++i)
        {
            const cv::Mat x3D  = (cv::Mat_<float>(3,1) << pre_3d[i].x, pre_3d[i].y, pre_3d[i].z);
            const cv::Mat x3D_c = MotionModel.rowRange(0,3).colRange(0,3)*x3D+MotionModel.rowRange(0,3).col(3);

            const float xc = x3D_c.at<float>(0);
            const float yc = x3D_c.at<float>(1);
            const float invzc = 1.0/x3D_c.at<float>(2);
            const float u = mCurrentFrame.fx*xc*invzc+mCurrentFrame.cx;
            const float v = mCurrentFrame.fy*yc*invzc+mCurrentFrame.cy;
            const float u_ = cur_2d[i].x - u;
            const float v_ = cur_2d[i].y - v;
            const float Rpe = std::sqrt(u_*u_ + v_*v_);
            if (Rpe<reprojectionError){
                MM_inlier.push_back(i);
            }
        }

        // cout << "Inlier Compare: " << "(1)AP3P RANSAC: " << inliers.rows << " (2)Motion Model: " << MM_inlier.size() << endl;

        // ===== decide which model is best now =====
        if (inliers.rows>MM_inlier.size())
        {
            // save the inliers IDs
            output = Mod;
            ObjId_sub.resize(inliers.rows);
            for (int i = 0; i < ObjId_sub.size(); ++i){
                ObjId_sub[i] = ObjId[inliers.at<int>(i)];
                ObjId_tmp[inliers.at<int>(i)] = ObjId[inliers.at<int>(i)];
            }
            // cout << "(Object) AP3P+RanSac inliers/total number: " << inliers.rows << "/" << ObjId.size() << endl;
        }
        else
        {
            output = MotionModel;
            ObjId_sub.resize(MM_inlier.size());
            for (int i = 0; i < ObjId_sub.size(); ++i){
                ObjId_sub[i] = ObjId[MM_inlier[i]];
                ObjId_tmp[MM_inlier[i]] = ObjId[MM_inlier[i]];
            }
            // cout << "(Object) Motion Model inliers/total number: " << MM_inlier.size() << "/" << ObjId.size() << endl;
        }
    }
    else
    {
        // save the inliers IDs
        output = Mod;
        ObjId_sub.resize(inliers.rows);
        for (int i = 0; i < ObjId_sub.size(); ++i){
            ObjId_sub[i] = ObjId[inliers.at<int>(i)];
            ObjId_tmp[inliers.at<int>(i)] = ObjId[inliers.at<int>(i)];
        }
        // cout << "(Object) AP3P+RanSac [No MM] inliers/total number: " << inliers.rows << "/" << ObjId.size() << endl;
    }

    // update on vObjLabel (Nov 19 2019)
    for (int i = 0; i < ObjId_tmp.size(); ++i)
    {
        if (ObjId_tmp[i]==-1)
            mCurrentFrame.vObjLabel[ObjId[i]]=-1;
    }

    return output;
}

void Tracking::DrawLine(cv::KeyPoint &keys, cv::Point2f &flow, cv::Mat &ref_image, const cv::Scalar &color, int thickness, int line_type, const cv::Point2i &offset)
{

    auto cv_p1 = cv::Point2i(keys.pt.x,keys.pt.y);
    auto cv_p2 = cv::Point2i(keys.pt.x+flow.x,keys.pt.y+flow.y);
    //cout << "p1: " << cv_p1 << endl;
    //cout << "p2: " << cv_p2 << endl;

    bool p1_in_bounds = true;
    bool p2_in_bounds = true;
    if ((cv_p1.x < 0) && (cv_p1.y < 0) && (cv_p1.x > ref_image.cols) && (cv_p1.y > ref_image.rows) )
        p1_in_bounds = false;

    if ((cv_p2.x < 0) && (cv_p2.y < 0) && (cv_p2.x > ref_image.cols) && (cv_p2.y > ref_image.rows) )
        p2_in_bounds = false;

    // Draw line, but only if both end-points project into the image!
    if (p1_in_bounds || p2_in_bounds) { // This is correct. Won't draw only if both lines are out of bounds.
        // Draw line
        auto p1_offs = offset+cv_p1;
        auto p2_offs = offset+cv_p2;
        if (cv::clipLine(cv::Size(ref_image.cols, ref_image.rows), p1_offs, p2_offs)) {
            //cv::line(ref_image, p1_offs, p2_offs, color, thickness, line_type);
            cv::arrowedLine(ref_image, p1_offs, p2_offs, color, thickness, line_type);
        }
    }
}

void Tracking::DrawTransparentSquare(cv::Point center, cv::Vec3b color, int radius, double alpha, cv::Mat &ref_image)
{
    for (int i=-radius; i<radius; i++) {
        for (int j=-radius; j<radius; j++) {
            int coord_y = center.y + i;
            int coord_x = center.x + j;

            if (coord_x>0 && coord_y>0 && coord_x<ref_image.cols && coord_y < ref_image.rows) {
                ref_image.at<cv::Vec3b>(cv::Point(coord_x,coord_y)) = (1.0-alpha)*ref_image.at<cv::Vec3b>(cv::Point(coord_x,coord_y)) + alpha*color;
            }
        }
    }
}

void Tracking::DrawGridBirdeye(double res_x, double res_z, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image)
{

    auto color = cv::Scalar(0.0, 0.0, 0.0);
    // Draw horizontal lines
    for (double i=0; i<viz_props.birdeye_far_plane_; i+=res_z) {
        double x_1 = viz_props.birdeye_left_plane_;
        double y_1 = i;
        double x_2 = viz_props.birdeye_right_plane_;
        double y_2 = i;
        TransformPointToScaledFrustum(x_1, y_1, viz_props);
        TransformPointToScaledFrustum(x_2, y_2, viz_props);
        auto p1 = cv::Point(x_1, y_1), p2=cv::Point(x_2,y_2);
        cv::line(ref_image, p1, p2, color);
    }

    // Draw vertical lines
    for (double i=viz_props.birdeye_left_plane_; i<viz_props.birdeye_right_plane_; i+=res_x) {
        double x_1 = i;
        double y_1 = 0;
        double x_2 = i;
        double y_2 = viz_props.birdeye_far_plane_;
        TransformPointToScaledFrustum(x_1, y_1, viz_props);
        TransformPointToScaledFrustum(x_2, y_2, viz_props);
        auto p1 = cv::Point(x_1, y_1), p2=cv::Point(x_2,y_2);
        cv::line(ref_image, p1, p2, color);
    }
}

void Tracking::DrawSparseFlowBirdeye(
        const std::vector<Eigen::Vector3d> &pts, const std::vector<Eigen::Vector3d> &vel,
        const cv::Mat &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image)
{

    // For scaling / flipping cov. matrices
    Eigen::Matrix2d flip_mat;
    flip_mat << viz_props.birdeye_scale_factor_*1.0, 0, 0, viz_props.birdeye_scale_factor_*1.0;
    Eigen::Matrix2d world_to_cam_mat;
    const Eigen::Matrix4d &ref_to_rt_inv = Converter::toMatrix4d(camera);
    world_to_cam_mat << ref_to_rt_inv(0,0), ref_to_rt_inv(2,0), ref_to_rt_inv(0,2), ref_to_rt_inv(2,2);
    flip_mat = flip_mat*world_to_cam_mat;

    // Parameters
    // const int line_width = 2;

    ref_image = cv::Mat(viz_props.birdeye_scale_factor_*viz_props.birdeye_far_plane_,
                        (-viz_props.birdeye_left_plane_+viz_props.birdeye_right_plane_)*viz_props.birdeye_scale_factor_, CV_32FC3);
    ref_image.setTo(cv::Scalar(1.0, 1.0, 1.0));
    Tracking::DrawGridBirdeye(1.0, 1.0, viz_props, ref_image);


    for (int i=0; i<pts.size(); i++) {

        Eigen::Vector3d p_3d = pts[i];
        Eigen::Vector3d p_vel = vel[i];

        if (p_3d[0]==-1 || p_3d[1]==-1 || p_3d[2]<0)
            continue;
        if (p_vel[0]>0.1 || p_vel[2]>0.1)
            continue;

        // float xc = p_3d[0];
        // float yc = p_3d[1];
        // float invzc = 1.0/p_3d[2];
        // float u = mCurrentFrame.fx*xc*invzc+mCurrentFrame.cx;
        // float v = mCurrentFrame.fy*yc*invzc+mCurrentFrame.cy;
        // Eigen::Vector3i p_proj = Eigen::Vector3i(round(u), round(v), 1);
        const Eigen::Vector2d velocity = Eigen::Vector2d(p_vel[0], p_vel[2]); // !!!
        Eigen::Vector3d dir(velocity[0], 0.0, velocity[1]);

        double x_1 = p_3d[0];
        double z_1 = p_3d[2];

        double x_2 = x_1 + dir[0];
        double z_2 = z_1 + dir[2];

        // cout << dir[0] << " " << dir[2] << endl;

        if (x_1 > viz_props.birdeye_left_plane_ && x_2 > viz_props.birdeye_left_plane_ &&
            x_1 < viz_props.birdeye_right_plane_ && x_2 < viz_props.birdeye_right_plane_ &&
            z_1 > 0 && z_2 > 0 &&
            z_1 < viz_props.birdeye_far_plane_ && z_2 < viz_props.birdeye_far_plane_) {

            TransformPointToScaledFrustum(x_1, z_1, viz_props); //velocity[0], velocity[1]);
            TransformPointToScaledFrustum(x_2, z_2, viz_props); //velocity[0], velocity[1]);

            cv::arrowedLine(ref_image, cv::Point(x_1, z_1), cv::Point(x_2, z_2), cv::Scalar(1.0, 0.0, 0.0), 1);
            cv::circle(ref_image, cv::Point(x_1, z_1), 3.0, cv::Scalar(0.0, 0.0, 1.0), -1.0);
        }
    }

    // Coord. sys.
    int arrow_len = 60;
    int offset_y = 10;
    cv::arrowedLine(ref_image, cv::Point(ref_image.cols/2, offset_y),
                    cv::Point(ref_image.cols/2+arrow_len, offset_y),
                    cv::Scalar(1.0, 0, 0), 2);
    cv::arrowedLine(ref_image, cv::Point(ref_image.cols/2, offset_y),
                    cv::Point(ref_image.cols/2, offset_y+arrow_len),
                    cv::Scalar(0.0, 1.0, 0), 2);

    //cv::putText(ref_image, "X", cv::Point(ref_image.cols/2+arrow_len+10, offset_y+10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(1.0, 0, 0));
    //cv::putText(ref_image, "Z", cv::Point(ref_image.cols/2+10, offset_y+arrow_len), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0.0, 1.0, 0));

    // Flip image, because it is more intuitive to have ref. point at the bottom of the image
    cv::Mat dst;
    cv::flip(ref_image, dst, 0);
    ref_image = dst;
}

void Tracking::TransformPointToScaledFrustum(double &pose_x, double &pose_z, const BirdEyeVizProperties &viz_props)
{
    pose_x += (-viz_props.birdeye_left_plane_);
    pose_x *= viz_props.birdeye_scale_factor_;
    pose_z *= viz_props.birdeye_scale_factor_;
}

cv::Mat Tracking::ObjPoseParsingKT(const std::vector<float> &vObjPose_gt)
{
    // assign t vector
    cv::Mat t(3, 1, CV_32FC1);
    t.at<float>(0) = vObjPose_gt[6];
    t.at<float>(1) = vObjPose_gt[7];
    t.at<float>(2) = vObjPose_gt[8];

    // from Euler to Rotation Matrix
    cv::Mat R(3, 3, CV_32FC1);

    // assign r vector
    float y = vObjPose_gt[9]+(3.1415926/2); // +(3.1415926/2)
    float x = 0.0;
    float z = 0.0;

    // the angles are in radians.
    float cy = cos(y);
    float sy = sin(y);
    float cx = cos(x);
    float sx = sin(x);
    float cz = cos(z);
    float sz = sin(z);

    float m00, m01, m02, m10, m11, m12, m20, m21, m22;

    // ====== R = Ry*Rx*Rz =======

    // m00 = cy;
    // m01 = -sy;
    // m02 = 0;
    // m10 = sy;
    // m11 = cy;
    // m12 = 0;
    // m20 = 0;
    // m21 = 0;
    // m22 = 1;

    m00 = cy*cz+sy*sx*sz;
    m01 = -cy*sz+sy*sx*cz;
    m02 = sy*cx;
    m10 = cx*sz;
    m11 = cx*cz;
    m12 = -sx;
    m20 = -sy*cz+cy*sx*sz;
    m21 = sy*sz+cy*sx*cz;
    m22 = cy*cx;

    // ***************** old **************************

    // float alpha = vObjPose_gt[7]; // 7
    // float beta = vObjPose_gt[5]+(3.1415926/2);  // 5
    // float gamma = vObjPose_gt[6]; // 6

    // the angles are in radians.
    // float ca = cos(alpha);
    // float sa = sin(alpha);
    // float cb = cos(beta);
    // float sb = sin(beta);
    // float cg = cos(gamma);
    // float sg = sin(gamma);

    // float m00, m01, m02, m10, m11, m12, m20, m21, m22;

    // default
    // m00 = cb*ca;
    // m01 = cb*sa;
    // m02 = -sb;
    // m10 = sb*sg*ca-sa*cg;
    // m11 = sb*sg*sa+ca*cg;
    // m12 = cb*sg;
    // m20 = sb*cg*ca+sa*sg;
    // m21 = sb*cg*sa-ca*sg;
    // m22 = cb*cg;

    // m00 = ca*cb;
    // m01 = ca*sb*sg - sa*cg;
    // m02 = ca*sb*cg + sa*sg;
    // m10 = sa*cb;
    // m11 = sa*sb*sg + ca*cg;
    // m12 = sa*sb*cg - ca*sg;
    // m20 = -sb;
    // m21 = cb*sg;
    // m22 = cb*cg;

    // **************************************************

    R.at<float>(0,0) = m00;
    R.at<float>(0,1) = m01;
    R.at<float>(0,2) = m02;
    R.at<float>(1,0) = m10;
    R.at<float>(1,1) = m11;
    R.at<float>(1,2) = m12;
    R.at<float>(2,0) = m20;
    R.at<float>(2,1) = m21;
    R.at<float>(2,2) = m22;

    // construct 4x4 transformation matrix
    cv::Mat Pose = cv::Mat::eye(4,4,CV_32F);
    Pose.at<float>(0,0) = R.at<float>(0,0); Pose.at<float>(0,1) = R.at<float>(0,1); Pose.at<float>(0,2) = R.at<float>(0,2); Pose.at<float>(0,3) = t.at<float>(0);
    Pose.at<float>(1,0) = R.at<float>(1,0); Pose.at<float>(1,1) = R.at<float>(1,1); Pose.at<float>(1,2) = R.at<float>(1,2); Pose.at<float>(1,3) = t.at<float>(1);
    Pose.at<float>(2,0) = R.at<float>(2,0); Pose.at<float>(2,1) = R.at<float>(2,1); Pose.at<float>(2,2) = R.at<float>(2,2); Pose.at<float>(2,3) = t.at<float>(2);

    // cout << "OBJ Pose: " << endl << Pose << endl;

    return Pose;

}

cv::Mat Tracking::ObjPoseParsingOX(const std::vector<float> &vObjPose_gt)
{
    // assign t vector
    cv::Mat t(3, 1, CV_32FC1);
    t.at<float>(0) = vObjPose_gt[2];
    t.at<float>(1) = vObjPose_gt[3];
    t.at<float>(2) = vObjPose_gt[4];

    // from axis-angle to Rotation Matrix
    cv::Mat R(3, 3, CV_32FC1);
    cv::Mat Rvec(3, 1, CV_32FC1);

    // assign r vector
    Rvec.at<float>(0,0) = vObjPose_gt[5];
    Rvec.at<float>(0,1) = vObjPose_gt[6];
    Rvec.at<float>(0,2) = vObjPose_gt[7];

    // *******************************************************************

    const float angle = std::sqrt(vObjPose_gt[5]*vObjPose_gt[5] + vObjPose_gt[6]*vObjPose_gt[6] + vObjPose_gt[7]*vObjPose_gt[7]);

    if (angle>0)
    {
        Rvec.at<float>(0,0) = Rvec.at<float>(0,0)/angle;
        Rvec.at<float>(0,1) = Rvec.at<float>(0,1)/angle;
        Rvec.at<float>(0,2) = Rvec.at<float>(0,2)/angle;
    }

    const float s = std::sin(angle);
    const float c = std::cos(angle);

    const float v = 1 - c;
    const float x = Rvec.at<float>(0,0);
    const float y = Rvec.at<float>(0,1);
    const float z = Rvec.at<float>(0,2);
    const float xyv = x*y*v;
    const float yzv = y*z*v;
    const float xzv = x*z*v;

    R.at<float>(0,0) = x*x*v + c;
    R.at<float>(0,1) = xyv - z*s;
    R.at<float>(0,2) = xzv + y*s;
    R.at<float>(1,0) = xyv + z*s;
    R.at<float>(1,1) = y*y*v + c;
    R.at<float>(1,2) = yzv - x*s;
    R.at<float>(2,0) = xzv - y*s;
    R.at<float>(2,1) = yzv + x*s;
    R.at<float>(2,2) = z*z*v + c;

    // ********************************************************************

    // cv::Rodrigues(Rvec, R);

    // construct 4x4 transformation matrix
    cv::Mat Pose = cv::Mat::eye(4,4,CV_32F);
    Pose.at<float>(0,0) = R.at<float>(0,0); Pose.at<float>(0,1) = R.at<float>(0,1); Pose.at<float>(0,2) = R.at<float>(0,2); Pose.at<float>(0,3) = t.at<float>(0);
    Pose.at<float>(1,0) = R.at<float>(1,0); Pose.at<float>(1,1) = R.at<float>(1,1); Pose.at<float>(1,2) = R.at<float>(1,2); Pose.at<float>(1,3) = t.at<float>(1);
    Pose.at<float>(2,0) = R.at<float>(2,0); Pose.at<float>(2,1) = R.at<float>(2,1); Pose.at<float>(2,2) = R.at<float>(2,2); Pose.at<float>(2,3) = t.at<float>(2);

    // cout << "OBJ Pose: " << endl << Pose << endl;

    return Converter::toInvMatrix(mOriginInv)*Pose;

}


void Tracking::StackObjInfo(std::vector<cv::KeyPoint> &FeatDynObj, std::vector<float> &DepDynObj,
                  std::vector<int> &FeatLabObj)
{
    for (int i = 0; i < mCurrentFrame.vnObjID.size(); ++i)
    {
        for (int j = 0; j < mCurrentFrame.vnObjID[i].size(); ++j)
        {
            FeatDynObj.push_back(mLastFrame.mvObjKeys[mCurrentFrame.vnObjID[i][j]]);
            FeatDynObj.push_back(mCurrentFrame.mvObjKeys[mCurrentFrame.vnObjID[i][j]]);
            DepDynObj.push_back(mLastFrame.mvObjDepth[mCurrentFrame.vnObjID[i][j]]);
            DepDynObj.push_back(mCurrentFrame.mvObjDepth[mCurrentFrame.vnObjID[i][j]]);
            FeatLabObj.push_back(mCurrentFrame.vObjLabel[mCurrentFrame.vnObjID[i][j]]);
        }
    }
}

std::pair<std::vector<std::vector<std::pair<int, int>>>, std::vector<std::vector<std::pair<int, int>>>> Tracking::GetStaticTrack()
{
    // Get temporal match from Map
    std::vector<std::vector<int>> TemporalMatch = mpMap->vnAssoSta;
    std::vector<std::vector<int>> TemporalMatch_line = mpMap->vnAssoSta_line;
    //number of frames
    int N = TemporalMatch.size();
    // save the track id in TrackLets for previous frame and current frame.
    std::vector<int> TrackCheck_pre;
    std::vector<int> TrackCheck_pre_line;
    // pair.first = frameID; pair.second = featureID;
    std::vector<std::vector<std::pair<int, int> > > TrackLets;
    std::vector<std::vector<std::pair<int, int>>> TrackLets_line;

    // main loop
    //these shows on which tracklet is is placed
    int IDsofar(0), IDsofar_line(0);
    //std::cout << "the size of TemporalMatch is " << TemporalMatch.size() << " the size of TemporalMatch_line is " << TemporalMatch_line.size() << std::endl;
    for (int i = 0; i < N; ++i)
    {
        // initialize TrackCheck
        std::vector<int> TrackCheck_cur(TemporalMatch[i].size(),-1);
        std::vector<int> TrackCheck_cur_line(TemporalMatch_line[i].size(), -1);
        // check each feature
        for (int j = 0; j < TemporalMatch[i].size(); ++j)
        {
            // first pair of frames (frame 0 and 1)
            if(i==0)
            {
                // check if there's association
                if (TemporalMatch[i][j]!=-1)
                {
                    // first, save one tracklet consisting of two featureID
                    // pair.first = frameID; pair.second = featureID
                    std::vector<std::pair<int, int> > TraLet(2);
                    TraLet[0] = std::make_pair(i,TemporalMatch[i][j]);
                    TraLet[1] = std::make_pair(i+1,j);
                    // then, save to the main tracklets list
                    TrackLets.push_back(TraLet);

                    // save tracklet ID
                    TrackCheck_cur[j] = IDsofar;
                    IDsofar = IDsofar + 1;
                }
                else
                    continue;
            }
            // frame i and i+1 (i>0)
            else
            {
                // check if there's association
                if (TemporalMatch[i][j]!=-1)
                {
                    // check the TrackID in previous frame
                    // if it is associated before, then add to existing tracklets.
                    if (TrackCheck_pre[TemporalMatch[i][j]]!=-1)
                    {
                        TrackLets[TrackCheck_pre[TemporalMatch[i][j]]].push_back(std::make_pair(i+1,j));
                        TrackCheck_cur[j] = TrackCheck_pre[TemporalMatch[i][j]];
                    }
                    // if not, insert new tracklets.
                    else
                    {
                        // first, save one tracklet consisting of two featureID
                        std::vector<std::pair<int, int> > TraLet(2);
                        TraLet[0] = std::make_pair(i,TemporalMatch[i][j]);
                        TraLet[1] = std::make_pair(i+1,j);
                        // then, save to the main tracklets list
                        TrackLets.push_back(TraLet);

                        // save tracklet ID
                        TrackCheck_cur[j] = IDsofar;
                        IDsofar = IDsofar + 1;
                    }
                }
                else
                    continue;
            }
        }
        //check each line
        for (int j = 0; j < TemporalMatch_line[i].size(); ++j)
        {
            // first pair of frames (frame 0 and 1)
            if(i==0)
            {
                // check if there's association
                if (TemporalMatch_line[i][j]!=-1)
                {
                    // first, save one tracklet consisting of two featureID
                    // pair.first = frameID; pair.second = featureID
                    std::vector<std::pair<int, int> > TraLet(2);
                    TraLet[0] = std::make_pair(i,TemporalMatch_line[i][j]);
                    TraLet[1] = std::make_pair(i+1,j);
                    // then, save to the main tracklets list
                    TrackLets_line.push_back(TraLet);

                    // save tracklet ID
                    TrackCheck_cur_line[j] = IDsofar_line;
                    IDsofar_line = IDsofar_line + 1;
                }
                else
                    continue;
            }
            // frame i and i+1 (i>0)
            else
            {
                // check if there's association
                if (TemporalMatch_line[i][j]!=-1)
                {
                    // check the TrackID in previous frame
                    // if it is associated before, then add to existing tracklets.
                    if (TrackCheck_pre_line[TemporalMatch_line[i][j]]!=-1)
                    {
                        TrackLets_line[TrackCheck_pre_line[TemporalMatch_line[i][j]]].push_back(std::make_pair(i+1,j));
                        TrackCheck_cur_line[j] = TrackCheck_pre_line[TemporalMatch_line[i][j]];
                    }
                    // if not, insert new tracklets.
                    else
                    {
                        // first, save one tracklet consisting of two featureID
                        std::vector<std::pair<int, int> > TraLet(2);
                        TraLet[0] = std::make_pair(i,TemporalMatch_line[i][j]);
                        TraLet[1] = std::make_pair(i+1,j);
                        // then, save to the main tracklets list
                        TrackLets_line.push_back(TraLet);

                        // save tracklet ID
                        TrackCheck_cur_line[j] = IDsofar_line;
                        IDsofar_line = IDsofar_line + 1;
                    }
                }
                else
                    continue;
            }
        }

        TrackCheck_pre = TrackCheck_cur;
        TrackCheck_pre_line = TrackCheck_cur_line;
    }

    // display info
    cout << endl;
    cout << "==============================================" << endl;
    cout << "the number of static feature tracklets and line tracklets: " << TrackLets.size() << " " << TrackLets_line.size() << endl;
    cout << "==============================================" << endl; 
    cout << endl;

    std::vector<int> TrackLength(N,0);
    for (int i = 0; i < TrackLets.size(); ++i)
        TrackLength[TrackLets[i].size()-2]++;

    // for (int i = 0; i < N; ++i)
    //     cout << "The length of " << i+2 << " tracklets is found with the amount of " << TrackLength[i] << " ..." << endl;
    // cout << endl;

    int LengthOver_5 = 0;
    ofstream save_track_distri;
    string save_td = "track_distribution_static.txt";
    save_track_distri.open(save_td.c_str(),ios::trunc);
    for (int i = 0; i < N; ++i){
        if(TrackLength[i]!=0)
            save_track_distri << TrackLength[i] << endl;
        if (i+2>=5)
            LengthOver_5 = LengthOver_5 + TrackLength[i];
    }
    save_track_distri.close();
    cout << "Length over 5 (STATIC):::::::::::::::: " << LengthOver_5 << endl;


    // Save tracklet lengths in histogram
    std::vector<int> TrackLength_line(N+1,0);

    for (int i = 0; i < TrackLets_line.size(); ++i)
    {
        TrackLength_line[TrackLets_line[i].size()-1]++;
    }
    save_td = "track_distribution_static_line.txt";
    save_track_distri.open(save_td.c_str(),ios::trunc);
    for (int i = 0; i < N+1; ++i){
            save_track_distri << TrackLength_line[i] << endl;
    }
    save_track_distri.close();


    return make_pair(TrackLets, TrackLets_line);
}

std::pair<std::vector<std::vector<std::pair<int, int> > >, std::vector<std::vector<std::pair<int, int> > >> Tracking::GetDynamicTrackNew()
{
    // Get temporal match from Map
    std::vector<std::vector<int> > TemporalMatch = mpMap->vnAssoDyn;
    std::vector<std::vector<int>> TemporalMatch_line = mpMap->vnAssoDyn_line;
    //The following are the labels we have given to the objects
    std::vector<std::vector<int> > ObjLab = mpMap->vnFeatLabel;
    std::vector<std::vector<int>> ObjLab_line = mpMap->vnFeatLabel_line;
    int N = TemporalMatch.size();
    //std::cout << "size of frames taken from points " << N << " size of frames taken from lines " << TemporalMatch_line.size() << std::endl;
    // save the track id in TrackLets for previous frame and current frame.
    std::vector<int> TrackCheck_pre;
    std::vector<int> TrackCheck_pre_line;
    // pair.first = frameID; pair.second = featureID;
    std::vector<std::vector<std::pair<int, int> > > TrackLets;
    std::vector<std::vector<std::pair<int, int>>> TrackLets_line;
    // save object id of each tracklets
    std::vector<int> ObjectID;
    std::vector<int> ObjectID_line;

    // main loop
    int IDsofar = 0;
    int IDsofar_line = 0;
    // std::cout << "TemporalMatch" << std::endl;
    // for (int i = 0; i < TemporalMatch.size(); ++i)
    // {
    //   for (int j = 0; j < TemporalMatch[i].size(); ++j)
    //   {
    //     cout << TemporalMatch[i][j] << " ";
    //   }
    //   std::cout << endl;
    // }
    // for (int i = 0; i < TemporalMatch_line.size(); ++i)
    // {
    //   for (int j = 0; j < TemporalMatch_line[i].size(); ++j)
    //   {
    //     cout << TemporalMatch_line[i][j] << " ";
    //   }
    //   std::cout << endl;
    // }
    for (int i = 0; i < N; ++i)
    {
        // initialize TrackCheck
        std::vector<int> TrackCheck_cur(TemporalMatch[i].size(),-1);
        //std::cout << "For i " << i << " TemporalMatch_line[i].size() " << TemporalMatch_line[i].size() << std::endl;
        std::vector<int> TrackCheck_cur_line(TemporalMatch_line[i].size(), -1);
        // check each feature
        for (int j = 0; j < TemporalMatch[i].size(); ++j)
        {
            // first pair of frames (frame 0 and 1)
            if(i==0)
            {
                // check if there's association
                if (TemporalMatch[i][j]!=-1)
                {
                    // first, save one tracklet consisting of two featureID
                    // pair.first = frameID, pair.second = featureID
                    std::vector<std::pair<int, int> > TraLet(2);
                    TraLet[0] = std::make_pair(i,TemporalMatch[i][j]);
                    TraLet[1] = std::make_pair(i+1,j);
                    // then, save to the main tracklets list
                    TrackLets.push_back(TraLet);
                    ObjectID.push_back(ObjLab[i][j]);

                    // save tracklet ID
                    TrackCheck_cur[j] = IDsofar;
                    IDsofar = IDsofar + 1;
                }
            }
            // frame i and i+1 (i>0)
            else
            {
                // check if there's association
                if (TemporalMatch[i][j]!=-1)
                {
                    // check the TrackID in previous frame
                    // if it is associated before, then add to existing tracklets.
                    if (TrackCheck_pre[TemporalMatch[i][j]]!=-1)
                    {
                        TrackLets[TrackCheck_pre[TemporalMatch[i][j]]].push_back(std::make_pair(i+1,j));
                        TrackCheck_cur[j] = TrackCheck_pre[TemporalMatch[i][j]];
                    }
                    // if not, insert new tracklets.
                    else
                    {
                        // first, save one tracklet consisting of two featureID
                        std::vector<std::pair<int, int> > TraLet(2);
                        TraLet[0] = std::make_pair(i,TemporalMatch[i][j]);
                        TraLet[1] = std::make_pair(i+1,j);
                        // then, save to the main tracklets list
                        TrackLets.push_back(TraLet);
                        ObjectID.push_back(ObjLab[i][j]);

                        // save tracklet ID
                        TrackCheck_cur[j] = IDsofar;
                        IDsofar = IDsofar + 1;
                    }
                }
            }
        }
        for (int j = 0; j < TemporalMatch_line[i].size(); ++j)
        {
            // first pair of frames (frame 0 and 1)
            if(i==0)
            {
                // check if there's association
                if (TemporalMatch_line[i][j]!=-1)
                {
                    // first, save one tracklet consisting of two featureID
                    // pair.first = frameID, pair.second = featureID
                    std::vector<std::pair<int, int> > TraLet(2);
                    TraLet[0] = std::make_pair(i,TemporalMatch_line[i][j]);
                    TraLet[1] = std::make_pair(i+1,j);
                    // then, save to the main tracklets list
                    TrackLets_line.push_back(TraLet);
                    ObjectID_line.push_back(ObjLab_line[i][j]);

                    // save tracklet ID
                    TrackCheck_cur_line[j] = IDsofar_line;
                    IDsofar_line = IDsofar_line + 1;
                }
            }
            // frame i and i+1 (i>0)
            else
            {

                // check if there's association
                if (TemporalMatch_line[i][j]!=-1)
                {
                    // check the TrackID in previous frame
                    // if it is associated before, then add to existing tracklets.
                    // std:: cout << "Inside TrackCheck_pre_line " << TemporalMatch_line[i][j] << std::endl;
                    // std::cout << "TrackCheck_pre_line.size[i] " << TrackCheck_pre_line.size() << std::endl; 
                    if (TrackCheck_pre_line[TemporalMatch_line[i][j]]!=-1)
                    {
                        TrackLets_line[TrackCheck_pre_line[TemporalMatch_line[i][j]]].push_back(std::make_pair(i+1,j));
                        TrackCheck_cur_line[j] = TrackCheck_pre_line[TemporalMatch_line[i][j]];
                    }
                    // if not, insert new tracklets.
                    else
                    {
                        // first, save one tracklet consisting of two featureID
                        std::vector<std::pair<int, int> > TraLet(2);
                        TraLet[0] = std::make_pair(i,TemporalMatch_line[i][j]);
                        TraLet[1] = std::make_pair(i+1,j);
                        // then, save to the main tracklets list
                        TrackLets_line.push_back(TraLet);
                        ObjectID_line.push_back(ObjLab_line[i][j]);
                        // save tracklet ID
                        TrackCheck_cur_line[j] = IDsofar_line;
                        IDsofar_line = IDsofar_line + 1;
                    }
                }
            }
        }
        TrackCheck_pre = TrackCheck_cur;
        TrackCheck_pre_line = TrackCheck_cur_line;
    }

    // update object ID list
    mpMap->nObjID = ObjectID;
    mpMap->nObjID_line = ObjectID_line;
    
    // display info
    cout << endl;
    cout << "==============================================" << endl;
    cout << "the number of dynamic feature tracklets and for lines: " << TrackLets.size() << " " << TrackLets_line.size() << endl;
    cout << "==============================================" << endl;
    cout << endl;

    std::vector<int> TrackLength(N,0);
    for (int i = 0; i < TrackLets.size(); ++i)
        TrackLength[TrackLets[i].size()-2]++;

    // for (int i = 0; i < N; ++i){
    //     if(TrackLength[i]!=0)
    //         cout << "The length of " << i+2 << " tracklets is found with the amount of " << TrackLength[i] << " ..." << endl;
    // }
    // cout << endl;

    int LengthOver_5 = 0;
    ofstream save_track_distri;
    string save_td = "track_distribution.txt";
    save_track_distri.open(save_td.c_str(),ios::trunc);
    for (int i = 0; i < N; ++i){
        if(TrackLength[i]!=0)
            save_track_distri << TrackLength[i] << endl;
        if (i+2>=5)
            LengthOver_5 = LengthOver_5 + TrackLength[i];
    }
    save_track_distri.close();
    
        // Save tracklet lengths in histogram
    std::vector<int> TrackLength_line(N+1,0);
    for (int i = 0; i < TrackLets_line.size(); ++i)
        TrackLength_line[TrackLets_line[i].size()-1]++;

    save_td = "track_distribution_dynamic_line.txt";
    save_track_distri.open(save_td.c_str(),ios::trunc);
    for (int i = 0; i < N+1; ++i){
            save_track_distri << TrackLength_line[i] << endl;
    }
    save_track_distri.close();


    cout << "Length over 5 (DYNAMIC):::::::::::::::: " << LengthOver_5 << endl;

    return std::make_pair(TrackLets, TrackLets_line);
}

std::vector<std::vector<int> > Tracking::GetObjTrackTime(std::vector<std::vector<int> > &ObjTrackLab, std::vector<std::vector<int> > &ObjSemanticLab,
                                                         std::vector<std::vector<int> > &vnSMLabGT)
{
    std::vector<int> TrackCount(max_id-1,0);
    std::vector<int> TrackCountGT(max_id-1,0);
    std::vector<int> SemanticLabel(max_id-1,0);
    std::vector<std::vector<int> > ObjTrackTime;

    // count each object track
    for (int i = 0; i < ObjTrackLab.size(); ++i)
    {
        if (ObjTrackLab[i].size()<2)
            continue;

        for (int j = 1; j < ObjTrackLab[i].size(); ++j)
        {
            // TrackCountGT[ObjTrackLab[i][j]-1] = TrackCountGT[ObjTrackLab[i][j]-1] + 1;
            TrackCount[ObjTrackLab[i][j]-1] = TrackCount[ObjTrackLab[i][j]-1] + 1;
            SemanticLabel[ObjTrackLab[i][j]-1] = ObjSemanticLab[i][j];
        }
    }

    // count each object track in ground truth
    for (int i = 0; i < vnSMLabGT.size(); ++i)
    {
        for (int j = 0; j < vnSMLabGT[i].size(); ++j)
        {
            for (int k = 0; k < SemanticLabel.size(); ++k)
            {
                if (SemanticLabel[k]==vnSMLabGT[i][j])
                {
                    TrackCountGT[k] = TrackCountGT[k] + 1;
                    break;
                }
            }
        }
    }

    mpMap->nObjTraCount = TrackCount;
    mpMap->nObjTraCountGT = TrackCountGT;
    mpMap->nObjTraSemLab = SemanticLabel;


    // // // show the object track count
    // cout << "Current Object Track Counting: " << endl;
    // int TotalCount = 0;
    // for (int i = 0; i < TrackCount.size(); ++i)
    // {
    //     TotalCount = TotalCount + TrackCount[i];
    //     cout << "Object " << i+1 << " has been tracked " << TrackCount[i] << " times." << endl;
    // }
    // cout << "Total Object Track Counting: " << TotalCount << endl;

    // save to each frame the count number (ObjTrackTime)
    for (int i = 0; i < ObjTrackLab.size(); ++i)
    {
        std::vector<int> TrackTimeTmp(ObjTrackLab[i].size(),0);

        if (TrackTimeTmp.size()<2)
        {
            ObjTrackTime.push_back(TrackTimeTmp);
            continue;
        }

        for (int j = 1; j < TrackTimeTmp.size(); ++j)
        {
            TrackTimeTmp[j] = TrackCount[ObjTrackLab[i][j]-1];
        }
        ObjTrackTime.push_back(TrackTimeTmp);
    }

    return ObjTrackTime;
}

std::vector<std::vector<std::pair<int, int> > > Tracking::GetDynamicTrack()
{
    std::vector<std::vector<cv::KeyPoint> > Feats = mpMap->vpFeatDyn;
    std::vector<std::vector<int> > ObjLab = mpMap->vnFeatLabel;
    int N = Feats.size();

    // pair.first = frameID; pair.second = featureID;
    std::vector<std::vector<std::pair<int, int> > > TrackLets;
    // save object id of each tracklets
    std::vector<int> ObjectID;
    // save the track id in TrackLets for previous frame and current frame.
    std::vector<int> TrackCheck_pre;


    // main loop
    int IDsofar = 0;
    for (int i = 0; i < N; ++i)
    {
        // initialize TrackCheck
        std::vector<int> TrackCheck_cur(Feats[i].size(),-1);

        // Check empty
        if (Feats[i].empty())
        {
           TrackCheck_pre = TrackCheck_cur;
           continue;
        }

        // first pair of frames (frame 0 and 1)
        if (i==0)
        {
            int M = Feats[i].size();
            for (int j = 0; j < M; j=j+2)
            {
                // first, save one tracklet consisting of two featureID
                std::vector<std::pair<int, int> > TraLet(2);
                TraLet[0] = std::make_pair(i,j);
                TraLet[1] = std::make_pair(i,j+1); // used to be i+1
                // then, save to the main tracklets list
                TrackLets.push_back(TraLet);
                ObjectID.push_back(ObjLab[i][j/2]);

                // finally, save tracklet ID
                TrackCheck_cur[j+1] = IDsofar;
                IDsofar = IDsofar + 1;
            }
        }
        // frame i and i+1 (i>0)
        else
        {
            int M_pre = TrackCheck_pre.size();
            int M_cur = Feats[i].size();

            if (M_pre==0)
            {
                for (int j = 0; j < M_cur; j=j+2)
                {
                    // first, save one tracklet consisting of two featureID
                    std::vector<std::pair<int, int> > TraLet(2);
                    TraLet[0] = std::make_pair(i,j);
                    TraLet[1] = std::make_pair(i,j+1); // used to be i+1
                    // then, save to the main tracklets list
                    TrackLets.push_back(TraLet);
                    ObjectID.push_back(ObjLab[i][j/2]);

                    // finally, save tracklet ID
                    TrackCheck_cur[j+1] = IDsofar;
                    IDsofar = IDsofar + 1;
                }
            }
            else
            {
                // (1) find the temporal matching list (TM) between
                // previous flow locations and current sampled locations
                vector<int> TM(M_cur,-1);
                std::vector<float> MinDist(M_cur,-1);
                int nmatches = 0;
                for (int k = 1; k < M_pre; k=k+2)
                {
                    float x_ = Feats[i-1][k].pt.x;
                    float y_ = Feats[i-1][k].pt.y;
                    float min_dist = 10;
                    int candi = -1;
                    for (int j = 0; j < M_cur; j=j+2)
                    {
                        if (ObjLab[i-1][(k-1)/2]!=ObjLab[i][j/2])
                            continue;

                        float x  = Feats[i][j].pt.x;
                        float y  = Feats[i][j].pt.y;
                        float dist = std::sqrt( (x_-x)*(x_-x) + (y_-y)*(y_-y) );

                        if (dist<min_dist){
                            min_dist = dist;
                            candi = j;
                        }
                    }
                    // threshold
                    if (min_dist<1.0)
                    {
                        // current feature not occupied -or- occupied but new distance is smaller
                        // then label current match
                        if (TM[candi]==-1 || (TM[candi]!=-1 && min_dist<MinDist[candi]))
                        {
                            TM[candi] = k;
                            MinDist[candi] = min_dist;
                            nmatches = nmatches + 1;
                        }
                    }
                }

                // (2) save tracklets according to TM
                for (int j = 0; j < M_cur; j=j+2)
                {
                    // check the TM. if it is associated with last frame, then add to existing tracklets.
                    if (TM[j]!=-1)
                    {
                        TrackLets[TrackCheck_pre[TM[j]]].push_back(std::make_pair(i,j+1)); // used to be i+1
                        TrackCheck_cur[j+1] = TrackCheck_pre[TM[j]];
                    }
                    else
                    {
                        std::vector<std::pair<int, int> > TraLet(2);
                        TraLet[0] = std::make_pair(i,j);
                        TraLet[1] = std::make_pair(i,j+1); // used to be i+1
                        // then, save to the main tracklets list
                        TrackLets.push_back(TraLet);
                        ObjectID.push_back(ObjLab[i][j/2]);

                        // save tracklet ID
                        TrackCheck_cur[j+1] = IDsofar;
                        IDsofar = IDsofar + 1;
                    }
                }
            }
        }

        TrackCheck_pre = TrackCheck_cur;
    }

    // update object ID list
    mpMap->nObjID = ObjectID;


    // display info
    cout << endl;
    cout << "==============================================" << endl;
    cout << "the number of object feature tracklets: " << TrackLets.size() << endl;
    cout << "==============================================" << endl;
    cout << endl;

    std::vector<int> TrackLength(N,0);
    for (int i = 0; i < TrackLets.size(); ++i)
        TrackLength[TrackLets[i].size()-2]++;

    for (int i = 0; i < N; ++i)
        cout << "The length of " << i+2 << " tracklets is found with the amount of " << TrackLength[i] << " ..." << endl;
    cout << endl;


    return TrackLets;
}

void Tracking::RenewFrameInfo(const std::vector<int> &TM_sta, const std::vector<int> &TM_sta_line)
{
    cout << endl << "Start Renew Frame Information......" << endl;
    // ---------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++ Update for static features +++++++++++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------

    // use sampled or detected features
    int max_num_sta = nMaxTrackPointBG;
    int max_num_obj = nMaxTrackPointOBJ;

    //TODO: change that to parameter
    int max_num_sta_line = 400;
    std::vector<cv::KeyPoint> mvKeysTmp;
    std::vector<cv::KeyPoint> mvCorresTmp;
    std::vector<cv::Point2f> mvFlowNextTmp;
    std::vector<int> StaInlierIDTmp;

    std::vector<cv::line_descriptor::KeyLine> mvKeysTmp_line;
    std::vector<cv::line_descriptor::KeyLine> mvCorresTmp_line;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> mvFlowNextTmp_line;
    std::vector<int> StaInlierIDTmp_line;


    // (1) Save the inliers from last frame
    for (int i = 0; i < TM_sta.size(); ++i)
    {
        //in the optimisation if the chi error is tool large we set the temperalmatch = -1
        if (TM_sta[i]==-1)
            continue;

        int x = mCurrentFrame.mvStatKeys[TM_sta[i]].pt.x;
        int y = mCurrentFrame.mvStatKeys[TM_sta[i]].pt.y;

        if (x>=mImGrayLast.cols || y>=mImGrayLast.rows || x<=0 || y<=0)
            continue;

        if (mSegMap.at<int>(y,x)!=0)
            continue;

        if (mDepthMap.at<float>(y,x)>40 || mDepthMap.at<float>(y,x)<=0)
            continue;

        float flow_xe = mFlowMap.at<cv::Vec2f>(y,x)[0];
        float flow_ye = mFlowMap.at<cv::Vec2f>(y,x)[1];

        if(flow_xe!=0 && flow_ye!=0)
        {
            if(mCurrentFrame.mvStatKeys[TM_sta[i]].pt.x+flow_xe < mImGrayLast.cols && mCurrentFrame.mvStatKeys[TM_sta[i]].pt.y+flow_ye < mImGrayLast.rows && mCurrentFrame.mvStatKeys[TM_sta[i]].pt.x+flow_xe>0 && mCurrentFrame.mvStatKeys[TM_sta[i]].pt.y+flow_ye>0)
            {
                mvKeysTmp.push_back(mCurrentFrame.mvStatKeys[TM_sta[i]]);
                mvCorresTmp.push_back(cv::KeyPoint(mCurrentFrame.mvStatKeys[TM_sta[i]].pt.x+flow_xe,mCurrentFrame.mvStatKeys[TM_sta[i]].pt.y+flow_ye,0,0,0,-1));
                mvFlowNextTmp.push_back(cv::Point2f(flow_xe,flow_ye));
                StaInlierIDTmp.push_back(TM_sta[i]);
            }
        }

        if (mvKeysTmp.size()>max_num_sta)
            break;
    }
    //now for lines
    for (int i=0; i < TM_sta_line.size(); ++i)
    {
        if (TM_sta_line[i] == -1)
            continue;

        int x_start = mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].startPointX;
        int y_start = mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].startPointY;
        int x_end = mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].endPointX;
        int y_end = mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].endPointY;

        if (fabs(x_start - x_end) < 1e-6 && fabs(y_start - y_end) < 1e-6)
            continue;

        if (x_start>=mImGrayLast.cols || y_start>=mImGrayLast.rows || x_start<=0 || y_start<=0)
            continue;
        
        if (x_end>=mImGrayLast.cols || y_end>=mImGrayLast.rows || x_end<=0 || y_end<=0)
            continue;

        if (mSegMap.at<int>(y_start,x_start)!=0 || mSegMap.at<int>(y_end,x_end)!=0)
            continue;

        if (mDepthMap.at<float>(y_start, x_start)>40 || mDepthMap.at<float>(y_start, x_start)<=0)
            continue;

        if (mDepthMap.at<float>(y_end, x_end)>40 || mDepthMap.at<float>(y_end, x_end)<=0)
            continue;

        float depthStart = mDepthMap.at<float>(y_start, x_start);
        float depthEnd = mDepthMap.at<float>(y_end, x_end);
        //mid point
        int xm = (x_start + x_end) / 2;
        int ym = (y_start + y_end) / 2;

        float depthMid = mDepthMap.at<float>(ym, xm);
        float depthMidExpected = (depthStart + depthEnd) / 2;
        float baseThreshold = 10.0;  // This is just an example value
        float lineLength = sqrt(pow(x_start - x_end, 2) + pow(y_start - y_end, 2));
        // Adjust the threshold based on the length of the line
        float threshold = baseThreshold * (lineLength / 1000);
        if (abs(depthMid - depthMidExpected) > threshold)
            continue;

        float flow_xe_start = mFlowMap.at<cv::Vec2f>(y_start,x_start)[0];
        float flow_ye_start = mFlowMap.at<cv::Vec2f>(y_start,x_start)[1];
        float flow_xe_end = mFlowMap.at<cv::Vec2f>(y_end,x_end)[0];
        float flow_ye_end = mFlowMap.at<cv::Vec2f>(y_end,x_end)[1];

        if (flow_xe_start != 0 && flow_ye_start != 0 && flow_xe_end != 0 && flow_ye_end != 0)
        {
            if (mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].startPointX+flow_xe_start < mImGrayLast.cols && mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].startPointY+flow_ye_start < mImGrayLast.rows && mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].startPointX+flow_xe_start > 0 && mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].startPointY+flow_ye_start > 0 && mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].endPointX+flow_xe_end < mImGrayLast.cols && mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].endPointY+flow_ye_end < mImGrayLast.rows && mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].endPointX+flow_xe_end > 0 && mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].endPointY+flow_ye_end > 0)
            {
                mvKeysTmp_line.push_back(mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]]);
                cv::line_descriptor::KeyLine corr_line;
                corr_line.startPointX = mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].startPointX+flow_xe_start;
                corr_line.startPointY = mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].startPointY+flow_ye_start;
                corr_line.endPointX = mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].endPointX+flow_xe_end;
                corr_line.endPointY = mCurrentFrame.mvStatKeys_Line[TM_sta_line[i]].endPointY+flow_ye_end;
                mvCorresTmp_line.push_back(corr_line);
                mvFlowNextTmp_line.push_back(std::make_pair(cv::Point2f(flow_xe_start,flow_ye_start),cv::Point2f(flow_xe_end,flow_ye_end)));
                StaInlierIDTmp_line.push_back(TM_sta_line[i]);
            }
        }
        if (mvKeysTmp_line.size() > max_num_sta_line)
            break;
    }

    //std::cout << "mvKeysTmp_line for debug " << mvKeysTmp_line.size() << std::endl;

    // cout << "accumulate static inlier number in: " << mvKeysTmp.size() << endl;

    // (2) Save extra key points to make it a fixed number (max = 1000, 1600)
    int tot_num = mvKeysTmp.size(), start_id = 0, step = 10;
    std::vector<cv::KeyPoint> mvKeysTmpCheck = mvKeysTmp;
    std::vector<cv::KeyPoint> mvKeysSample;
    if (nUseSampleFea==1)
        mvKeysSample = mCurrentFrame.mvStatKeysTmp;
    else
        mvKeysSample = mCurrentFrame.mvKeys;
    while (tot_num<max_num_sta)
    {
        // start id > step number, then stop
        if (start_id==step)
            break;

        for (int i = start_id; i < mvKeysSample.size(); i=i+step)
        {
            // check if this key point is already been used
            float min_dist = 100;
            bool used = false;
            for (int j = 0; j < mvKeysTmpCheck.size(); ++j)
            {
                float cur_dist = std::sqrt( (mvKeysTmpCheck[j].pt.x-mvKeysSample[i].pt.x)*(mvKeysTmpCheck[j].pt.x-mvKeysSample[i].pt.x) + (mvKeysTmpCheck[j].pt.y-mvKeysSample[i].pt.y)*(mvKeysTmpCheck[j].pt.y-mvKeysSample[i].pt.y) );
                if (cur_dist<min_dist)
                    min_dist = cur_dist;
                if (min_dist<1.0)
                {
                    used = true;
                    break;
                }
            }
            if (used)
                continue;

            int x = mvKeysSample[i].pt.x;
            int y = mvKeysSample[i].pt.y;

            if (x>=mImGrayLast.cols || y>=mImGrayLast.rows || x<=0 || y<=0)
                continue;

            if (mSegMap.at<int>(y,x)!=0)
                continue;

            if (mDepthMap.at<float>(y,x)>40 || mDepthMap.at<float>(y,x)<=0)
                continue;

            float flow_xe = mFlowMap.at<cv::Vec2f>(y,x)[0];
            float flow_ye = mFlowMap.at<cv::Vec2f>(y,x)[1];

            if(flow_xe!=0 && flow_ye!=0)
            {
                if(mvKeysSample[i].pt.x+flow_xe < mImGrayLast.cols && mvKeysSample[i].pt.y+flow_ye < mImGrayLast.rows && mvKeysSample[i].pt.x+flow_xe > 0 && mvKeysSample[i].pt.y+flow_ye > 0)
                {
                    mvKeysTmp.push_back(mvKeysSample[i]);
                    mvCorresTmp.push_back(cv::KeyPoint(mvKeysSample[i].pt.x+flow_xe,mvKeysSample[i].pt.y+flow_ye,0,0,0,-1));
                    mvFlowNextTmp.push_back(cv::Point2f(flow_xe,flow_ye));
                    StaInlierIDTmp.push_back(-1);
                    tot_num = tot_num + 1;
                }
            }

            if (tot_num>=max_num_sta)
                break;
        }
        start_id = start_id + 1;
    }

    //Save extra key lines to make it a fixed number

    int tot_num_line = mvKeysTmp_line.size(), start_id_line = 0;
    step = 1;

    std::vector<cv::line_descriptor::KeyLine> mvKeysTmpCheck_line;
    mvKeysTmpCheck_line = mvKeysTmp_line;
    std::vector<cv::line_descriptor::KeyLine> mvKeysSample_line;
    mvKeysSample_line = mCurrentFrame.mvStatKeysLineTmp;
    
    int linesChecked = 0;    
    //TODO: change this to a parameter
    while (tot_num_line < max_num_sta_line && linesChecked < mvKeysSample_line.size())
    {
        if (start_id_line == step)
            continue;

        for (int i = start_id_line; i<mvKeysSample_line.size(); i = i + step)
        {
            linesChecked++;
            bool used = false;
            //Check if this key_line is already used. I will check the difference in agles and the distance of the middle point of the line
            cv::Point2f dir1 = mvKeysSample_line[i].getEndPoint() - mvKeysSample_line[i].getStartPoint();
            cv::Point2f midpoint1 = (mvKeysSample_line[i].getEndPoint() + mvKeysSample_line[i].getStartPoint()) * 0.5;
            float lenght1 = std::sqrt(dir1.x*dir1.x + dir1.y*dir1.y);
            for (int j = 0; j < mvKeysTmpCheck_line.size(); ++j)
            {
                cv::Point2f dir2 = mvKeysTmpCheck_line[j].getEndPoint() - mvKeysTmpCheck_line[j].getStartPoint();
                float length2 = std::sqrt(dir2.x*dir2.x + dir2.y*dir2.y);
                float product = dir1.x*dir2.x + dir1.y*dir2.y;
                //product/(length1*length2) = cos(angle_diff)

                // //if angle is > 180 degrees put it there
                // if (angle_diff > CV_PI)
                //     angle_diff = 2 * CV_PI - angle_diff;
                
                cv::Point2f midpoint2 = (mvKeysTmpCheck_line[j].getEndPoint() + mvKeysTmpCheck_line[j].getStartPoint()) * 0.5;

                float line_dist = std::sqrt((midpoint1.x - midpoint2.x)*(midpoint1.x - midpoint2.x)+(midpoint1.y - midpoint2.y) * (midpoint1.y - midpoint2.y));

                if (product/(lenght1 * length2) > cos(CV_PI / 30)  && line_dist < std::max(lenght1, length2) * 0.5f)
                    {
                        used = true;
                        //std::cout << "Found a used line" << std::endl;
                        break;
                    }
            }
            if (used)
                continue;
            int x_start = mvKeysSample_line[i].startPointX, y_start = mvKeysSample_line[i].startPointY;
            int x_end = mvKeysSample_line[i].endPointX, y_end = mvKeysSample_line[i].endPointY;


            float depthStart = mDepthMap.at<float>(y_start, x_start);
            float depthEnd = mDepthMap.at<float>(y_end, x_end);
            //mid point
            int xm = (x_start + x_end) / 2;
            int ym = (y_start + y_end) / 2;

            float depthMid = mDepthMap.at<float>(ym, xm);
            float depthMidExpected = (depthStart + depthEnd) / 2;
            float baseThreshold = 10.0;  // This is just an example value
            float lineLength = sqrt(pow(x_start - x_end, 2) + pow(y_start - y_end, 2));
            // Adjust the threshold based on the length of the line
            float threshold = baseThreshold * (lineLength / 1000);
            if (abs(depthMid - depthMidExpected) > threshold)
                continue;

            if (x_start>=mImGrayLast.cols || x_end >= mImGrayLast.cols || y_start >= mImGrayLast.rows || y_end >= mImGrayLast.rows || x_start <= 0 || x_end <= 0 || y_start <= 0 || y_end <= 0)
                continue;

            if (mSegMap.at<int>(y_start, x_start) != 0 || mSegMap.at<int>(y_end, x_end)!=0)
                continue;

            if (mDepthMap.at<float>(y_start, x_start) <= 0 || mDepthMap.at<float>(y_end, x_end) <= 0 || mDepthMap.at<float>(y_start, x_start) > 40 || mDepthMap.at<float>(y_end, x_end) > 40)
                continue;

            float flow_xe_start = mFlowMap.at<cv::Vec2f>(y_start, x_start)[0];
            float flow_ye_start = mFlowMap.at<cv::Vec2f>(y_start, x_start)[1];
            float flow_xe_end = mFlowMap.at<cv::Vec2f>(y_end, x_end)[0];
            float flow_ye_end = mFlowMap.at<cv::Vec2f>(y_end, x_end)[1];

            if (flow_xe_start!=0 || flow_ye_start!=0 || flow_xe_end!=0 || flow_ye_end!=0)
            {
                if (x_start + flow_xe_start < mImGrayLast.cols && y_start + flow_ye_start < mImGrayLast.rows && x_start + flow_xe_start > 0 && y_start + flow_ye_start > 0 && x_end + flow_xe_end < mImGrayLast.cols && y_end + flow_ye_end < mImGrayLast.rows && x_end + flow_xe_end > 0 && y_end + flow_ye_end > 0)
                {
                    //If all these conditions are met we are adding the lines to be in the next computation
                    mvKeysTmp_line.push_back(mvKeysSample_line[i]);
                    cv::line_descriptor::KeyLine corr_line;
                    corr_line.startPointX = x_start + flow_xe_start;
                    corr_line.startPointY = y_start + flow_ye_start;
                    corr_line.endPointX = x_end + flow_xe_end;
                    corr_line.endPointY = y_end + flow_ye_end;
                    mvCorresTmp_line.push_back(corr_line);
                    mvFlowNextTmp_line.push_back(std::make_pair(cv::Point2f(flow_xe_start,flow_ye_start),cv::Point2f(flow_xe_end,flow_ye_end)));
                    StaInlierIDTmp_line.push_back(-1);
                    tot_num_line++;
                }
            }
            if (tot_num_line>=max_num_sta_line)
                break;
        }
        start_id_line = start_id_line + 1;

    }

    mCurrentFrame.N_s_tmp = mvKeysTmp.size();
    mCurrentFrame.N_s_line_tmp = mvKeysTmp_line.size();

    //std::cout << "mvKeysTmp for debugging after the addition of current lines " << mvKeysTmp_line.size() << std::endl;   
    //print mvKeysTmp_line
    // std::cout << "mvKeysTmp_line for debugging " << std::endl;
    // for (int i = 0; i < mvKeysTmp_line.size(); ++i)
    // {
    //     std::cout << "Line " << i << " start point: " << mvKeysTmp_line[i].startPointX << " " << mvKeysTmp_line[i].startPointY << " end point: " << mvKeysTmp_line[i].endPointX << " " << mvKeysTmp_line[i].endPointY << std::endl;
    // }

    // (3) assign the depth value to each key point
    std::vector<float> mvDepthTmp(mCurrentFrame.N_s_tmp,-1);
    for(int i=0; i<mCurrentFrame.N_s_tmp; i++)
    {
        const cv::KeyPoint &kp = mvKeysTmp[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        float d = mDepthMap.at<float>(v,u); // be careful with the order  !!!

        if(d>0)
            mvDepthTmp[i] = d;
    }

    std::vector<std::pair<float, float>> mvDepthTmp_line(mCurrentFrame.N_s_line_tmp, std::make_pair(-1,-1));
    for (int i = 0; i < mCurrentFrame.N_s_line_tmp; i++)
    {
        const cv::line_descriptor::KeyLine &kl = mvKeysTmp_line[i];

        const float &v_start = kl.startPointY;
        const float &u_start = kl.startPointX;
        const float &v_end = kl.endPointY;
        const float &u_end = kl.endPointX;

        float d_start = mDepthMap.at<float>(v_start, u_start);
        float d_end = mDepthMap.at<float>(v_end, u_end);

        if (d_start > 0 && d_end > 0)
            mvDepthTmp_line[i] = std::make_pair(d_start, d_end);
    }

    // (4) create 3d point based on key point, depth and pose
    std::vector<cv::Mat> mv3DPointTmp(mCurrentFrame.N_s_tmp);
    for (int i = 0; i < mCurrentFrame.N_s_tmp; ++i)
    {
        mv3DPointTmp[i] = Optimizer::Get3DinWorld(mvKeysTmp[i], mvDepthTmp[i], mK, Converter::toInvMatrix(mCurrentFrame.mTcw));
    }

    std::vector<std::pair<cv::Mat, cv::Mat>> mv3DLineTmp(mCurrentFrame.N_s_line_tmp);
    for (int i = 0; i < mCurrentFrame.N_s_line_tmp; ++i)
    {
        mv3DLineTmp[i] = Optimizer::Get3DinWorld_line(mvKeysTmp_line[i], mvDepthTmp_line[i], mK, Converter::toInvMatrix(mCurrentFrame.mTcw));
    }

    //CREATE 3D PLOT OF THE LINES
    // cv::viz::Viz3d window("Coordinate Frame");
    // window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    // for (int i = 0; i < mv3DLineTmp.size(); ++i)
    // {
    //     cv::Point3f point1(mv3DLineTmp[i].first.at<float>(0), mv3DLineTmp[i].first.at<float>(1), mv3DLineTmp[i].first.at<float>(2));
    //     cv::Point3f point2(mv3DLineTmp[i].second.at<float>(0), mv3DLineTmp[i].second.at<float>(1), mv3DLineTmp[i].second.at<float>(2));
        
    //     cv::viz::WLine line_widget(point1, point2, cv::viz::Color::green());
    //     window.showWidget("line"+std::to_string(i), line_widget);
    // }

    // while(!window.wasStopped())
    // {
    //     window.spinOnce(1, true);
    //     int key = cv::waitKey(1);
    //     if(key == 'q' || key == 'Q')  // Press 'q' or 'Q' to continue
    //         break;
    // }


    // //Draw mvKeysTmp_line
    // cv::Mat img = mCurrentFrame.imGray_.clone();
    // for (int i = 0; i < mvKeysTmp_line.size(); ++i)
    // {
    //     cv::line(img, cv::Point(mvKeysTmp_line[i].startPointX, mvKeysTmp_line[i].startPointY), cv::Point(mvKeysTmp_line[i].endPointX, mvKeysTmp_line[i].endPointY), cv::Scalar(0, 255, 0), 2);
    // }
    // cv::imshow("mvKeysTmp_line", img);
    // cv::waitKey(0);





    // Obtain inlier ID
    mCurrentFrame.nStaInlierID = StaInlierIDTmp;

    // Update
    mCurrentFrame.mvStatKeysTmp = mvKeysTmp;
    mCurrentFrame.mvStatDepthTmp = mvDepthTmp;
    mCurrentFrame.mvStat3DPointTmp = mv3DPointTmp;
    mCurrentFrame.mvFlowNext = mvFlowNextTmp;
    mCurrentFrame.mvCorres = mvCorresTmp;

    //Now for lines
    // Obtain inlier ID
    mCurrentFrame.nStaInlierID_line = StaInlierIDTmp_line;

    //Update
    mCurrentFrame.mvStatKeysLineTmp = mvKeysTmp_line;
    mCurrentFrame.mvStatDepthLineTmp = mvDepthTmp_line;
    mCurrentFrame.mvStat3DLineTmp = mv3DLineTmp;
    mCurrentFrame.mvFlowNext_Line = mvFlowNextTmp_line;
    mCurrentFrame.mvCorresLine = mvCorresTmp_line;

    // cout << "updating STATIC features finished...... " << mvKeysTmp.size() << endl;

    // ---------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++ Update for Dynamic Object Features +++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------

    std::vector<cv::KeyPoint> mvObjKeysTmp;
    std::vector<float> mvObjDepthTmp;
    std::vector<cv::KeyPoint> mvObjCorresTmp;
    std::vector<cv::Point2f> mvObjFlowNextTmp;
    std::vector<int> vSemObjLabelTmp;
    std::vector<int> DynInlierIDTmp;
    std::vector<int> vObjLabelTmp;

    // (1) Again, save the inliers from last frame
    std::vector<std::vector<int> > ObjInlierSet = mCurrentFrame.vnObjInlierID;
    std::vector<int> ObjFeaCount(ObjInlierSet.size());

    //for lines
    std::vector<cv::line_descriptor::KeyLine> mvObjKeysTmp_line;
    std::vector<std::pair<float, float>> mvObjDepthTmp_line;
    std::vector<cv::line_descriptor::KeyLine> mvObjCorresTmp_line;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> mvObjFlowNextTmp_line;
    std::vector<int> vSemObjLabelTmp_line;
    std::vector<int> DynInlierIDTmp_line;
    std::vector<int> vObjLabelTmp_line;

    std::vector<std::vector<int>> ObjInlierSet_line = mCurrentFrame.vnObjInlierID_line;
    std::vector<int> ObjFeaCount_line(ObjInlierSet_line.size());

    //for each inlier object:
    for (int i = 0; i < ObjInlierSet.size(); ++i)
    {
        // remove failure object
        if (!mCurrentFrame.bObjStat[i])
        {
            ObjFeaCount[i] = -1;
            ObjFeaCount_line[i] = -1;
            continue;
        }

        int count = 0;
        for (int j = 0; j < ObjInlierSet[i].size(); ++j)
        {
            const int x = mCurrentFrame.mvObjKeys[ObjInlierSet[i][j]].pt.x;
            const int y = mCurrentFrame.mvObjKeys[ObjInlierSet[i][j]].pt.y;

            if (x>=mImGrayLast.cols || y>=mImGrayLast.rows || x<=0 || y<=0)
                continue;

            if (mSegMap.at<int>(y,x)!=0 && mDepthMap.at<float>(y,x)<25 && mDepthMap.at<float>(y,x)>0)
            {
                const float flow_x = mFlowMap.at<cv::Vec2f>(y,x)[0];
                const float flow_y = mFlowMap.at<cv::Vec2f>(y,x)[1];

                if (x+flow_x < mImGrayLast.cols && y+flow_y < mImGrayLast.rows && x+flow_x>0 && y+flow_y>0)
                {
                    mvObjKeysTmp.push_back(cv::KeyPoint(x,y,0,0,0,-1));
                    mvObjDepthTmp.push_back(mDepthMap.at<float>(y,x));
                    vSemObjLabelTmp.push_back(mSegMap.at<int>(y,x));
                    mvObjFlowNextTmp.push_back(cv::Point2f(flow_x,flow_y));
                    mvObjCorresTmp.push_back(cv::KeyPoint(x+flow_x,y+flow_y,0,0,0,-1));
                    DynInlierIDTmp.push_back(ObjInlierSet[i][j]);
                    vObjLabelTmp.push_back(mCurrentFrame.vObjLabel[ObjInlierSet[i][j]]);
                    count = count + 1;
                }
            }
        }

        int count_line = 0;
        for (int j =0; j < ObjInlierSet_line[i].size(); ++j)
        {
            const int x_start = mCurrentFrame.mvObjKeys_Line[ObjInlierSet_line[i][j]].startPointX;
            const int y_start = mCurrentFrame.mvObjKeys_Line[ObjInlierSet_line[i][j]].startPointY;
            const int x_end = mCurrentFrame.mvObjKeys_Line[ObjInlierSet_line[i][j]].endPointX;
            const int y_end = mCurrentFrame.mvObjKeys_Line[ObjInlierSet_line[i][j]].endPointY;  

            if (x_start >= mImGrayLast.cols || x_end >= mImGrayLast.cols || y_start >= mImGrayLast.rows || y_end >= mImGrayLast.rows || x_start <= 0 || x_end <= 0 || y_start <= 0 || y_end <= 0)
                continue;
            if (fabs(x_start - x_end) < 1e-6 && fabs(y_start - y_end) < 1e-6)
                continue;

            if (mSegMap.at<int>(y_start, x_start)!=0 && mSegMap.at<int>(y_end, x_end)!=0 && mDepthMap.at<float>(y_start, x_start)<25 && mDepthMap.at<float>(y_start, x_start)>0 && mDepthMap.at<float>(y_end, x_end)<25 && mDepthMap.at<float>(y_end, x_end)>0)
            {
                const float flow_x_start = mFlowMap.at<cv::Vec2f>(y_start, x_start)[0];
                const float flow_y_start = mFlowMap.at<cv::Vec2f>(y_start, x_start)[1];
                const float flow_x_end = mFlowMap.at<cv::Vec2f>(y_end, x_end)[0];
                const float flow_y_end = mFlowMap.at<cv::Vec2f>(y_end, x_end)[1];
                if (x_start + flow_x_start < mImGrayLast.cols && y_start + flow_y_start < mImGrayLast.rows && x_start + flow_x_start > 0 && y_start + flow_y_start > 0 && x_end + flow_x_end < mImGrayLast.cols && y_end + flow_y_end < mImGrayLast.rows && x_end + flow_x_end > 0 && y_end + flow_y_end > 0)
                {
                    cv::line_descriptor::KeyLine tmp_line;
                    tmp_line.startPointX = x_start;
                    tmp_line.startPointY = y_start;
                    tmp_line.endPointX = x_end;
                    tmp_line.endPointY = y_end;
                    mvObjKeysTmp_line.push_back(tmp_line);
                    mvObjDepthTmp_line.push_back(std::make_pair(mDepthMap.at<float>(y_start, x_start), mDepthMap.at<float>(y_end, x_end)));
                    //Assuming that the start and end point have the same label
                    vSemObjLabelTmp_line.push_back(mSegMap.at<int>(y_start, x_start));
                    mvObjFlowNextTmp_line.push_back(std::make_pair(cv::Point2f(flow_x_start,flow_y_start),cv::Point2f(flow_x_end,flow_y_end)));
                    cv::line_descriptor::KeyLine corr_line;
                    corr_line.startPointX = x_start + flow_x_start;
                    corr_line.startPointY = y_start + flow_y_start;
                    corr_line.endPointX = x_end + flow_x_end;

                    corr_line.endPointY = y_end + flow_y_end;
                    mvObjCorresTmp_line.push_back(corr_line);

                    DynInlierIDTmp_line.push_back(ObjInlierSet_line[i][j]);
                    vObjLabelTmp_line.push_back(mCurrentFrame.vObjLabel_Line[ObjInlierSet_line[i][j]]);
                    count_line = count_line + 1;
                }
            }
        }


        ObjFeaCount[i] = count;
        ObjFeaCount_line[i] = count_line;
        // cout << "accumulate dynamic inlier number: " << ObjFeaCount[i] << endl;
    }


    // (2) Save extra key points to make each object having a fixed number (max = 400, 800, 1000)
    std::vector<std::vector<int> > ObjSet = mCurrentFrame.vnObjID;

    std::vector<cv::KeyPoint> mvObjKeysTmpCheck = mvObjKeysTmp;
    std::vector<cv::line_descriptor::KeyLine> mvObjKeysTmpCheck_line = mvObjKeysTmp_line;
    for (int i = 0; i < ObjSet.size(); ++i)
    {
        // remove failure object
        if (!mCurrentFrame.bObjStat[i])
            continue;

        int SemLabel = mCurrentFrame.nSemPosition[i];
        int tot_num = ObjFeaCount[i];
        int start_id = 0, step = 15;
        while (tot_num<max_num_obj)
        {
            // start id > step number, then stop
            if (start_id==step){
                // cout << "run on all the original objset... tot_num: " << tot_num << endl;
                break;
            }

            for (int j = start_id; j < mvTmpSemObjLabel.size(); j=j+step)
            {
                // check the semantic label if it is the same
                if (mvTmpSemObjLabel[j]!=SemLabel)
                    continue;

                // check if this key point is already been used
                float min_dist = 100;
                bool used = false;
                for (int k = 0; k < mvObjKeysTmpCheck.size(); ++k)
                {
                    float cur_dist = std::sqrt( (mvObjKeysTmpCheck[k].pt.x-mvTmpObjKeys[j].pt.x)*(mvObjKeysTmpCheck[k].pt.x-mvTmpObjKeys[j].pt.x) + (mvObjKeysTmpCheck[k].pt.y-mvTmpObjKeys[j].pt.y)*(mvObjKeysTmpCheck[k].pt.y-mvTmpObjKeys[j].pt.y) );
                    if (cur_dist<min_dist)
                        min_dist = cur_dist;
                    if (min_dist<1.0)
                    {
                        used = true;
                        break;
                    }
                }
                if (used)
                    continue;

                // save the found one
                mvObjKeysTmp.push_back(mvTmpObjKeys[j]);
                mvObjDepthTmp.push_back(mvTmpObjDepth[j]);
                vSemObjLabelTmp.push_back(mvTmpSemObjLabel[j]);
                mvObjFlowNextTmp.push_back(mvTmpObjFlowNext[j]);
                mvObjCorresTmp.push_back(mvTmpObjCorres[j]);
                DynInlierIDTmp.push_back(-1);
                vObjLabelTmp.push_back(mCurrentFrame.nModLabel[i]);
                tot_num = tot_num + 1;

                if (tot_num>=max_num_obj){
                    // cout << "reach max_num_obj... tot_num: " << tot_num << endl;
                    break;
                }
            }
            start_id = start_id + 1;
        }

        int tot_num_line = ObjFeaCount_line[i];
        start_id = 0; step = 2;
        int max_num_obj_line = 100;
        linesChecked = 0;
        while (tot_num_line <  max_num_obj_line && linesChecked < mvTmpSemObjLabel_line.size())
        {
            if (start_id==step){
                break;
            }

            for (int j = start_id; j < mvTmpSemObjLabel_line.size(); j=j+step)
            {
                linesChecked++;
                // check the semantic label if it is the same
                if (mvTmpSemObjLabel_line[j]!=SemLabel)
                    continue;
                if (fabs(mvTmpObjKeys_line[j].startPointX - mvTmpObjKeys_line[j].endPointX) < 1e-6 && fabs(mvTmpObjKeys_line[j].startPointY - mvTmpObjKeys_line[j].endPointY) < 1e-6)
                    continue;

                // check if this key point is already been used
                bool used = false;
                cv::Point2f dir1 = mvTmpObjKeys_line[j].getEndPoint() - mvTmpObjKeys_line[j].getStartPoint();
                cv::Point2f midpoint1 = (mvTmpObjKeys_line[j].getEndPoint() + mvTmpObjKeys_line[j].getStartPoint()) * 0.5;

                for (int k = 0; k < mvObjKeysTmpCheck_line.size(); ++k)
                {
                    cv::Point2f dir2 = mvObjKeysTmpCheck_line[k].getEndPoint() - mvObjKeysTmpCheck_line[k].getStartPoint();
                    float angle_diff = fabs(atan2(dir1.y, dir1.x) - atan2(dir2.y, dir2.x));
                    //if angle is > 180 degrees put it there
                    if (angle_diff > CV_PI)
                        angle_diff = 2 * CV_PI - angle_diff;
                    
                    cv::Point2f midpoint2 = (mvObjKeysTmpCheck_line[k].getEndPoint() + mvObjKeysTmpCheck_line[k].getStartPoint()) * 0.5;

                    float line_dist = std::sqrt((midpoint1.x - midpoint2.x)*(midpoint1.x - midpoint2.x)+(midpoint1.y - midpoint2.y) * (midpoint1.y - midpoint2.y));

                    if (angle_diff < 1.0 && line_dist < 1)
                    {
                        used = true;
                        break;
                    }
                }
                if (used)
                    continue;

                // save the found one
                mvObjKeysTmp_line.push_back(mvTmpObjKeys_line[j]);
                mvObjDepthTmp_line.push_back(mvTmpObjDepth_line[j]);
                vSemObjLabelTmp_line.push_back(mvTmpSemObjLabel_line[j]);
                mvObjFlowNextTmp_line.push_back(mvTmpObjFlowNext_line[j]);
                mvObjCorresTmp_line.push_back(mvTmpObjCorres_line[j]);
                DynInlierIDTmp_line.push_back(-1);
                vObjLabelTmp_line.push_back(mCurrentFrame.nModLabel[i]);
                tot_num_line = tot_num_line + 1;

                if (tot_num_line>=max_num_obj_line){
                    break;
                }
            }
            
            start_id = start_id + 1;

        }


    }
 
    // (3) Update new appearing objects
    // (3.1) find the unique labels in semantic label
    auto UniLab = mvTmpSemObjLabel;
    std::sort(UniLab.begin(), UniLab.end());
    UniLab.erase(std::unique( UniLab.begin(), UniLab.end() ), UniLab.end() );
    // (3.2) find new appearing label
    std::vector<bool> NewLab(UniLab.size(),false);
    for (int i = 0; i < mCurrentFrame.nSemPosition.size(); ++i)
    {
        int CurSemLabel = mCurrentFrame.nSemPosition[i];
        for (int j = 0; j < UniLab.size(); ++j)
        {
            if (UniLab[j]==CurSemLabel && mCurrentFrame.bObjStat[i]) // && mCurrentFrame.bObjStat[i]
            {
                //if the label is already in the list, then set it to true, so is not a neww object
                NewLab[j] = true;
                break;
            }
        }

    }
    // (3.3) add the new object key points
    for (int i = 0; i < NewLab.size(); ++i)
    {
        if (NewLab[i]==false)
        {
            for (int j = 0; j < mvTmpSemObjLabel.size(); j++)
            {
                if (UniLab[i]==mvTmpSemObjLabel[j])
                {
                    // save the found one
                    mvObjKeysTmp.push_back(mvTmpObjKeys[j]);
                    mvObjDepthTmp.push_back(mvTmpObjDepth[j]);
                    vSemObjLabelTmp.push_back(mvTmpSemObjLabel[j]);
                    mvObjFlowNextTmp.push_back(mvTmpObjFlowNext[j]);
                    mvObjCorresTmp.push_back(mvTmpObjCorres[j]);
                    DynInlierIDTmp.push_back(-1);
                    vObjLabelTmp.push_back(-2);
                }
            }
            //check also if there are lines with new labels
            for (int j = 0; j < mvTmpSemObjLabel_line.size(); j++)
            {
                if (UniLab[i]==mvTmpSemObjLabel_line[j])
                {
                    if (fabs(mvTmpObjKeys_line[j].startPointX - mvTmpObjKeys_line[j].endPointX) < 1e-6 && fabs(mvTmpObjKeys_line[j].startPointY - mvTmpObjKeys_line[j].endPointY) < 1e-6)
                        continue;
                    mvObjKeysTmp_line.push_back(mvTmpObjKeys_line[j]);
                    mvObjDepthTmp_line.push_back(mvTmpObjDepth_line[j]);
                    vSemObjLabelTmp_line.push_back(mvTmpSemObjLabel_line[j]);
                    mvObjFlowNextTmp_line.push_back(mvTmpObjFlowNext_line[j]);
                    mvObjCorresTmp_line.push_back(mvTmpObjCorres_line[j]);
                    DynInlierIDTmp_line.push_back(-1);
                    vObjLabelTmp_line.push_back(-2);
                }
            }
        }
    }
   //visualize the lines added
    // cv::Mat img = mCurrentFrame.imGray_.clone();
    // for (int i = 0; i < mvObjKeysTmp_line.size(); ++i)
    // {
    //     cv::line(img, cv::Point(mvObjKeysTmp_line[i].startPointX, mvObjKeysTmp_line[i].startPointY), cv::Point(mvObjKeysTmp_line[i].endPointX, mvObjKeysTmp_line[i].endPointY), cv::Scalar(0, 255, 0), 2);
    // }
    // cv::imshow("mvObjKeysTmp_line", img);
    //cv::waitKey(0);

    // (4) create 3d point based on key point, depth and pose
    std::vector<cv::Mat> mvObj3DPointTmp(mvObjKeysTmp.size());
    for (int i = 0; i < mvObjKeysTmp.size(); ++i)
        mvObj3DPointTmp[i] = Optimizer::Get3DinWorld(mvObjKeysTmp[i], mvObjDepthTmp[i], mK, Converter::toInvMatrix(mCurrentFrame.mTcw));
    //TODO: Create the 3d point for the line too

    std::vector<std::pair<cv::Mat, cv::Mat>> mvObj3DLineTmp(mvObjKeysTmp_line.size());
    for (int i = 0; i < mvObjKeysTmp_line.size(); ++i)
    {
        mvObj3DLineTmp[i] = Optimizer::Get3DinWorld_line(mvObjKeysTmp_line[i], mvObjDepthTmp_line[i], mK, Converter::toInvMatrix(mCurrentFrame.mTcw));
    }
    
    // update
    mCurrentFrame.mvObjKeys = mvObjKeysTmp;
    mCurrentFrame.mvObjDepth = mvObjDepthTmp;
    mCurrentFrame.mvObj3DPoint = mvObj3DPointTmp;
    mCurrentFrame.mvObjCorres = mvObjCorresTmp;
    mCurrentFrame.mvObjFlowNext = mvObjFlowNextTmp;
    mCurrentFrame.vSemObjLabel = vSemObjLabelTmp;
    mCurrentFrame.nDynInlierID = DynInlierIDTmp;
    mCurrentFrame.vObjLabel = vObjLabelTmp;

    //for lines
    mCurrentFrame.mvObjKeys_Line = mvObjKeysTmp_line;
    mCurrentFrame.mvObjDepth_line = mvObjDepthTmp_line;
    mCurrentFrame.mvObj3DLine = mvObj3DLineTmp;
    mCurrentFrame.mvObjCorres_Line = mvObjCorresTmp_line;
    mCurrentFrame.mvObjFlowNext_Line = mvObjFlowNextTmp_line;
    mCurrentFrame.vSemObjLabel_Line = vSemObjLabelTmp_line;
    mCurrentFrame.nDynInlierID_line = DynInlierIDTmp_line;
    mCurrentFrame.vObjLabel_Line = vObjLabelTmp_line;

    // cout << "updating DYNAMIC features finished...... " << mvObjKeysTmp.size() << endl;
    cout << "Renew Frame Info, Done!" << endl;
}

void Tracking::UpdateMask()
{
    cout << "Update Mask ......" << endl;

    // find the unique labels in semantic label

    //cout << "printing the contents of vSemObjLabel";
    //for (const auto& semobj : mLastFrame.vSemObjLabel) {
    //    cout << semobj << " ";
    //}

    //cout << endl;
    //cout << "vSemObjLabel size: " << mLastFrame.vSemObjLabel.size() << endl;

    auto UniLab = mLastFrame.vSemObjLabel;
    std::sort(UniLab.begin(), UniLab.end());
    UniLab.erase(std::unique( UniLab.begin(), UniLab.end() ), UniLab.end() );

    // collect the predicted labels and semantic labels in vector
    std::vector<std::vector<int> > ObjID(UniLab.size());
    for (int i = 0; i < mLastFrame.vSemObjLabel.size(); ++i)
    {
        // save object label
        for (int j = 0; j < UniLab.size(); ++j)
        {
            if(mLastFrame.vSemObjLabel[i]==UniLab[j]){
                ObjID[j].push_back(i);
                break;
            }
        }
    }

    // check each object label distribution in the coming frame
    for (int i = 0; i < ObjID.size(); ++i)
    {
        // collect labels
        std::vector<int> LabTmp;
        for (int j = 0; j < ObjID[i].size(); ++j)
        {
            const int u = mLastFrame.mvObjCorres[ObjID[i][j]].pt.x;
            const int v = mLastFrame.mvObjCorres[ObjID[i][j]].pt.y;
            if (u<mImGray.cols && u>0 && v<mImGray.rows && v>0)
            {
                LabTmp.push_back(mSegMap.at<int>(v,u));
            }
        }

        if (LabTmp.size()<100)
            continue;

        // find label that appears most in LabTmp()
        // (1) count duplicates
        std::map<int, int> dups;
        for(int k : LabTmp)
            ++dups[k];
        // (2) and sort them by descending order
        std::vector<std::pair<int, int> > sorted;
        for (auto k : dups)
            sorted.push_back(std::make_pair(k.first,k.second));
        std::sort(sorted.begin(), sorted.end(), SortPairInt);

        // recover the missing mask (time consuming!)
        if (sorted[0].first==0) // && sorted[0].second==LabTmp.size()
        {
            for (int j = 0; j < mImGrayLast.rows; j++)
            {
                for (int k = 0; k < mImGrayLast.cols; k++)
                {
                    if (mSegMapLast.at<int>(j,k)==UniLab[i])
                    {
                        const int flow_x = mFlowMapLast.at<cv::Vec2f>(j,k)[0];
                        const int flow_y = mFlowMapLast.at<cv::Vec2f>(j,k)[1];

                        if(k+flow_x < mImGrayLast.cols && k+flow_x > 0 && j+flow_y < mImGrayLast.rows && j+flow_y > 0)
                            mSegMap.at<int>(j+flow_y,k+flow_x) = UniLab[i];
                    }
                }
            }
        }
        // end of recovery
    }

    // // === verify the updated labels ===
    // cv::Mat imgLabel(mImGray.rows,mImGray.cols,CV_8UC3); // for display
    // for (int i = 0; i < mSegMap.rows; ++i)
    // {
    //     for (int j = 0; j < mSegMap.cols; ++j)
    //     {
    //         int tmp = mSegMap.at<int>(i,j);
    //         if (tmp>50)
    //             tmp = tmp/2;
    //         switch (tmp)
    //         {
    //             case 0:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,240);
    //                 break;
    //             case 1:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,255);
    //                 break;
    //             case 2:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,0,0);
    //                 break;
    //             case 3:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,0);
    //                 break;
    //             case 4:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(47,255,173); // greenyellow
    //                 break;
    //             case 5:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 128);
    //                 break;
    //             case 6:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(203,192,255);
    //                 break;
    //             case 7:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(196,228,255);
    //                 break;
    //             case 8:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(42,42,165);
    //                 break;
    //             case 9:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
    //                 break;
    //             case 10:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(245,245,245); // whitesmoke
    //                 break;
    //             case 11:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,165,255); // orange
    //                 break;
    //             case 12:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(230,216,173); // lightblue
    //                 break;
    //             case 13:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128,128,128); // grey
    //                 break;
    //             case 14:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,215,255); // gold
    //                 break;
    //             case 15:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(30,105,210); // chocolate
    //                 break;
    //             case 16:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);  // green
    //                 break;
    //             case 17:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
    //                 break;
    //             case 18:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
    //                 break;
    //             case 19:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
    //                 break;
    //             case 20:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
    //                 break;
    //             case 21:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(225, 228, 255);  // mistyrose
    //                 break;
    //             case 22:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 0);  // navy
    //                 break;
    //             case 23:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(35, 142, 107);  // olivedrab
    //                 break;
    //             case 24:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(45, 82, 160);  // sienna
    //                 break;
    //             case 25:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 255, 127); // chartreuse
    //                 break;
    //             case 26:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(139, 0, 0);  // darkblue
    //                 break;
    //             case 27:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(60, 20, 220);  // crimson
    //                 break;
    //             case 28:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 139);  // darkred
    //                 break;
    //             case 29:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(211, 0, 148);  // darkviolet
    //                 break;
    //             case 30:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 144, 30);  // dodgerblue
    //                 break;
    //             case 31:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(105, 105, 105);  // dimgray
    //                 break;
    //             case 32:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(180, 105, 255);  // hotpink
    //                 break;
    //             case 33:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(204, 209, 72);  // mediumturquoise
    //                 break;
    //             case 34:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(173, 222, 255);  // navajowhite
    //                 break;
    //             case 35:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(143, 143, 188); // rosybrown
    //                 break;
    //             case 36:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(50, 205, 50);  // limegreen
    //                 break;
    //             case 37:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
    //                 break;
    //             case 38:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
    //                 break;
    //             case 39:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
    //                 break;
    //             case 40:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
    //                 break;
    //             case 41:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(225, 228, 255);  // mistyrose
    //                 break;
    //             case 42:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 0);  // navy
    //                 break;
    //             case 43:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(35, 142, 107);  // olivedrab
    //                 break;
    //             case 44:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(45, 82, 160);  // sienna
    //                 break;
    //             case 45:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(30,105,210); // chocolate
    //                 break;
    //             case 46:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);  // green
    //                 break;
    //             case 47:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
    //                 break;
    //             case 48:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
    //                 break;
    //             case 49:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
    //                 break;
    //             case 50:
    //                 imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
    //                 break;
    //         }
    //     }
    // }
    // cv::imshow("Updated Mask Image", imgLabel);
    // cv::waitKey(1);

    cout << "Update Mask, Done!" << endl;
}

void Tracking::GetMetricError(const std::vector<cv::Mat> &CamPose, const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &ObjPosePre,
                    const std::vector<cv::Mat> &CamPose_gt, const std::vector<std::vector<cv::Mat> > &RigMot_gt,
                    const std::vector<std::vector<bool> > &ObjStat)
{
    bool bRMSError = false;
    cout << "=================================================" << endl;

    std::ofstream output_file_errors;
    output_file_errors.open("Results/Metrix_error.txt", std::ios::app);
    //absolute trajectory error for CAMERA (RMSE)
    cout << "CAMERA:" << endl;
    output_file_errors << "CAMERA:" << endl;
    float t_sum = 0, r_sum = 0;
    for (int i = 1; i < CamPose.size(); ++i)
    {
        cv::Mat T_lc_inv = CamPose[i]*Converter::toInvMatrix(CamPose[i-1]);
        cv::Mat T_lc_gt = CamPose_gt[i-1]*Converter::toInvMatrix(CamPose_gt[i]);
        cv::Mat ate_cam = T_lc_inv*T_lc_gt;
        // cv::Mat ate_cam = CamPose[i]*Converter::toInvMatrix(CamPose_gt[i]);

        // translation
        float t_ate_cam = std::sqrt(ate_cam.at<float>(0,3)*ate_cam.at<float>(0,3) + ate_cam.at<float>(1,3)*ate_cam.at<float>(1,3) + ate_cam.at<float>(2,3)*ate_cam.at<float>(2,3));
        if (bRMSError)
            t_sum = t_sum + t_ate_cam*t_ate_cam;
        else
            t_sum = t_sum + t_ate_cam;

        // rotation
        float trace_ate = 0;
        for (int j = 0; j < 3; ++j)
        {
            if (ate_cam.at<float>(j,j)>1.0)
                trace_ate = trace_ate + 1.0-(ate_cam.at<float>(j,j)-1.0);
            else
                trace_ate = trace_ate + ate_cam.at<float>(j,j);
        }
        float r_ate_cam = acos( (trace_ate -1.0)/2.0 )*180.0/3.1415926;
        if (bRMSError)
            r_sum = r_sum + r_ate_cam*r_ate_cam;
        else
            r_sum = r_sum + r_ate_cam;

        // cout << " t: " << t_ate_cam << " R: " << r_ate_cam << endl;
    }
    if (bRMSError)
    {
        t_sum = std::sqrt(t_sum/(CamPose.size()-1));
        r_sum = std::sqrt(r_sum/(CamPose.size()-1));
    }
    else
    {
        t_sum = t_sum/(CamPose.size()-1);
        r_sum = r_sum/(CamPose.size()-1);
    }

    cout << "average error (Camera):" << " t: " << t_sum << " R: " << r_sum << endl;
    output_file_errors << "average error (Camera):" << " t: " << t_sum << " R: " << r_sum << endl;
    std::vector<float> each_obj_t(max_id-1,0);
    std::vector<float> each_obj_r(max_id-1,0);
    std::vector<int> each_obj_count(max_id-1,0);

    // all motion error for OBJECTS (mean error)
    cout << "OBJECTS:" << endl;
    output_file_errors << "OBJECTS:" << endl;
    float r_rpe_sum = 0, t_rpe_sum = 0, obj_count = 0;
    for (int i = 0; i < RigMot.size(); ++i)
    {
        if (RigMot[i].size()>1)
        {
            for (int j = 1; j < RigMot[i].size(); ++j)
            {
                if (!ObjStat[i][j])
                {
                    cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case." << endl;
                    output_file_errors << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case." << endl;
                    continue;
                }

                cv::Mat RigMotBody = Converter::toInvMatrix(ObjPosePre[i][j])*RigMot[i][j]*ObjPosePre[i][j];
                cv::Mat rpe_obj = Converter::toInvMatrix(RigMotBody)*RigMot_gt[i][j];

                // translation error
                float t_rpe_obj = std::sqrt( rpe_obj.at<float>(0,3)*rpe_obj.at<float>(0,3) + rpe_obj.at<float>(1,3)*rpe_obj.at<float>(1,3) + rpe_obj.at<float>(2,3)*rpe_obj.at<float>(2,3) );
                if (bRMSError){
                    each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj*t_rpe_obj;
                    t_rpe_sum = t_rpe_sum + t_rpe_obj*t_rpe_obj;
                }
                else{
                    each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj;
                    t_rpe_sum = t_rpe_sum + t_rpe_obj;
                }

                // rotation error
                float trace_rpe = 0;
                for (int k = 0; k < 3; ++k)
                {
                    if (rpe_obj.at<float>(k,k)>1.0)
                        trace_rpe = trace_rpe + 1.0-(rpe_obj.at<float>(k,k)-1.0);
                    else
                        trace_rpe = trace_rpe + rpe_obj.at<float>(k,k);
                }
                float r_rpe_obj = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.1415926;
                if (bRMSError){
                    each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj*r_rpe_obj;
                    r_rpe_sum = r_rpe_sum + r_rpe_obj*r_rpe_obj;
                }
                else{
                    each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj;
                    r_rpe_sum = r_rpe_sum + r_rpe_obj;
                }

                // cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " t: " << t_rpe_obj << " R: " << r_rpe_obj << endl;
                obj_count++;
                each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1] + 1;
            }
        }
    }
    if (bRMSError)
    {
        t_rpe_sum = std::sqrt(t_rpe_sum/obj_count);
        r_rpe_sum = std::sqrt(r_rpe_sum/obj_count);
    }
    else
    {
        t_rpe_sum = t_rpe_sum/obj_count;
        r_rpe_sum = r_rpe_sum/obj_count;
    }
    cout << "average error (Over All Objects):" << " t: " << t_rpe_sum << " R: " << r_rpe_sum << endl;
    output_file_errors << "average error (Over All Objects):" << " t: " << t_rpe_sum << " R: " << r_rpe_sum << endl;
    // show each object
    for (int i = 0; i < each_obj_count.size(); ++i)
    {
        if (bRMSError)
        {
            each_obj_t[i] = std::sqrt(each_obj_t[i]/each_obj_count[i]);
            each_obj_r[i] = std::sqrt(each_obj_r[i]/each_obj_count[i]);
        }
        else
        {
            each_obj_t[i] = each_obj_t[i]/each_obj_count[i];
            each_obj_r[i] = each_obj_r[i]/each_obj_count[i];
        }
        if (each_obj_count[i]>=3)
            cout << endl << "average error of Object " << i+1 << ": " << " t: " << each_obj_t[i] << " R: " << each_obj_r[i] << " TrackCount: " << each_obj_count[i] << endl;
            output_file_errors << endl << "average error of Object " << i+1 << ": " << " t: " << each_obj_t[i] << " R: " << each_obj_r[i] << " TrackCount: " << each_obj_count[i] << endl;
    }

    cout << "=================================================" << endl;
    output_file_errors.close();

}

void Tracking::PlotMetricError(const std::vector<cv::Mat> &CamPose, const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &ObjPosePre,
                    const std::vector<cv::Mat> &CamPose_gt, const std::vector<std::vector<cv::Mat> > &RigMot_gt,
                    const std::vector<std::vector<bool> > &ObjStat)
{
    // saved evaluated errors
    std::vector<float> CamRotErr(CamPose.size()-1);
    std::vector<float> CamTraErr(CamPose.size()-1);
    std::vector<std::vector<float> > ObjRotErr(max_id-1);
    std::vector<std::vector<float> > ObjTraErr(max_id-1);

    bool bRMSError = true, bAccumError = false;
    cout << "=================================================" << endl;

    // absolute trajectory error for CAMERA (RMSE)
    cout << "CAMERA:" << endl;
    float t_sum = 0, r_sum = 0;
    for (int i = 1; i < CamPose.size(); ++i)
    {
        cv::Mat T_lc_inv = CamPose[i]*Converter::toInvMatrix(CamPose[i-1]);
        cv::Mat T_lc_gt = CamPose_gt[i-1]*Converter::toInvMatrix(CamPose_gt[i]);
        cv::Mat ate_cam = T_lc_inv*T_lc_gt;
        // cv::Mat ate_cam = CamPose[i]*Converter::toInvMatrix(CamPose_gt[i]);

        // translation
        float t_ate_cam = std::sqrt(ate_cam.at<float>(0,3)*ate_cam.at<float>(0,3) + ate_cam.at<float>(1,3)*ate_cam.at<float>(1,3) + ate_cam.at<float>(2,3)*ate_cam.at<float>(2,3));
        if (bRMSError)
            t_sum = t_sum + t_ate_cam*t_ate_cam;
        else
            t_sum = t_sum + t_ate_cam;

        // rotation
        float trace_ate = 0;
        for (int j = 0; j < 3; ++j)
        {
            if (ate_cam.at<float>(j,j)>1.0)
                trace_ate = trace_ate + 1.0-(ate_cam.at<float>(j,j)-1.0);
            else
                trace_ate = trace_ate + ate_cam.at<float>(j,j);
        }
        float r_ate_cam = acos( (trace_ate -1.0)/2.0 )*180.0/3.1415926;
        if (bRMSError)
            r_sum = r_sum + r_ate_cam*r_ate_cam;
        else
            r_sum = r_sum + r_ate_cam;

        if (bAccumError)
        {
            CamRotErr[i-1] = r_ate_cam/i;
            CamTraErr[i-1] = t_ate_cam/i;
        }
        else
        {
            CamRotErr[i-1] = r_ate_cam;
            CamTraErr[i-1] = t_ate_cam;            
        }




        // cout << " t: " << t_ate_cam << " R: " << r_ate_cam << endl;
    }
    if (bRMSError)
    {
        t_sum = std::sqrt(t_sum/(CamPose.size()-1));
        r_sum = std::sqrt(r_sum/(CamPose.size()-1));
    }
    else
    {
        t_sum = t_sum/(CamPose.size()-1);
        r_sum = r_sum/(CamPose.size()-1);
    }

    cout << "average error (Camera):" << " t: " << t_sum << " R: " << r_sum << endl;

    std::vector<float> each_obj_t(max_id-1,0);
    std::vector<float> each_obj_r(max_id-1,0);
    std::vector<int> each_obj_count(max_id-1,0);

    // all motion error for OBJECTS (mean error)
    cout << "OBJECTS:" << endl;
    float r_rpe_sum = 0, t_rpe_sum = 0, obj_count = 0;
    for (int i = 0; i < RigMot.size(); ++i)
    {
        if (RigMot[i].size()>1)
        {
            for (int j = 1; j < RigMot[i].size(); ++j)
            {
                if (!ObjStat[i][j])
                {
                    cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case." << endl;
                    continue;
                }

                cv::Mat RigMotBody = Converter::toInvMatrix(ObjPosePre[i][j])*RigMot[i][j]*ObjPosePre[i][j];
                cv::Mat rpe_obj = Converter::toInvMatrix(RigMotBody)*RigMot_gt[i][j];

                // translation error
                float t_rpe_obj = std::sqrt( rpe_obj.at<float>(0,3)*rpe_obj.at<float>(0,3) + rpe_obj.at<float>(1,3)*rpe_obj.at<float>(1,3) + rpe_obj.at<float>(2,3)*rpe_obj.at<float>(2,3) );
                if (bRMSError){
                    each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj*t_rpe_obj;
                    t_rpe_sum = t_rpe_sum + t_rpe_obj*t_rpe_obj;
                }
                else{
                    each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj;
                    t_rpe_sum = t_rpe_sum + t_rpe_obj;
                }

                // rotation error
                float trace_rpe = 0;
                for (int k = 0; k < 3; ++k)
                {
                    if (rpe_obj.at<float>(k,k)>1.0)
                        trace_rpe = trace_rpe + 1.0-(rpe_obj.at<float>(k,k)-1.0);
                    else
                        trace_rpe = trace_rpe + rpe_obj.at<float>(k,k);
                }
                float r_rpe_obj = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.1415926; 
                if (bRMSError){
                    each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj*r_rpe_obj;
                    r_rpe_sum = r_rpe_sum + r_rpe_obj*r_rpe_obj;
                }
                else{
                    each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj;
                    r_rpe_sum = r_rpe_sum + r_rpe_obj;
                }

                // cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " t: " << t_rpe_obj << " R: " << r_rpe_obj << endl;
                obj_count++;
                each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1] + 1;
                if (bAccumError)
                {
                    ObjTraErr[mpMap->vnRMLabel[i][j]-1].push_back(each_obj_t[mpMap->vnRMLabel[i][j]-1]/each_obj_count[mpMap->vnRMLabel[i][j]-1]);
                    ObjRotErr[mpMap->vnRMLabel[i][j]-1].push_back(each_obj_r[mpMap->vnRMLabel[i][j]-1]/each_obj_count[mpMap->vnRMLabel[i][j]-1]);
                }
                else
                {
                    ObjTraErr[mpMap->vnRMLabel[i][j]-1].push_back(t_rpe_obj);
                    ObjRotErr[mpMap->vnRMLabel[i][j]-1].push_back(r_rpe_obj);           
                }

            }
        }
    }
    if (bRMSError)
    {
        t_rpe_sum = std::sqrt(t_rpe_sum/obj_count);
        r_rpe_sum = std::sqrt(r_rpe_sum/obj_count);
    }
    else
    {
        t_rpe_sum = t_rpe_sum/obj_count;
        r_rpe_sum = r_rpe_sum/obj_count;
    }
    cout << "average error (Over All Objects):" << " t: " << t_rpe_sum << " R: " << r_rpe_sum << endl;

    // show each object
    for (int i = 0; i < each_obj_count.size(); ++i)
    {
        if (bRMSError)
        {
            each_obj_t[i] = std::sqrt(each_obj_t[i]/each_obj_count[i]);
            each_obj_r[i] = std::sqrt(each_obj_r[i]/each_obj_count[i]);
        }
        else
        {
            each_obj_t[i] = each_obj_t[i]/each_obj_count[i];
            each_obj_r[i] = each_obj_r[i]/each_obj_count[i];
        }
        if (each_obj_count[i]>=3)
            cout << endl << "average error of Object " << i+1 << ": " << " t: " << each_obj_t[i] << " R: " << each_obj_r[i] << endl;
    }

    cout << "=================================================" << endl;


    auto name1 = "Translation";
    cvplot::setWindowTitle(name1, "Translation Error (Meter)");
    cvplot::moveWindow(name1, 0, 240);
    cvplot::resizeWindow(name1, 800, 240);
    auto &figure1 = cvplot::figure(name1);

    auto name2 = "Rotation";
    cvplot::setWindowTitle(name2, "Rotation Error (Degree)");
    cvplot::resizeWindow(name2, 800, 240);
    auto &figure2 = cvplot::figure(name2);

    figure1.series("Camera")
        .setValue(CamTraErr)
        .type(cvplot::DotLine)
        .color(cvplot::Red);

    figure2.series("Camera")
        .setValue(CamRotErr)
        .type(cvplot::DotLine)
        .color(cvplot::Red);

    for (int i = 0; i < max_id-1; ++i)
    {
        switch (i)
        {
            case 0:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Purple);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Purple);
                break;
            case 1:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Green);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Green);
                break;
            case 2:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Cyan);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Cyan);
                break;
            case 3:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Blue);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Blue);
                break;
            case 4:
                figure1.series("Object "+std::to_string(i+1))
                    .setValue(ObjTraErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Pink);
                figure2.series("Object "+std::to_string(i+1))
                    .setValue(ObjRotErr[i])
                    .type(cvplot::DotLine)
                    .color(cvplot::Pink);
                break;
        }
    }

    figure1.show(true);
    figure2.show(true);

}

void Tracking::GetVelocityError(const std::vector<std::vector<cv::Mat> > &RigMot, const std::vector<std::vector<cv::Mat> > &PointDyn,
                                const std::vector<std::vector<int> > &FeaLab, const std::vector<std::vector<int> > &RMLab,
                                const std::vector<std::vector<float> > &Velo_gt, const std::vector<std::vector<int> > &TmpMatch,
                                const std::vector<std::vector<bool> > &ObjStat)
{
    bool bRMSError = true;
    float s_sum = 0, s_gt_sum = 0, obj_count = 0;

    string path = "/Users/steed/work/code/Evaluation/ijrr2020/";
    string path_sp_e = path + "speed_error.txt";
    string path_sp_est = path + "speed_estimated.txt";
    string path_sp_gt = path + "speed_groundtruth.txt";
    string path_track = path + "tracking_id.txt";
    ofstream save_sp_e, save_sp_est, save_sp_gt, save_tra;
    save_sp_e.open(path_sp_e.c_str(),ios::trunc);
    save_sp_est.open(path_sp_est.c_str(),ios::trunc);
    save_sp_gt.open(path_sp_gt.c_str(),ios::trunc);
    save_tra.open(path_track.c_str(),ios::trunc);

    std::vector<float> each_obj_est(max_id-1,0);
    std::vector<float> each_obj_gt(max_id-1,0);
    std::vector<int> each_obj_count(max_id-1,0);

    cout << "OBJECTS SPEED:" << endl;

    // Main loop for each frame
    for (int i = 0; i < RigMot.size(); ++i)
    {
        save_tra << i << " ";

        // Check if there are moving objects, and if all the variables are consistent
        if (RigMot[i].size()>1 && Velo_gt[i].size()>1 && RMLab[i].size()>1)
        {
            // Loop for each object in each frame
            for (int j = 1; j < RigMot[i].size(); ++j)
            {
                // check if this is valid object estimate
                if (!ObjStat[i][j])
                {
                    cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case." << endl;
                    continue;
                }

                // (1) Compute each object centroid
                cv::Mat ObjCenter = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
                float ObjFeaCount = 0;
                if (i==0)
                {
                    for (int k = 0; k < PointDyn[i+1].size(); ++k)
                    {
                        if (FeaLab[i][k]!=RMLab[i][j])
                            continue;
                        if (TmpMatch[i][k]==-1)
                            continue;

                        ObjCenter = ObjCenter + PointDyn[i][TmpMatch[i][k]];
                        ObjFeaCount = ObjFeaCount + 1;
                    }
                    ObjCenter = ObjCenter/ObjFeaCount;
                }
                else
                {
                    for (int k = 0; k < PointDyn[i+1].size(); ++k)
                    {
                        if (FeaLab[i][k]!=RMLab[i][j])
                            continue;
                        if (TmpMatch[i][k]==-1)
                            continue;

                        ObjCenter = ObjCenter + PointDyn[i][TmpMatch[i][k]];
                        ObjFeaCount = ObjFeaCount + 1;
                    }
                    ObjCenter = ObjCenter/ObjFeaCount;
                }


                // (2) Compute object velocity
                cv::Mat sp_est_v = RigMot[i][j].rowRange(0,3).col(3) - (cv::Mat::eye(3,3,CV_32F)-RigMot[i][j].rowRange(0,3).colRange(0,3))*ObjCenter;
                float sp_est_norm = std::sqrt( sp_est_v.at<float>(0)*sp_est_v.at<float>(0) + sp_est_v.at<float>(1)*sp_est_v.at<float>(1) + sp_est_v.at<float>(2)*sp_est_v.at<float>(2) )*36;

                // (3) Compute velocity error
                float speed_error = sp_est_norm - Velo_gt[i][j];
                if (bRMSError){
                    each_obj_est[mpMap->vnRMLabel[i][j]-1] = each_obj_est[mpMap->vnRMLabel[i][j]-1] + sp_est_norm*sp_est_norm;
                    each_obj_gt[mpMap->vnRMLabel[i][j]-1] = each_obj_gt[mpMap->vnRMLabel[i][j]-1] + Velo_gt[i][j]*Velo_gt[i][j];
                    s_sum = s_sum + speed_error*speed_error;
                }
                else{
                    each_obj_est[mpMap->vnRMLabel[i][j]-1] = each_obj_est[mpMap->vnRMLabel[i][j]-1] + sp_est_norm;
                    each_obj_gt[mpMap->vnRMLabel[i][j]-1] = each_obj_gt[mpMap->vnRMLabel[i][j]-1] + Velo_gt[i][j];
                    s_sum = s_sum + speed_error;
                }

                // (4) sum ground truth speed
                s_gt_sum = s_gt_sum + Velo_gt[i][j];

                save_sp_e << fixed << setprecision(6) << speed_error << " ";
                save_sp_est << fixed << setprecision(6) << sp_est_norm << " ";
                save_sp_gt << fixed << setprecision(6) << Velo_gt[i][j] << " ";
                save_tra << mpMap->vnRMLabel[i][j] << " ";

                // cout << "(" << i+1 << "/" << mpMap->vnRMLabel[i][j] << ")" << " s: " << speed_error << " est: " << sp_est_norm << " gt: " << Velo_gt[i][j] << endl;
                obj_count = obj_count + 1;
                each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1] + 1;
            }
            save_sp_e << endl;
            save_sp_est << endl;
            save_sp_gt << endl;
            save_tra << endl;
        }
    }

    save_sp_e.close();
    save_sp_est.close();
    save_sp_gt.close();

    if (bRMSError)
        s_sum = std::sqrt(s_sum/obj_count);
    else
        s_sum = std::abs(s_sum/obj_count);

    s_gt_sum = s_gt_sum/obj_count;

    cout << "average speed error (All Objects):" << " s: " << s_sum << "km/h " << "Track Num: " << (int)obj_count << " GT AVG SPEED: " << s_gt_sum << endl;

    for (int i = 0; i < each_obj_count.size(); ++i)
    {
        if (bRMSError){
            each_obj_est[i] = std::sqrt(each_obj_est[i]/each_obj_count[i]);
            each_obj_gt[i] = std::sqrt(each_obj_gt[i]/each_obj_count[i]);
        }
        else{
            each_obj_est[i] = each_obj_est[i]/each_obj_count[i];
            each_obj_gt[i] = each_obj_gt[i]/each_obj_count[i];
        }
        if (mpMap->nObjTraCount[i]>=3)
            cout << endl << "average error of Object " << i+1 << " (" << mpMap->nObjTraCount[i] << "/" << mpMap->nObjTraCountGT[i] <<  "/" << mpMap->nObjTraSemLab[i]  << "): " << " (est) " << each_obj_est[i] << " (gt) " << each_obj_gt[i] << endl;
    }

    cout << "=================================================" << endl << endl;

}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

} //namespace SDPL_SLAM
