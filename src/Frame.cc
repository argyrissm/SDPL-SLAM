/**
* This file is part of SDPL-SLAM.
*
* Copyright (C) 2024 Argyris Manetas, National Technical University of Athens
* For more information see <https://github.com/argyrissm/SDPL-SLAM>
*
**/

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/flann.hpp>

#include "Frame.h"
#include "Converter.h"
#include <thread>
#include<time.h>
#include<chrono>

namespace SDPL_SLAM
{

using namespace std;

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), mThDepthObj(frame.mThDepthObj), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight), mvDepth(frame.mvDepth),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvbOutlier(frame.mvbOutlier), mnId(frame.mnId), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor), 
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),  
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     // new added
     mTcw_gt(frame.mTcw_gt), vObjPose_gt(frame.vObjPose_gt), nSemPosi_gt(frame.nSemPosi_gt),
     vObjLabel(frame.vObjLabel), nModLabel(frame.nModLabel), nSemPosition(frame.nSemPosition),
     bObjStat(frame.bObjStat), vObjMod(frame.vObjMod), mvCorres(frame.mvCorres), mvObjCorres(frame.mvObjCorres),
     mvFlowNext(frame.mvFlowNext), mvObjFlowNext(frame.mvObjFlowNext), mvFlowNext_Line(frame.mvFlowNext_Line),
     //ORB-LINE-SLAM
     mpLineextractorLeft(frame.mpLineextractorLeft), mpLineextractorRight(frame.mpLineextractorRight),
     mvKeys_Line(frame.mvKeys_Line), mvKeysRight_Line(frame.mvKeysRight_Line),
     mvDepth_Line(frame.mvDepth_Line), mDescriptors_Line(frame.mDescriptors_Line), mDescriptorsRight_Line(frame.mDescriptorsRight_Line),
     N_sta_l(frame.N_sta_l), mvStatKeys_Line(frame.mvStatKeys_Line), mvStatKeysRight_Line(frame.mvStatKeysRight_Line),
     mvObjKeys_Line(frame.mvObjKeys_Line), mvObjDepth_line(frame.mvObjDepth_line), mvObjCorres_Line(frame.mvObjCorres_Line),
     mvObjFlowGT_Line(frame.mvObjFlowGT_Line), mvObjFlowNext_Line(frame.mvObjFlowNext_Line), vSemObjLabel_Line(frame.vSemObjLabel_Line),
     mvCorresLine(frame.mvCorresLine), mvStatKeysLineTmp(frame.mvStatKeysLineTmp), mvObjKeys_Linetmp(frame.mvObjKeys_Linetmp),
     imGray_(frame.imGray_), //delete if i don't want to show line correspondences
     mvInfiniteLinesCorr(frame.mvInfiniteLinesCorr), vObjLabel_Line(frame.vObjLabel_Line)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imFlow, const cv::Mat &maskSEM,
    const double &timeStamp, ORBextractor* extractor,cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const float &thDepthObj, const int &UseSampleFea)
    :mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mThDepthObj(thDepthObj)
{

    cout << "Start Constructing Frame......" << endl;

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();


    // ------------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++ New added for background features +++++++++++++++++++++++++++
    // ------------------------------------------------------------------------------------------

    // clock_t s_1, e_1;
    // double fea_det_time;
    // s_1 = clock();
    // ORB extraction
    ExtractORB(0,imGray);
    // e_1 = clock();
    // fea_det_time = (double)(e_1-s_1)/CLOCKS_PER_SEC*1000;
    // cout << "feature detection time: " << fea_det_time << endl;

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    if (UseSampleFea==0)
    {
        // // // Option I: ~~~~~~~ use detected features ~~~~~~~~~~ // // //

        for (int i = 0; i < mvKeys.size(); ++i)
        {
            int x = mvKeys[i].pt.x;
            int y = mvKeys[i].pt.y;

            if (maskSEM.at<int>(y,x)!=0)  // new added in Jun 13 2019
                continue;

            if (imDepth.at<float>(y,x)>mThDepth || imDepth.at<float>(y,x)<=0)  // new added in Aug 21 2019
                continue;

            float flow_xe = imFlow.at<cv::Vec2f>(y,x)[0];
            float flow_ye = imFlow.at<cv::Vec2f>(y,x)[1];


            if(flow_xe!=0 && flow_ye!=0)
            {
                if(mvKeys[i].pt.x+flow_xe < imGray.cols && mvKeys[i].pt.y+flow_ye < imGray.rows && mvKeys[i].pt.x < imGray.cols && mvKeys[i].pt.y < imGray.rows)
                {
                    mvStatKeysTmp.push_back(mvKeys[i]);
                    mvCorres.push_back(cv::KeyPoint(mvKeys[i].pt.x+flow_xe,mvKeys[i].pt.y+flow_ye,0,0,0,mvKeys[i].octave,-1));
                    mvFlowNext.push_back(cv::Point2f(flow_xe,flow_ye));
                }
            }
        }
    }
    else
    {
        // // // Option II: ~~~~~~~ use sampled features ~~~~~~~~~~ // // //

        clock_t s_1, e_1;
        double fea_det_time;
        s_1 = clock();
        std::vector<cv::KeyPoint> mvKeysSamp = SampleKeyPoints(imGray.rows, imGray.cols);
        e_1 = clock();
        fea_det_time = (double)(e_1-s_1)/CLOCKS_PER_SEC*1000;
        std::cout << "feature detection time: " << fea_det_time << std::endl;

        for (int i = 0; i < mvKeysSamp.size(); ++i)
        {
            int x = mvKeysSamp[i].pt.x;
            int y = mvKeysSamp[i].pt.y;

            if (maskSEM.at<int>(y,x)!=0)  // new added in Jun 13 2019
                continue;

            if (imDepth.at<float>(y,x)>mThDepth || imDepth.at<float>(y,x)<=0)  // new added in Aug 21 2019
                continue;

            float flow_xe = imFlow.at<cv::Vec2f>(y,x)[0];
            float flow_ye = imFlow.at<cv::Vec2f>(y,x)[1];


            if(flow_xe!=0 && flow_ye!=0)
            {
                if(mvKeysSamp[i].pt.x+flow_xe < imGray.cols && mvKeysSamp[i].pt.y+flow_ye < imGray.rows && mvKeysSamp[i].pt.x+flow_xe>0 && mvKeysSamp[i].pt.y+flow_ye>0)
                {
                    mvStatKeysTmp.push_back(mvKeysSamp[i]);
                    mvCorres.push_back(cv::KeyPoint(mvKeysSamp[i].pt.x+flow_xe,mvKeysSamp[i].pt.y+flow_ye,0,0,0,mvKeysSamp[i].octave,-1));
                    mvFlowNext.push_back(cv::Point2f(flow_xe,flow_ye));

                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------

    // cv::Mat img_show;
    // cv::drawKeypoints(imGray, mvKeysSamp, img_show, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    // cv::imshow("KeyPoints on Background", img_show);
    // cv::waitKey(0);

    N_s_tmp = mvCorres.size();
    // cout << "number of random sample points: " << mvCorres.size() << endl;


    //TO-DO do for lines
    // assign the depth value to each keypoint
    mvStatDepthTmp = vector<float>(N_s_tmp,-1);
    for(int i=0; i<N_s_tmp; i++)
    {
        const cv::KeyPoint &kp = mvStatKeysTmp[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        float d = imDepth.at<float>(v,u); // be careful with the order  !!!

        if(d>0)
            mvStatDepthTmp[i] = d;
    }

    // ---------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++ New added for dense object features ++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------

    // semi-dense features on objects
    int step = 4; // 3
    for (int i = 0; i < imGray.rows; i=i+step)
    {
        for (int j = 0; j < imGray.cols; j=j+step)
        {

            // check ground truth motion mask
            if (maskSEM.at<int>(i,j)!=0 && imDepth.at<float>(i,j)<mThDepthObj && imDepth.at<float>(i,j)>0)
            {
                // get flow
                const float flow_x = imFlow.at<cv::Vec2f>(i,j)[0];
                const float flow_y = imFlow.at<cv::Vec2f>(i,j)[1];

                if(j+flow_x < imGray.cols && j+flow_x > 0 && i+flow_y < imGray.rows && i+flow_y > 0)
                {
                    // save correspondences
                    mvObjFlowNext.push_back(cv::Point2f(flow_x,flow_y));
                    mvObjCorres.push_back(cv::KeyPoint(j+flow_x,i+flow_y,0,0,0,-1));
                    // save pixel location
                    mvObjKeys.push_back(cv::KeyPoint(j,i,0,0,0,-1));
                    // save depth
                    mvObjDepth.push_back(imDepth.at<float>(i,j));
                    // save label
                    vSemObjLabel.push_back(maskSEM.at<int>(i,j));
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------

    //TO-DO, I think the following two are not used anywhere else
    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();

    cout << "Constructing Frame, Done!" << endl;
}


//Constructor for RGB-D with lines
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imFlow, const cv::Mat &maskSEM, 
    const double &timeStamp, ORBextractor* extractor, Lineextractor* lextractor, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const float &thDepthObj, const int &UseSampleFea)
    : imGray_(imGray), mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mpLineextractorLeft(lextractor), mpLineextractorRight(static_cast<Lineextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mThDepthObj(thDepthObj)
{

    cout << "Start Constructing Frame......" << endl;

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    //Scale Level info for lines
    mnScaleLevels_l = mpLineextractorLeft->nlevels_l;
    mvScaleFactors_l = mpLineextractorLeft->mvScaleFactor_l;
    mvInvScaleFactors_l = mpLineextractorLeft->mvInvScaleFactor_l;
    mvLevelSigma2_l =  mpLineextractorLeft->mvLevelSigma2_l;
    mvInvLevelSigma2_l = mpLineextractorLeft->mvInvLevelSigma2_l;



    // ------------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++ New added for background features +++++++++++++++++++++++++++
    // ------------------------------------------------------------------------------------------

    // clock_t s_1, e_1;
    // double fea_det_time;
    // s_1 = clock();
    // ORB extraction
    ExtractORB(0,imGray);
    // e_1 = clock();
    // fea_det_time = (double)(e_1-s_1)/CLOCKS_PER_SEC*1000;
    // cout << "feature detection time: " << fea_det_time << endl;


    //I am going to crop the images because i don't want lines on the edges
    // int cropSize = 10;

    // // Crop the image
    // cv::Rect roi(cropSize, cropSize, imGray.cols - 2*cropSize, imGray.rows - 2*cropSize);
    // cv::Mat imGrayCropped = imGray(roi);

    //Lines extraction
    ExtractLines(0, imGray);

    // Adjust the coordinates of the detected lines
    // for (auto& line : mvKeys_Line) {
    //     cv::Point2f start = line.getStartPoint();
    //     cv::Point2f end = line.getEndPoint();

    //     // Add the cropSize to the coordinates
    //     start.x += cropSize;
    //     start.y += cropSize;
    //     end.x += cropSize;
    //     end.y += cropSize;
        
    //     // Update the line points
    //     line.startPointX = start.x;
    //     line.startPointY = start.y;
    //     line.endPointX = end.x;
    //     line.endPointY = end.y;
    
    // }

    for (int i = 0; i < mvKeys_Line.size(); i++)
    {
        cv::line_descriptor::KeyLine l = mvKeys_Line[i];
        int x1 = l.getStartPoint().x;
        int y1 = l.getStartPoint().y;
        int x2 = l.getEndPoint().x;
        int y2 = l.getEndPoint().y;

        // Calculate the mid point
        int xm = (x1 + x2) / 2;
        int ym = (y1 + y2) / 2;

        // Get the depth at the start, mid, and end points
        float depthStart = imDepth.at<float>(y1, x1);
        float depthMid = imDepth.at<float>(ym, xm);
        float depthEnd = imDepth.at<float>(y2, x2);

        // Calculate the expected depth at the mid point if the depth change was linear
        float depthMidExpected = (depthStart + depthEnd) / 2;

        float baseThreshold = 10.0;  // This is just an example value
        float lineLength = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
        // Adjust the threshold based on the length of the line
        float threshold = baseThreshold * (lineLength / 1000);

        // If the actual and expected depth at the mid point differ by more than the threshold, discard the line
        if (abs(depthMid - depthMidExpected) > threshold)
        {
            mvKeys_Line.erase(mvKeys_Line.begin() + i);
            i--;  // Decrement the counter to account for the removed element
        }
    }

    for (int i = 0; i < mvKeys_Line.size(); ++i)
    {
        if (maskSEM.at<int>(mvKeys_Line[i].getStartPoint().y, mvKeys_Line[i].getStartPoint().x) != maskSEM.at<int>(mvKeys_Line[i].getEndPoint().y, mvKeys_Line[i].getEndPoint().x))   
            {
                mvKeys_Line.erase(mvKeys_Line.begin() + i);
                i--;  // Decrement the counter to account for the removed element
            }
    }

    //Number of keypoints and number of lines
    N = mvKeys.size();
    N_l = mvKeys_Line.size();
    
    cout << "number of features detected in image " << N << endl;
    cout << "number of lines detected in image " << N_l << endl;
    //Visualise lines
    
    cv::Mat img_show = imGray.clone();
    for (const auto& line : mvKeys_Line) {
    cv::Point2f start = line.getStartPoint();  // Get the start point of the line
    cv::Point2f end = line.getEndPoint();      // Get the end point of the line

    cv::line(img_show, start, end, cv::Scalar(0, 0, 255), 2);  // Draw the line on the image
    }

    cv::imshow("Lines on Image", img_show);  // Show the image with lines
    //cv::waitKey(0);

    // std::vector<int> labels;
    // int numberOfLines = cv::partition(mvKeys_Line, labels, isEqual);

    // std::cout << "Number of lines " << mvKeys_Line.size() << std::endl;
    // std::cout << "Number of clusters " << labels.size() << std::endl;
    // std::vector<double> line_angles;
    // std::vector<double> line_distances_from_origin;
    // std::vector<std::vector<int> > possible_line_connections;
    // std::vector<int> index_line_connections(N_l, -1);
    // //Find angles with x axis and distance from origin of lines
    // //angle
    // for (int i = 0; i < N_l; ++i)
    // {
    //     const cv::line_descriptor::KeyLine &line = mvKeys_Line[i];
    //     double angle;
    //     angle = atan2((line.endPointY - line.startPointY), (line.endPointX - line.startPointX));
    //     line_angles.push_back(angle);

    //     //distance from origin
    //     Eigen::Vector3d start_point; start_point << mvKeys_Line[i].startPointX, mvKeys_Line[i].startPointY, 1;
    //     Eigen::Vector3d end_point; end_point << mvKeys_Line[i].endPointX, mvKeys_Line[i].endPointY, 1;
    //     //Cross Product devided by norm ...
    //     Eigen::Vector3d inf_line; inf_line << (start_point.cross(end_point)).normalized();
    //     double distance_from_origin;
    //     distance_from_origin = inf_line(2);
    //     line_distances_from_origin.push_back(distance_from_origin);
    // }    


    // std::cout << "N_l = " << N_l  << std::endl;
    // //ind possible line segments to be connected
    // for (int i = 0; i < N_l; ++i)
    // {
    //     for (int j = i; j < N_l; ++j)
    //     {
    //         if (i == j)
    //             continue;
    //         std::cout << "I am stuck ye ye" << std::endl;
    //         if (abs(line_distances_from_origin[j] - line_distances_from_origin[i]) < 5 && abs(line_angles[j] - line_angles[i])  < 0.07)
    //         {
    //             std::cout << "Possible lines to connect found" << endl;
    //             //put them in a vector
    //             std::vector<int> possible_line_connection;
    //             //lines we not connected to another before
    //             if (index_line_connections[i] == -1 && index_line_connections[j] ==-1)
    //             {
    //                 possible_line_connection.push_back(i);
    //                 possible_line_connection.push_back(j);
    //                 possible_line_connections.push_back(possible_line_connection);
    //                 index_line_connections[i] = possible_line_connections.size() - 1;
    //                 index_line_connections[j] = possible_line_connections.size() - 1;
    //             }
    //             else if (index_line_connections[i] != -1 && index_line_connections[j] ==-1)
    //             {
    //                 possible_line_connections[index_line_connections[i]].push_back(j);
    //                 index_line_connections[j] = index_line_connections[i];
    //             }
    //             else if (index_line_connections[i] == -1 && index_line_connections[j] !=-1)
    //             {
    //                 possible_line_connections[index_line_connections[j]].push_back(i);
    //                 index_line_connections[i] = index_line_connections[j];
    //             }
    //             else
    //             {
    //                 std::cout << "Both lines are already connected to another line segment" << endl;
    //             }
    //         }
    //     }

    // }
    

    if(mvKeys.empty() && mvKeys_Line.empty())
        return;

    if (UseSampleFea==0)
    {
        // // // Option I: ~~~~~~~ use detected features ~~~~~~~~~~ // // //


        //for points
        for (int i = 0; i < mvKeys.size(); ++i)
        {
            int x = mvKeys[i].pt.x;
            int y = mvKeys[i].pt.y;

            if (maskSEM.at<int>(y,x)!=0)  // new added in Jun 13 2019
                continue;

            if (imDepth.at<float>(y,x)>mThDepth || imDepth.at<float>(y,x)<=0)  // new added in Aug 21 2019
                continue;

            float flow_xe = imFlow.at<cv::Vec2f>(y,x)[0];
            float flow_ye = imFlow.at<cv::Vec2f>(y,x)[1];


            if(flow_xe!=0 && flow_ye!=0)
            {
                if(mvKeys[i].pt.x+flow_xe < imGray.cols && mvKeys[i].pt.y+flow_ye < imGray.rows && mvKeys[i].pt.x < imGray.cols && mvKeys[i].pt.y < imGray.rows)
                {
                    mvStatKeysTmp.push_back(mvKeys[i]);
                    mvCorres.push_back(cv::KeyPoint(mvKeys[i].pt.x+flow_xe,mvKeys[i].pt.y+flow_ye,0,0,0,mvKeys[i].octave,-1));
                    mvFlowNext.push_back(cv::Point2f(flow_xe,flow_ye));
                }
            }
        }
        for (int i = 0; i < mvKeys_Line.size(); ++i)
        {

            //for starting point
            Point2f start_point = mvKeys_Line[i].getStartPoint();
            Point2f end_point = mvKeys_Line[i].getEndPoint();

            int start_x = start_point.x;
            int start_y = start_point.y;
            int end_x = end_point.x;
            int end_y  = end_point.y;

            //check that the line is not on an object. If it is, put it in a different vector
            if (maskSEM.at<int>(start_y,start_x)!=0 && maskSEM.at<int>(end_y,end_x)!=0)
                {
                    if (maskSEM.at<int>(start_y,start_x)==maskSEM.at<int>(end_y,end_x))
                        mvObjKeys_Line.push_back(mvKeys_Line[i]);
                    continue;
                }

            if (maskSEM.at<int>(start_y,start_x)!=0 || maskSEM.at<int>(end_y,end_x)!=0)
                continue;

            if (fabs(start_x - end_x)<1e-6 && fabs(start_y - end_y) < 1e-6)
                continue;

            //check that the line is not in the back of the camera or too far away
            if (imDepth.at<float>(start_y,start_x)>mThDepth || imDepth.at<float>(start_y, start_x)<=0 || imDepth.at<float>(end_y,end_x)>mThDepth || imDepth.at<float>(end_y, end_x)<=0)  // new added in Aug 21 2019
                {
                //std::cout << "line depth is too large or too small" << endl;
                continue;
                }

            //optical flow of starting point and ending point
            float flow_start_xe = imFlow.at<cv::Vec2f>(start_y, start_x)[0];
            float flow_start_ye = imFlow.at<cv::Vec2f>(start_y, start_x)[1];
            float flow_end_xe = imFlow.at<cv::Vec2f>(end_y, end_x)[0];
            float flow_end_ye = imFlow.at<cv::Vec2f>(end_y, end_x)[1];

            //like in the case of points, create correspondences for the lines
            if(flow_start_xe!=0 && flow_start_ye!=0 && flow_end_xe!=0 && flow_end_ye!=0)
            {
                //std::cout << "Flow is ok" << endl;
                if(start_x+flow_start_xe < imGray.cols && start_y+flow_start_ye < imGray.rows && 
                end_x+flow_end_xe < imGray.cols && end_y+flow_end_ye < imGray.rows &&
                start_x + flow_start_xe > 0 && start_y + flow_start_ye > 0 &&
                end_x + flow_end_xe > 0 && end_y + flow_end_ye > 0)
                {
                    //create the line from the starting point and ending point
                    mvStatKeysLineTmp.push_back(mvKeys_Line[i]);
                    //std::cout << "Size of mvStatKeysLineTmp: " << mvStatKeysLineTmp.size() << endl;
                    //create correspondence for line
                    KeyLine corr_line;
                    corr_line.startPointX = start_x+flow_start_xe;
                    corr_line.startPointY = start_y+flow_start_ye;
                    corr_line.endPointX = end_x+flow_end_xe;
                    corr_line.endPointY = end_y+flow_end_ye;
                    corr_line.octave = mvKeys_Line[i].octave;
                    corr_line.angle = atan2( (corr_line.endPointY-corr_line.startPointY), (corr_line.endPointX-corr_line.startPointX) );
                    corr_line.pt = Point2f((corr_line.startPointX+corr_line.endPointX)/2, (corr_line.startPointY+corr_line.endPointY)/2);
                    corr_line.size = (corr_line.endPointX - corr_line.startPointX) * ( corr_line.endPointY - corr_line.startPointY );
                    corr_line.lineLength = (float) sqrt( pow((corr_line.endPointX-corr_line.startPointX),2) + pow((corr_line.endPointY-corr_line.startPointY),2) );

                    //these are wrong, but i think i won't use them anyway
                    corr_line.response = 0;
                    corr_line.class_id = -1;
                    corr_line.numOfPixels = 0;

                    mvCorresLine.push_back(corr_line);
                    mvFlowNext_Line.push_back(make_pair(cv::Point2f(flow_start_xe,flow_start_ye), cv::Point2f(flow_end_xe,flow_end_ye)));

                    //Store the infinite lines that correspond to each line segment in corr_line

                    Eigen::Vector3d start_point; start_point << corr_line.startPointX, corr_line.startPointY, 1;
                    Eigen::Vector3d end_point; end_point << corr_line.endPointX, corr_line.endPointY, 1;
                    //Cross Product devided by norm ...
                    Eigen::Vector3d inf_line; inf_line << (start_point.cross(end_point)).normalized();
                    mvInfiniteLinesCorr.push_back(inf_line);
                }
                // else {
                //     //std::cout << "Positions not ok" << endl;
                // }
            }
            // else {
            //     //std::cout << "Flow is not ok" << endl;
            // }

        }
    }
    else
    {
        // // // Option II: ~~~~~~~ use sampled features ~~~~~~~~~~ // // //

        clock_t s_1, e_1;
        double fea_det_time;
        s_1 = clock();
        std::vector<cv::KeyPoint> mvKeysSamp = SampleKeyPoints(imGray.rows, imGray.cols);
        e_1 = clock();
        fea_det_time = (double)(e_1-s_1)/CLOCKS_PER_SEC*1000;
        std::cout << "feature detection time: " << fea_det_time << std::endl;

        for (int i = 0; i < mvKeysSamp.size(); ++i)
        {
            int x = mvKeysSamp[i].pt.x;
            int y = mvKeysSamp[i].pt.y;

            if (maskSEM.at<int>(y,x)!=0)  // new added in Jun 13 2019
                continue;

            if (imDepth.at<float>(y,x)>mThDepth || imDepth.at<float>(y,x)<=0)  // new added in Aug 21 2019
                continue;

            float flow_xe = imFlow.at<cv::Vec2f>(y,x)[0];
            float flow_ye = imFlow.at<cv::Vec2f>(y,x)[1];


            if(flow_xe!=0 && flow_ye!=0)
            {
                if(mvKeysSamp[i].pt.x+flow_xe < imGray.cols && mvKeysSamp[i].pt.y+flow_ye < imGray.rows && mvKeysSamp[i].pt.x+flow_xe>0 && mvKeysSamp[i].pt.y+flow_ye>0)
                {
                    mvStatKeysTmp.push_back(mvKeysSamp[i]);
                    mvCorres.push_back(cv::KeyPoint(mvKeysSamp[i].pt.x+flow_xe,mvKeysSamp[i].pt.y+flow_ye,0,0,0,mvKeysSamp[i].octave,-1));
                    mvFlowNext.push_back(cv::Point2f(flow_xe,flow_ye));

                }
            }
        }

        // I dont think lines need to be sampled as they are already quite less than features
        for (int i = 0; i < mvKeys_Line.size(); ++i)
        {
            //for starting point
            Point2f start_point = mvKeys_Line[i].getStartPoint();
            Point2f end_point = mvKeys_Line[i].getEndPoint();

            int start_x = start_point.x;
            int start_y = start_point.y;
            int end_x = end_point.x;
            int end_y  = end_point.y;

            //check that the line is not an object
            if (maskSEM.at<int>(start_y,start_x)!=0 && maskSEM.at<int>(end_y,end_x)!=0)  // new added in Jun 13 2019               
                {
                if (maskSEM.at<int>(start_y,start_x)==maskSEM.at<int>(end_y,end_x))
                    mvObjKeys_Line.push_back(mvKeys_Line[i]);
                continue;
                }

            if (maskSEM.at<int>(start_y,start_x)!=0 || maskSEM.at<int>(end_y,end_x)!=0)
                continue;   

            //check that the line is not in the back of the camera or too far away
            if (imDepth.at<float>(start_y,start_x)>mThDepth || imDepth.at<float>(start_y, start_x)<=0 || imDepth.at<float>(end_y,end_x)>mThDepth || imDepth.at<float>(end_y, end_x)<=0)  // new added in Aug 21 2019
                continue;

            //optical flow of starting point and ending point
            float flow_start_xe = imFlow.at<cv::Vec2f>(start_y, start_x)[0];
            float flow_start_ye = imFlow.at<cv::Vec2f>(start_y, start_x)[1];
            float flow_end_xe = imFlow.at<cv::Vec2f>(end_y, end_x)[0];
            float flow_end_ye = imFlow.at<cv::Vec2f>(end_y, end_x)[1];

            //like in the case of points, create correspondences for the lines
            if(flow_start_xe!=0 && flow_start_ye!=0 && flow_end_xe!=0 && flow_end_ye!=0)
            {
                if(start_x+flow_start_xe < imGray.cols && start_y+flow_start_ye < imGray.rows && 
                end_x+flow_end_xe < imGray.cols && end_y+flow_end_ye < imGray.rows &&
                start_x + flow_start_xe > 0 && start_y + flow_start_ye > 0 &&
                end_x + flow_end_xe > 0 && end_y + flow_end_ye > 0)
                {
                    //create the line from the starting point and ending point
                    mvStatKeysLineTmp.push_back(mvKeys_Line[i]);
                    //create correspondence for line
                    KeyLine corr_line;
                    corr_line.startPointX = start_x+flow_start_xe;
                    corr_line.startPointY = start_y+flow_start_ye;
                    corr_line.endPointX = end_x+flow_end_xe;
                    corr_line.endPointY = end_y+flow_end_ye;
                    corr_line.octave = mvKeys_Line[i].octave;
                    corr_line.angle = atan2( (corr_line.endPointY-corr_line.startPointY), (corr_line.endPointX-corr_line.startPointX) );
                    corr_line.pt = Point2f((corr_line.startPointX+corr_line.endPointX)/2, (corr_line.startPointY+corr_line.endPointY)/2);
                    corr_line.size = (corr_line.endPointX - corr_line.startPointX) * ( corr_line.endPointY - corr_line.startPointY );
                    corr_line.lineLength = (float) sqrt( pow((corr_line.endPointX-corr_line.startPointX),2) + pow((corr_line.endPointY-corr_line.startPointY),2) );

                    //these are wrong, but i think i won't use them anyway
                    corr_line.response = 0;
                    corr_line.class_id = -1;
                    corr_line.numOfPixels = 0;

                    mvCorresLine.push_back(corr_line);
                    mvFlowNext_Line.push_back(make_pair(cv::Point2f(flow_start_xe,flow_start_ye), cv::Point2f(flow_end_xe,flow_end_ye)));

                    //Store the infinite lines that correspond to each line segment in corr_line

                    Eigen::Vector3d start_point; start_point << corr_line.startPointX, corr_line.startPointY, 1;
                    Eigen::Vector3d end_point; end_point << corr_line.endPointX, corr_line.endPointY, 1;
                    //Cross Product devided by norm ...
                    Eigen::Vector3d inf_line; inf_line << (start_point.cross(end_point)).normalized();
                    mvInfiniteLinesCorr.push_back(inf_line);
                }
            }

        }
    }

    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------

    // cv::Mat img_show;
    // cv::drawKeypoints(imGray, mvKeysSamp, img_show, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    // cv::imshow("KeyPoints on Background", img_show);
    // cv::waitKey(0);

    N_s_tmp = mvStatKeysTmp.size();
    N_s_line_tmp = mvStatKeysLineTmp.size();
    // cout << "number of random sample points: " << mvCorres.size() << endl;
    // assign the depth value to each keypoint
    mvStatDepthTmp = vector<float>(N_s_tmp,-1);
    for(int i=0; i<N_s_tmp; i++)
    {
        const cv::KeyPoint &kp = mvStatKeysTmp[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        float d = imDepth.at<float>(v,u); // be careful with the order  !!!

        if(d>0)
            mvStatDepthTmp[i] = d;
    }

    // assign the depth value to each line
    mvStatDepthLineTmp = vector<pair<float, float>>(N_s_line_tmp, make_pair(-1,-1));
    for (int i=0; i < N_s_line_tmp; i++) 
    {
        const KeyLine &kl = mvStatKeysLineTmp[i];

        const float &v_start = kl.startPointY;
        const float &u_start = kl.startPointX;
        const float &v_end = kl.endPointY;
        const float &u_end = kl.endPointX;

        float d_start = imDepth.at<float>(v_start,u_start);
        float d_end = imDepth.at<float>(v_end,u_end);

        if (d_start>0 && d_end>0)
            mvStatDepthLineTmp[i].first = d_start;
            mvStatDepthLineTmp[i].second = d_end;
    }

    // ---------------------------------------------------------------------------------------
    // ++++++++++++++++++++++++++++ New added for dense object features ++++++++++++++++++++++
    // ---------------------------------------------------------------------------------------

    // semi-dense features on objects
    int step = 4; // 3
    for (int i = 0; i < imGray.rows; i=i+step)
    {
        for (int j = 0; j < imGray.cols; j=j+step)
        {
            // std::cout << "Size of imGray " << imGray.size() << std::endl;
            // std::cout << "SIze of maskSEM: " << maskSEM.size() << std::endl;
            // std::cout << "SIze of imDepth: " << imDepth.size() << std::endl;
            // std::cout << "i is " << i << " and j is " << j << std::endl;
            // try
            // {
            //     std::cout << "maskSEM.at<int>(1000,300) is " << maskSEM.at<int>(300, 1000) << std::endl;
            // }
            // catch(const std::exception& e)
            // {
            //     std::cerr << e.what() << '\n';
            // }

            // check ground truth motion mask
            if (maskSEM.at<int>(i,j)!=0 && imDepth.at<float>(i,j)<mThDepthObj && imDepth.at<float>(i,j)>0)
            {
                // get flow
                const float flow_x = imFlow.at<cv::Vec2f>(i,j)[0];
                const float flow_y = imFlow.at<cv::Vec2f>(i,j)[1];

                if(j+flow_x < imGray.cols && j+flow_x > 0 && i+flow_y < imGray.rows && i+flow_y > 0)
                {
                    // save correspondences
                    mvObjFlowNext.push_back(cv::Point2f(flow_x,flow_y));
                    mvObjCorres.push_back(cv::KeyPoint(j+flow_x,i+flow_y,0,0,0,-1));
                    // save pixel location
                    mvObjKeys.push_back(cv::KeyPoint(j,i,0,0,0,-1));
                    // save depth
                    mvObjDepth.push_back(imDepth.at<float>(i,j));
                    // save label
                    vSemObjLabel.push_back(maskSEM.at<int>(i,j));
                }
            }
        }
    }

    cout << "Number of lines on objects " << mvObjKeys_Line.size() << endl;

    // now for lines. Iterate mvObjKeys_Line
    for (int i = 0; i < mvObjKeys_Line.size(); i++)
    {   
        Point2f start_point = mvObjKeys_Line[i].getStartPoint();
        Point2f end_point = mvObjKeys_Line[i].getEndPoint();
        if (fabs(start_point.x -end_point.x) < 1e-6 && fabs(start_point.y - end_point.y) < 1e-6 )
        {
            //remove from mvoBJKeys_Line
            mvObjKeys_Line.erase(mvObjKeys_Line.begin() + i);
            i--;
            continue;
        }

        float start_depth = imDepth.at<float>(start_point.y, start_point.x);
        float end_depth = imDepth.at<float>(end_point.y, end_point.x);
        if (start_depth<mThDepthObj && start_depth > 0 && end_depth<mThDepthObj && end_depth > 0)
        {
            const float start_flow_x = imFlow.at<cv::Vec2f>(mvObjKeys_Line[i].getStartPoint().y , mvObjKeys_Line[i].getStartPoint().x)[0];
            const float start_flow_y = imFlow.at<cv::Vec2f>(mvObjKeys_Line[i].getStartPoint().y, mvObjKeys_Line[i].getStartPoint().x)[1];
            const float end_flow_x = imFlow.at<cv::Vec2f>(mvObjKeys_Line[i].getEndPoint().y, mvObjKeys_Line[i].getEndPoint().x)[0];
            const float end_flow_y = imFlow.at<cv::Vec2f>(mvObjKeys_Line[i].getEndPoint().y, mvObjKeys_Line[i].getEndPoint().x)[1];
            
            //check if the flow is going to end up in the image
            if (start_point.x + start_flow_x < imGray.cols && start_point.x + start_flow_x > 0 && start_point.y + start_flow_y < imGray.rows && start_point.y + start_flow_y > 0 && end_point.x + end_flow_x < imGray.cols && end_point.x + end_flow_x > 0 && end_point.y + end_flow_y < imGray.rows && end_point.y + end_flow_y > 0)
            {
                //i store it also here because if in some case we dont get in the if there will be no way to correspond the keylines with the correspondences
                mvObjKeys_Linetmp.push_back(mvObjKeys_Line[i]);
                //save correspondences
                mvObjFlowNext_Line.push_back(make_pair(Point2f(start_flow_x,start_flow_y),Point2f(end_flow_x,end_flow_y)));
                KeyLine corr_line;
                corr_line.startPointX = start_point.x + start_flow_x;
                corr_line.startPointY = start_point.y + start_flow_y;
                corr_line.endPointX = end_point.x + end_flow_x;
                corr_line.endPointY = end_point.y + end_flow_y;
                corr_line.octave = mvObjKeys_Line[i].octave;
                corr_line.angle = atan2( (corr_line.endPointY-corr_line.startPointY), (corr_line.endPointX-corr_line.startPointX) );
                corr_line.pt = Point2f((corr_line.startPointX+corr_line.endPointX)/2, (corr_line.startPointY+corr_line.endPointY)/2);
                corr_line.size = (corr_line.endPointX - corr_line.startPointX) * ( corr_line.endPointY - corr_line.startPointY );
                corr_line.lineLength = (float) sqrt( pow((corr_line.endPointX-corr_line.startPointX),2) + pow((corr_line.endPointY-corr_line.startPointY),2) );         
                //these are wrong, but i think i won't use them anyway
                corr_line.response = 0;
                corr_line.class_id = -1;
                corr_line.numOfPixels = 0;
                mvObjCorres_Line.push_back(corr_line);
                //save depth
                mvObjDepth_line.push_back(make_pair(start_depth,end_depth));
                //save label (I suppose that both start and end points have the same label, so I just save the start point label)
                vSemObjLabel_Line.push_back(maskSEM.at<int>(start_point.y, start_point.x)); 
            }
            else 
            {
                //remove from mvoBJKeys_Line
                mvObjKeys_Line.erase(mvObjKeys_Line.begin() + i);
                i--;
            }
        }
        else
        {
            //remove from mvoBJKeys_Line
            mvObjKeys_Line.erase(mvObjKeys_Line.begin() + i);
            i--;
        }
    }

    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();

    cout << "Constructing Frame, Done!" << endl;
}


void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0){
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
        // cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create(400);
        // cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
        // f2d->compute(im, mvKeys, mSift);
    }
    else{
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
        // cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create(400);
        // cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
        // f2d->compute(im, mvKeysRight, mSiftRight);
    }
}

void Frame::ExtractLines(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpLineextractorLeft)(im,cv::Mat(),mvKeys_Line,mDescriptors_Line);
    else
        (*mpLineextractorRight)(im,cv::Mat(),mvKeysRight_Line,mDescriptorsRight_Line);
}


void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    // mTcw.at<float>(0,0)=Tcw.at<float>(0,0);mTcw.at<float>(0,1)=Tcw.at<float>(0,1);mTcw.at<float>(0,2)=Tcw.at<float>(0,2);mTcw.at<float>(0,3)=Tcw.at<float>(0,3);
    // mTcw.at<float>(1,0)=Tcw.at<float>(1,0);mTcw.at<float>(1,1)=Tcw.at<float>(1,1);mTcw.at<float>(1,2)=Tcw.at<float>(1,2);mTcw.at<float>(1,3)=Tcw.at<float>(1,3);
    // mTcw.at<float>(2,0)=Tcw.at<float>(2,0);mTcw.at<float>(2,1)=Tcw.at<float>(2,1);mTcw.at<float>(2,2)=Tcw.at<float>(2,2);mTcw.at<float>(2,3)=Tcw.at<float>(2,3);
    // mTcw.at<float>(3,0)=Tcw.at<float>(3,0);mTcw.at<float>(3,1)=Tcw.at<float>(3,1);mTcw.at<float>(3,2)=Tcw.at<float>(3,2);mTcw.at<float>(2,3)=Tcw.at<float>(3,3);
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u); // be careful with the order

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        // cout << "xyz: " << u << " " << v << " " << z << endl;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

cv::Mat Frame::UnprojectStereoStat(const int &i, const bool &addnoise)
{
    float z = mvStatDepth[i];

    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        z = z + rng.gaussian(z*z/(725*0.5)*0.15);  // sigma = z*0.01
        // z = z + 0.0;
    }

    if(z>0)
    {
        const float u = mvStatKeys[i].pt.x;
        const float v = mvStatKeys[i].pt.y;
        // cout << "xyz: " << u << " " << v << " " << z << endl;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        // using ground truth
        const cv::Mat Rlw = mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat Rwl = Rlw.t();
        const cv::Mat tlw = mTcw.rowRange(0,3).col(3);
        const cv::Mat twl = -Rlw.t()*tlw;

        return Rwl*x3Dc+twl;
        // return mRwc*x3Dc+mOw;
    }
    else
    {
        cout << "found a depth value < 0 ..." << endl;
        return cv::Mat();
    }
}

//NOTE: i think the tmps must be used because in the optimisation we want the lines found in the previous frame and the corresponding lines (we get by adding the flow) in the current frame. If we do not put the tmps, we are using the corresponding lines the previous frame from the previous previous frame.
std::pair<cv::Mat, cv::Mat> Frame::UnprojectStereoStatLine(const int &i, const bool &addnoise)
{
    // std::cout << "i " << i << std::endl;
    // std::cout << "mvStatDepth_Line->size() " << mvStatDepth_Line.size() << std::endl;
    std::pair<float, float> z = mvStatDepth_Line[i];

//    std::cout << "We are here unproject1" << std::endl;
    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        z.first = z.first + rng.gaussian(z.first*z.first/(725*0.5)*0.15);  // sigma = z*0.01
        z.second = z.second + rng.gaussian(z.second*z.second/(725*0.5)*0.15);  // sigma = z*0.01
    }

    if (z.first > 0 && z.second > 0)
    {
 //       std::cout << "mvStatKeys_Line->size() " << mvStatKeys_Line.size() << std::endl;
        const float u_start = (mvStatKeys_Line[i].getStartPoint()).x;
        const float v_start = (mvStatKeys_Line[i].getStartPoint()).y;
        const float u_end = (mvStatKeys_Line[i].getEndPoint()).x;
        const float v_end = (mvStatKeys_Line[i].getEndPoint()).y;

        const float x_start = (u_start-cx)*z.first*invfx;
        const float y_start = (v_start-cy)*z.first*invfy;
        const float x_end = (u_end-cx)*z.second*invfx;
        const float y_end = (v_end-cy)*z.second*invfy;

        cv::Mat x3Dc_start = (cv::Mat_<float>(3,1) << x_start, y_start, z.first);
        cv::Mat x3Dc_end = (cv::Mat_<float>(3,1) << x_end, y_end, z.second);

        // using ground truth
        const cv::Mat Rlw = mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat Rwl = Rlw.t();
        const cv::Mat tlw = mTcw.rowRange(0,3).col(3);
        const cv::Mat twl = -Rlw.t()*tlw;

        // Unproject the line endpoints to world coordinates
        cv::Mat x3Dw_start = Rwl * x3Dc_start + twl;
        cv::Mat x3Dw_end = Rwl * x3Dc_end + twl;

        // Return the 3D points in the world frame
        return std::make_pair(x3Dw_start, x3Dw_end);
    }
    else {
        cout << "Found depth values < 0 ..." << endl;
        return std::make_pair(cv::Mat(), cv::Mat());
    }
}

cv::Mat Frame::UnprojectStereoObject(const int &i, const bool &addnoise)
{
    float z = mvObjDepth[i];

    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        float noise = rng.gaussian(z*z/(725*0.5)*0.15);
        z = z + noise;  // sigma = z*0.01
        // z = z + 0.0;
        // cout << "noise: " << noise << endl;
    }

    if(z>0)
    {
        const float u = mvObjKeys[i].pt.x;
        const float v = mvObjKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        const cv::Mat Rlw = mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat Rwl = Rlw.t();
        const cv::Mat tlw = mTcw.rowRange(0,3).col(3);
        const cv::Mat twl = -Rlw.t()*tlw;

        return Rwl*x3Dc+twl;
    }
    else{
        cout << "found a depth value < 0 ..." << endl;
        return cv::Mat();
    }
}

//Now for lines
std::pair<cv::Mat, cv::Mat> Frame::UnprojectStereoObjectLine(const int &i, const bool &addnoise)
{
    std::pair<float, float> z = mvObjDepth_line[i];

    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        z.first = z.first + rng.gaussian(z.first*z.first/(725*0.5)*0.15);
        z.second = z.second + rng.gaussian(z.second*z.second/(725*0.5)*0.15);
    }

    if(z.first > 0 && z.second > 0)
    {
        const float u_start = (mvObjKeys_Line[i].getStartPoint()).x;
        const float v_start = (mvObjKeys_Line[i].getStartPoint()).y;
        const float u_end = (mvObjKeys_Line[i].getEndPoint()).x;
        const float v_end = (mvObjKeys_Line[i].getEndPoint()).y;

        const float x_start = (u_start-cx)*z.first*invfx;
        const float y_start = (v_start-cy)*z.first*invfy;
        const float x_end = (u_end-cx)*z.second*invfx;
        const float y_end = (v_end-cy)*z.second*invfy;

        cv::Mat x3Dc_start = (cv::Mat_<float>(3,1) << x_start, y_start, z.first);
        cv::Mat x3Dc_end = (cv::Mat_<float>(3,1) << x_end, y_end, z.second);

        const cv::Mat Rlw = mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat Rwl = Rlw.t();
        const cv::Mat tlw = mTcw.rowRange(0,3).col(3);
        const cv::Mat twl = -Rlw.t()*tlw;

        // Unproject the line endpoints to world coordinates
        cv::Mat x3Dw_start = Rwl * x3Dc_start + twl;
        cv::Mat x3Dw_end = Rwl * x3Dc_end + twl;

        // Return the 3D points in the world frame
        return std::make_pair(x3Dw_start, x3Dw_end);
    }
    else {
        cout << "Found depth values < 0 ..." << endl;
        return std::make_pair(cv::Mat(), cv::Mat());
    }
}

cv::Mat Frame::UnprojectStereoObjectCamera(const int &i, const bool &addnoise)
{
    float z = mvObjDepth[i];
    // cout << "depth check: " << z << endl;

    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        // sigma = z*0.01
        z = z + rng.gaussian(z*z/(725*0.5)*0.15);  // sigma = z*0.01
        // z = z + 0.0;
    }

    if(z>0)
    {
        const float u = mvObjKeys[i].pt.x;
        const float v = mvObjKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        return x3Dc;
    }
    else{
        cout << "found a depth value < 0 ..." << endl;
        return cv::Mat();
    }
}

cv::Mat Frame::UnprojectStereoObjectNoise(const int &i, const cv::Point2f of_error)
{
    float z = mvObjDepth[i];

    // if(addnoise){
    //     z = z + rng.gaussian(z*0.01);  // sigma = z*0.01
    // }

    if(z>0)
    {
        const float u = mvObjKeys[i].pt.x + of_error.x;
        const float v = mvObjKeys[i].pt.y + of_error.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        // using ground truth
        const cv::Mat Rlw = mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat Rwl = Rlw.t();
        const cv::Mat tlw = mTcw.rowRange(0,3).col(3);
        const cv::Mat twl = -Rlw.t()*tlw;

        return Rwl*x3Dc+twl;
    }
    else{
        cout << "found a depth value < 0 ..." << endl;
        return cv::Mat();
    }
}

cv::Mat Frame::ObtainFlowDepthObject(const int &i, const bool &addnoise)
{
    float z = mvObjDepth[i];

    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        z = z + rng.gaussian(z*z/(725*0.5)*0.15);  // sigma = z*0.01 or z*z/(725*0.5)*0.12
        // z = z + 0.0;
    }

    if(z>0)
    {
        const float flow_u = mvObjFlowNext[i].x;
        const float flow_v = mvObjFlowNext[i].y;

        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << flow_u, flow_v, z);

        return x3Dc;
    }
    else{
        cout << "found a depth value < 0 ..." << endl;
        return cv::Mat();
    }
}

std::pair<cv::Mat, cv::Mat> Frame::ObtainFlowDepthObject_Line(const int &i, const bool &addnoise)
{
    std::pair<float, float> z = mvObjDepth_line[i];

    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        z.first = z.first + rng.gaussian(z.first*z.first/(725*0.5)*0.15);  // sigma = z*0.01 or z*z/(725*0.5)*0.12
        z.second = z.second + rng.gaussian(z.second*z.second/(725*0.5)*0.15); 
        // z = z + 0.0;
    }

    if(z.first>0 && z.second > 0)
    {
        const float flow_u_start = mvObjFlowNext_Line[i].first.x;
        const float flow_v_start = mvObjFlowNext_Line[i].first.y;
        const float flow_u_end = mvObjFlowNext_Line[i].second.x;
        const float flow_v_end = mvObjFlowNext_Line[i].second.y;

        cv::Mat x3Dc_start = (cv::Mat_<float>(3,1) << flow_u_start, flow_v_start, z.first);
        cv::Mat x3Dc_end = (cv::Mat_<float>(3,1) << flow_u_end, flow_v_end, z.second);
        std::pair<cv::Mat, cv::Mat> x3Dc = std::make_pair(x3Dc_start, x3Dc_end);
        return x3Dc;
    }
    else
    {
        cout << "found a depth value < 0 ..." << endl;
        return std::make_pair(cv::Mat(), cv::Mat());
    }
}

cv::Mat Frame::ObtainFlowDepthCamera(const int &i, const bool &addnoise)
{
    float z = mvStatDepth[i];

    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        z = z + rng.gaussian(z*z/(725*0.5)*0.15);  // sigma = z*0.01 or z*z/(725*0.5)*0.12
        // z = z + 0.0;
    }

    if(z>0)
    {
        const float flow_u = mvFlowNext[i].x;
        const float flow_v = mvFlowNext[i].y;

        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << flow_u, flow_v, z);

        return x3Dc;
    }
    else
    {
        cout << "found a depth value < 0 ..." << endl;
        return cv::Mat();
    }
}

// Stat Depth is wrong for the same reasons as mentioned in unproject. TODO !!!
std::pair<cv::Mat, cv::Mat> Frame::ObtainFlowDepthCamera_Line(const int &i, const bool &addnoise)
{
    //std::cout << "ObtainFlowDepthCamera_Line" << std::endl;
    //std::cout << "size of mvStatDepth_Line " << mvStatDepth_Line.size() << std::endl;
    //std::cout << "i " << i << std::endl; 
    std::pair<float, float> z = mvStatDepth_Line[i];
    //std::cout << "z: " << z.first << " " << z.second << std::endl;
    // used for adding noise
    cv::RNG rng((unsigned)time(NULL));

    if(addnoise){
        z.first = z.first + rng.gaussian(z.first*z.first/(725*0.5)*0.15);  // sigma = z*0.01 or z*z/(725*0.5)*0.12
        z.second = z.second + rng.gaussian(z.second*z.second/(725*0.5)*0.15); 
        // z = z + 0.0;
    }

    //std::cout << "size of mvflownext_line " << mvFlowNext_Line.size() << std::endl;

    if(z.first>0 && z.second > 0)
    {
        const float flow_u_start = mvFlowNext_Line[i].first.x;
        const float flow_v_start = mvFlowNext_Line[i].first.y;
        const float flow_u_end = mvFlowNext_Line[i].second.x;
        const float flow_v_end = mvFlowNext_Line[i].second.y;

        cv::Mat x3Dc_start = (cv::Mat_<float>(3,1) << flow_u_start, flow_v_start, z.first);
        cv::Mat x3Dc_end = (cv::Mat_<float>(3,1) << flow_u_end, flow_v_end, z.second);
        std::pair<cv::Mat, cv::Mat> x3Dc = std::make_pair(x3Dc_start, x3Dc_end);
        return x3Dc;
    }
    else
    {
        cout << "found a depth value < 0 ..." << endl;
        return std::make_pair(cv::Mat(), cv::Mat());
    }
}

std::vector<cv::KeyPoint> Frame::SampleKeyPoints(const int &rows, const int &cols)
{
    cv::RNG rng((unsigned)time(NULL));
    // rows = 480, cols = 640.
    int N = 3000;
    int n_div = 20;
    std::vector<cv::KeyPoint> KeySave;
    std::vector<std::vector<cv::KeyPoint> >  KeyinGrid(n_div*n_div);

    // (1) construct grid
    int x_step = cols/n_div, y_step = rows/n_div;

    // main loop
    int key_num = 0;
    while (key_num<N)
    {
        for (int i = 0; i < n_div; ++i)
        {
            for (int j = 0; j < n_div; ++j)
            {
                const float x = rng.uniform(i*x_step,(i+1)*x_step);
                const float y = rng.uniform(j*y_step,(j+1)*y_step);

                if (x>=cols || y>=rows || x<=0 || y<=0)
                    continue;

                // // check if this key point is already been used
                // float min_dist = 1000;
                // bool used = false;
                // for (int k = 0; k < KeyinGrid[].size(); ++k)
                // {
                //     float cur_dist = std::sqrt( (KeyinGrid[].pt.x-x)*(KeyinGrid[].pt.x-x) + (KeyinGrid[].pt.y-y)*(KeyinGrid[].pt.y-y) );
                //     if (cur_dist<min_dist)
                //         min_dist = cur_dist;
                //     if (min_dist<5.0)
                //     {
                //         used = true;
                //         break;
                //     }
                // }

                // if (used)
                //     continue;

                cv::KeyPoint Key_tmp = cv::KeyPoint(x,y,0,0,0,-1);
                KeyinGrid[i*n_div+j].push_back(Key_tmp);
                key_num = key_num + 1;
                if (key_num>=N)
                    break;
            }
            if (key_num>=N)
                break;
        }
    }

    // cout << "key_num: " << key_num << endl;

    // save points
    for (int i = 0; i < KeyinGrid.size(); ++i)
    {
        for (int j = 0; j < KeyinGrid[i].size(); ++j)
        {
            KeySave.push_back(KeyinGrid[i][j]);
        }
    }

    return KeySave;

}

std::vector<cv::Mat> Frame::CalculatePlucker(const std::vector<std::pair<cv::Mat,cv::Mat>> &mvLineTmp)
{
    std::vector<cv::Mat> mvPlucker;
    for (int i=0; i < mvLineTmp.size(); ++i)
    {
        cv::Mat start_point = mvLineTmp[i].first;
        cv::Mat end_point = mvLineTmp[i].second;
        cv::Mat direction = end_point - start_point;
        direction = direction / cv::norm(direction);
        cv::Mat n = start_point.cross(direction);
        cv::Mat plucker = (cv::Mat_<float>(6,1) << n.at<float>(0,0), n.at<float>(1,0), n.at<float>(2,0), direction.at<float>(0,0), direction.at<float>(1,0), direction.at<float>(2,0));

        mvPlucker.push_back(plucker);
    }
    return mvPlucker;
}


bool Frame::isEqual(const cv::line_descriptor::KeyLine& _l1, const cv::line_descriptor::KeyLine& _l2)
{
    cv::line_descriptor::KeyLine l1(_l1), l2(_l2);

    cv::Point2f dir1 = l1.getEndPoint() - l1.getStartPoint();
    float lenght1 = std::sqrt(dir1.x*dir1.x + dir1.y*dir1.y);
    cv::Point2f dir2 = l2.getEndPoint() - l2.getStartPoint();
    float length2 = std::sqrt(dir2.x*dir2.x + dir2.y*dir2.y);

    float product = dir1.x*dir2.x + dir1.y*dir2.y;

    if (fabs(product / (lenght1 * length2)) < cos(CV_PI / 30))
        return false;

    cv::Point2f midpoint1 = (l1.getEndPoint() + l1.getStartPoint()) * 0.5;

    float mx1 = midpoint1.x;
    float mx2 = midpoint1.y;
    
    cv::Point2f midpoint2 = (l2.getEndPoint() + l2.getStartPoint()) * 0.5;
    float my1 = midpoint2.x;
    float my2 = midpoint2.y;
    float dist = sqrtf((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2));

    if (dist > std::max(lenght1, length2) * 0.5f)
        return false;

    return true;
}
//used to plot lines that are going to be used in flow optimization




// //used to plot infinite lines
// void Frame::plotINFlines(int i) {
//     cv::Mat img_cpy = imGray_.clone();
//     Eigen::Vector3d line = mvStatInfiniteLines[i];

//     // Convert Eigen::Vector3d to cv::Vec3f
//     cv::Vec3f convertedLine(line[0], line[1], line[2]);

//     float a = line[0];
//     float b = line[1];
//     float c = line[2];

//     cv::Point2f p1, p2;
//     p1.x = 0;
//     p1.y = -c/b;
//     p2.x = img_cpy.cols;
//     p2.y = -(c + a*img_cpy.cols)/b;

//     cv::line(img_cpy, p1, p2, cv::Scalar(0,0,255), 1, 8);
    
//     std::cout << "a " << a << " b " << b << " c " << c << std::endl;
//     cv::imshow("infinite lines", img_cpy);
//     cv::waitKey(0);
    
// }


} //namespace SDPL_SLAM
