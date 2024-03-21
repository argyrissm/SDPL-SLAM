/**
* This file is part of VDO-SLAM.
*
* Copyright (C) 2019-2020 Jun Zhang <jun doc zhang2 at anu dot edu doc au> (The Australian National University)
* For more information see <https://github.com/halajun/DynamicObjectSLAM>
*
**/

#include "types_dyn_slam3d.h"

#include "../core/factory.h"
#include "../stuff/macros.h"

namespace g2o {

using namespace std;
using namespace Eigen;

// ************************************************************************************************

LandmarkMotionTernaryEdge::LandmarkMotionTernaryEdge() : BaseMultiEdge<3,Vector3>()
{
    resize(3);
    J.fill(0);
    J.block<3,3>(0,0) = Matrix3::Identity();
}

bool LandmarkMotionTernaryEdge::read(std::istream& is)
{
    Vector3d meas;
    for (int i=0; i<3; i++) is >> meas[i];
    setMeasurement(meas);

    for (int i=0; i<3; i++)
      for (int j=i; j<3; j++) {
        is >> information()(i,j);
        if (i!=j)
          information()(j,i) = information()(i,j);
      }
    return true;

}

bool LandmarkMotionTernaryEdge::write(std::ostream& os) const
{
    for (int i=0; i<3; i++) os  << measurement()[i] << " ";
    for (int i=0; i<3; ++i)
      for (int j=i; j<3; ++j)
        os <<  information()(i, j) << " " ;
    return os.good();
}

void LandmarkMotionTernaryEdge::computeError()
{
    VertexPointXYZ* point1 = dynamic_cast<VertexPointXYZ*>(_vertices[0]);
    VertexPointXYZ* point2 = dynamic_cast<VertexPointXYZ*>(_vertices[1]);
    VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);

    Vector3 expected = point1->estimate() - H->estimate().inverse()*point2->estimate();
    _error = expected - _measurement;
}

Vector3 LandmarkMotionTernaryEdge::returnError()
{
    VertexPointXYZ* point1 = dynamic_cast<VertexPointXYZ*>(_vertices[0]);
    VertexPointXYZ* point2 = dynamic_cast<VertexPointXYZ*>(_vertices[1]);
    VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);

    Vector3 expected = point1->estimate() - H->estimate().inverse()*point2->estimate();
    Vector3 tmp_error;
    tmp_error = expected - _measurement;
    return tmp_error;
}

void LandmarkMotionTernaryEdge::linearizeOplus()
{

    //VertexPointXYZ* point1 = dynamic_cast<VertexPointXYZ*>(_vertices[0]);
    VertexPointXYZ* point2 = dynamic_cast<VertexPointXYZ*>(_vertices[1]);
    VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);

    Vector3 invHl2 = H->estimate().inverse()*point2->estimate();

    // jacobian wrt H
    J(0,4) =  invHl2(2);
    J(0,5) = -invHl2(1);
    J(1,3) = -invHl2(2);
    J(1,5) =  invHl2(0);
    J(2,3) =  invHl2(1);
    J(2,4) = -invHl2(0);

    Eigen::Matrix<number_t,3,6,Eigen::ColMajor> Jhom = J;

    _jacobianOplus[0] = Matrix3::Identity();
    _jacobianOplus[1] = -H->estimate().inverse().rotation();
    _jacobianOplus[2] = Jhom;
}

// ************************************************************************************************



LineLandmarkMotionTernaryEdge::LineLandmarkMotionTernaryEdge() : BaseMultiEdge<2,Vector2>()
{
    resize(3);
    //J.fill(0);
    //J.block<3,3>(0,0) = Matrix3::Identity();
}

bool LineLandmarkMotionTernaryEdge::read(std::istream& is)
{
    Vector2d meas;
    for (int i=0; i<2; i++) is >> meas[i];
    setMeasurement(meas);

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++) {
        is >> information()(i,j);
        if (i!=j)
          information()(j,i) = information()(i,j);
      }
    return true;

}

bool LineLandmarkMotionTernaryEdge::write(std::ostream& os) const
{
    for (int i=0; i<2; i++) os  << measurement()[i] << " ";
    for (int i=0; i<2; ++i)
      for (int j=i; j<2; ++j)
        os <<  information()(i, j) << " " ;
    return os.good();
}

// void LineLandmarkMotionTernaryEdge::computeError()
// {
//     // VertexPointXYZ* point1 = dynamic_cast<VertexPointXYZ*>(_vertices[0]);
//     // VertexPointXYZ* point2 = dynamic_cast<VertexPointXYZ*>(_vertices[1]);
//     // VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);

//     // Vector3 expected = point1->estimate() - H->estimate().inverse()*point2->estimate();
//     // _error = expected - _measurement;
//     VertexLine* line1 = dynamic_cast<VertexLine*>(_vertices[0]);
//     VertexLine* line2 = dynamic_cast<VertexLine*>(_vertices[1]);
//     VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);
//     Eigen::Matrix<double, 6, 1> PluckerLine1 = orthonormal2plucker(line1->estimate());
//     Eigen::Matrix<double, 6, 1> PluckerLine2 = orthonormal2plucker(line2->estimate());
//     //TODO check that the below are correct!
//     Eigen::Vector3d translation = H->estimate().inverse().translation();
//     Eigen::Matrix3d rotation = H->estimate().inverse().rotation();
//     Eigen::Matrix<double, 6, 6> LineTransformation;
//     LineTransformation.block<3,3>(0,0) = rotation;
//     LineTransformation.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
//     LineTransformation.block<3,3>(3,3) = rotation;
//     Eigen::Matrix<double, 3, 3> skew_translation;
//     skew_translation << 0, -translation(2), translation(1),
//                         translation(2), 0, -translation(0),
//                         -translation(1), translation(0), 0;
//     LineTransformation.block<3,3>(0,3) = skew_translation * rotation;
//     Eigen::Matrix<double, 6, 1> PluckerLine2_transformed = LineTransformation * PluckerLine2;
//     Eigen::Matrix<double, 3, 1> error1;
//     error1 << PluckerLine1.head<3>() - PluckerLine2_transformed.head<3>();

//     Eigen::Matrix<double, 3, 1> normalised_u1 = ;
//     double error2 = acos((PluckerLine1.tail<3>() / PluckerLine1.tail<3>().norm()).dot(PluckerLine2_transformed.tail<3>() / PluckerLine2_transformed.tail<3>().norm()));

// }

Vector2 LineLandmarkMotionTernaryEdge::returnError()
{

    //Line for time k-1
    VertexLine* line1 = dynamic_cast<VertexLine*>(_vertices[0]);
    //Line for time k
    VertexLine* line2 = dynamic_cast<VertexLine*>(_vertices[1]);
    VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);
    Eigen::Matrix<double, 6, 1> PluckerLine1 = orthonormal2plucker(line1->estimate());
    Eigen::Matrix<double, 6, 1> PluckerLine2 = orthonormal2plucker(line2->estimate());
    //TODO check that the below are correct!
    //NOTE i am not putting inverse below!
    Eigen::Vector3d translation = H->estimate().translation();
    Eigen::Matrix3d rotation = H->estimate().rotation();
    Eigen::Matrix<double, 6, 6> LineTransformation;
    LineTransformation.block<3,3>(0,0) = rotation;
    LineTransformation.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
    LineTransformation.block<3,3>(3,3) = rotation;
    Eigen::Matrix<double, 3, 3> skew_translation;
    skew_translation << 0, -translation(2), translation(1),
                        translation(2), 0, -translation(0),
                        -translation(1), translation(0), 0;
    LineTransformation.block<3,3>(0,3) = skew_translation * rotation;
    Eigen::Matrix<double, 6, 1> PluckerLine1_transformed = LineTransformation * PluckerLine1;
    double error1; //distance of two lines
    //check if v1, v2 are collinear
    Eigen::Vector3d n1 = PluckerLine1_transformed.head<3>();
    Eigen::Vector3d n2 = PluckerLine2.head<3>();
    
    Eigen::Vector3d v1 = PluckerLine1_transformed.tail<3>();
    Eigen::Vector3d v2 = PluckerLine2.tail<3>();
    Eigen::Vector3d v3 = v1.cross(v2);
    double tmp;
    if (abs(v3.norm()) < 1e-6){
        //TODO
        // v2 / v1 = s
        double s;
        s = ((v2(0) / v1(0)) + (v2(1) / v1(1) )+ (v2(2) / v1(2))) / 3;
        tmp = v1.cross(n1 - (n2 / s)).norm();
        error1 = tmp / pow(v1.norm(), 2);
    }
    else{
        tmp = abs(v1.dot(n2) + v2.dot(n1));
        error1 = tmp / (v1.cross(v2)).norm();
    }

    // Angle of two lines
    double error2 = 1 - abs((PluckerLine1_transformed.tail<3>() / PluckerLine1_transformed.tail<3>().norm()).dot(PluckerLine2.tail<3>() / PluckerLine2.tail<3>().norm()));
    Vector2 tmp_error;
    tmp_error << error1, error2;
    if (std::isnan(tmp_error(0)) || std::isnan(tmp_error(1)))
    {
        std::exit(EXIT_FAILURE);
    }
    return tmp_error;
}

void LineLandmarkMotionTernaryEdge::computeError()
{
    // VertexPointXYZ* point1 = dynamic_cast<VertexPointXYZ*>(_vertices[0]);
    // VertexPointXYZ* point2 = dynamic_cast<VertexPointXYZ*>(_vertices[1]);
    // VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);

    // Vector3 expected = point1->estimate() - H->estimate().inverse()*point2->estimate();
    // _error = expected - _measurement;
    //Line for time k-1
    VertexLine* line1 = dynamic_cast<VertexLine*>(_vertices[0]);
    //Line for time k
    VertexLine* line2 = dynamic_cast<VertexLine*>(_vertices[1]);
    VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);
    Eigen::Matrix<double, 6, 1> PluckerLine1 = orthonormal2plucker(line1->estimate());
    Eigen::Matrix<double, 6, 1> PluckerLine2 = orthonormal2plucker(line2->estimate());
    //TODO check that the below are correct!
    //NOTE i am not putting inverse below!
    Eigen::Vector3d translation = H->estimate().translation();
    Eigen::Matrix3d rotation = H->estimate().rotation();
    Eigen::Matrix<double, 6, 6> LineTransformation;
    LineTransformation.block<3,3>(0,0) = rotation;
    LineTransformation.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
    LineTransformation.block<3,3>(3,3) = rotation;
    Eigen::Matrix<double, 3, 3> skew_translation;
    skew_translation << 0, -translation(2), translation(1),
                        translation(2), 0, -translation(0),
                        -translation(1), translation(0), 0;
    LineTransformation.block<3,3>(0,3) = skew_translation * rotation;
    Eigen::Matrix<double, 6, 1> PluckerLine1_transformed = LineTransformation * PluckerLine1;
    double error1; //distance of two lines
    //check if v1, v2 are collinear
    Eigen::Vector3d n1 = PluckerLine1_transformed.head<3>();
    Eigen::Vector3d n2 = PluckerLine2.head<3>();
    
    Eigen::Vector3d v1 = PluckerLine1_transformed.tail<3>();
    Eigen::Vector3d v2 = PluckerLine2.tail<3>();
    Eigen::Vector3d v3 = v1.cross(v2);
    double tmp;
    if (abs(v3.norm()) < 1e-6){
        //TODO
        // v2 / v1 = s
        double s;
        s = ((v2(0) / v1(0)) + (v2(1) / v1(1) )+ (v2(2) / v1(2))) / 3;
        tmp = v1.cross(n1 - (n2 / s)).norm();
        error1 = tmp / pow(v1.norm(), 2);
    }
    else{
        tmp = abs(v1.dot(n2) + v2.dot(n1));
        error1 = tmp / (v1.cross(v2)).norm();
    }

    // Angle of two lines
    double error2 = 1 - abs((PluckerLine1_transformed.tail<3>() / PluckerLine1_transformed.tail<3>().norm()).dot(PluckerLine2.tail<3>() / PluckerLine2.tail<3>().norm()));
    _error << error1, error2;
    // if (std::isnan(_error(0)) || std::isnan(_error(1)))
    // {
    //     std::cout << "error1: " << error1 << std::endl;
    //     std::cout << "error2: " << error2 << std::endl;
    //     std::cout << "PluckerLine1_transformed: " << PluckerLine1_transformed << std::endl;
    //     std::cout << "PluckerLine1: " << PluckerLine1 << std::endl;
    //     std::cout << "PluckerLine2: " << PluckerLine2 << std::endl;
    //     std::cout << "v1: " << v1 << std::endl;
    //     std::cout << "v2: " << v2 << std::endl;
    //     std::cout << "v3: " << v3 << std::endl;
    //     std::cout << "n1: " << n1 << std::endl;
    //     std::cout << "n2: " << n2 << std::endl;
    //     std::cout << "tmp: " << tmp << std::endl;
    //     std::cout << "v1.cross(v2): " << v1.cross(v2) << std::endl;
    //     std::cout << "v1.cross(n2): " << v1.cross(n2) << std::endl;
    //     std::cout << "v2.cross(n1): " << v2.cross(n1) << std::endl;
    //     std::cout << "v1.norm(): " << v1.norm() << std::endl;
    //     std::cout << "v2.norm(): " << v2.norm() << std::endl;
    //     std::cout << "v1.cross(v2).norm(): " << v1.cross(v2).norm() << std::endl;
    //     std::cout << "v1.cross(n2).norm(): " << v1.cross(n2).norm() << std::endl;
    //     std::cout << "v2.cross(n1).norm(): " << v2.cross(n1).norm() << std::endl;
    //     std::cout << "v1.dot(n2): " << v1.dot(n2) << std::endl;
    //     std::cout << "v2.dot(n1): " << v2.dot(n1) << std::endl;
    //     std::cout << "v1.cross(v2).dot(v1.cross(n2)): " << v1.cross(v2).dot(v1.cross(n2)) << std::endl;
    //     std::cout << "v1.cross(v2).dot(v2.cross(n1)): " << v1.cross(v2).dot(v2.cross(n1)) << std::endl;
    //     std::cout << "v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)): " << v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)) << std::endl;
    //     std::cout << "v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)): " << v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)) << std::endl;
    //     std::cout << "v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)): " << v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)) << std::endl;
    //     std::cout << "v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)): " << v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)) << std::endl;
    //     std::cout << "v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)): " << v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)) << std::endl;
    //     std::cout << "v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)): " << v1.cross(v2).dot(v1.cross(n2)) + v1.cross(v2).dot(v2.cross(n1)) << std::endl;

    //     std::exit(EXIT_FAILURE);
    // }
    
}

// void LineLandmarkMotionTernaryEdge::linearizeOplus()
// {
//     //Line for time k-1
//     VertexLine* line1 = dynamic_cast<VertexLine*>(_vertices[0]);
//     //for time k
//     VertexLine* line2 = dynamic_cast<VertexLine*>(_vertices[1]);
//     VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);

//     Eigen::Matrix<double, 6, 1> Plucker1 = orthonormal2plucker(line1->estimate());
//     Eigen::Matrix<double, 6, 1> Plucker2 = orthonormal2plucker(line2->estimate());
//     Eigen::Vector3d translation = H->estimate().translation();
//     Eigen::Matrix3d rotation = H->estimate().rotation();
//     Eigen::Matrix<double, 6, 6> LineTransformation;
//     LineTransformation.block<3,3>(0,0) = rotation;
//     LineTransformation.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
//     LineTransformation.block<3,3>(3,3) = rotation;
//     Eigen::Matrix<double, 3, 3> skew_translation;
//     skew_translation << 0, -translation(2), translation(1),
//                         translation(2), 0, -translation(0),
//                         -translation(1), translation(0), 0;
//     LineTransformation.block<3,3>(0,3) = skew_translation * rotation;
    
//     Eigen::Matrix<double, 6, 1> plucker_transformed1 = LineTransformation * Plucker1;
//     double n_2x, n_2y, n_2z, u_2x, u_2y, u_2z, n_1x, n_1y, n_1z, u_1x, u_1y, u_1z;
//     n_2x = plucker_transformed1(0);
//     n_2y = plucker_transformed1(1);
//     n_2z = plucker_transformed1(2);
//     u_2x = plucker_transformed1(3);
//     u_2y = plucker_transformed1(4);
//     u_2z = plucker_transformed1(5);
//     n_1x = Plucker2(0);
//     n_1y = Plucker2(1);
//     n_1z = Plucker2(2);
//     u_1x = Plucker2(3);
//     u_1y = Plucker2(4);
//     u_1z = Plucker2(5);
    
//     Eigen::Matrix<double, 4, 6> der_e_Lk;
//     der_e_Lk.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
//     der_e_Lk.block<3,3>(0,3) = Eigen::Matrix3d::Zero();
//     der_e_Lk.block<1,3>(3,0) = Eigen::Vector3d::Zero();
//     double nom, tmp, u1_norm, u2_norm, u1u2;
//     u1_norm = sqrt(u_1x * u_1x + u_1y * u_1y + u_1z * u_1z);
//     u2_norm = sqrt(u_2x * u_2x + u_2y * u_2y + u_2z * u_2z);
//     u1u2 = u_1x * u_2x + u_1y * u_2y + u_1z * u_2z;
//     nom = 1 / sqrt(1 - pow((u1u2/(u1_norm * u2_norm), 2)));
//     tmp = (u_2x - u_1x * (u_2y * v_1y + v_2z * v_1z)) / (u2_norm * (pow(u_1x * u_1x + 1), 0.5));
//     der_e_Lk(3,3) = nom * tmp;
//     tmp = (u_2y - u_1y * (u_2x * v_1x + v_2z * v_1z)) / (u2_norm * (pow(u_1y * u_1y + 1), 0.5));
//     der_e_Lk(3,4) = nom * tmp;
//     tmp = (u_2z - u_1z * (u_2x * v_1x + v_2y * v_1y)) / (u2_norm * (pow(u_1z * u_1z + 1), 0.5));
//     der_e_Lk(3,5) = nom * tmp;

//     Eigen::Vector3d u1, u2;
//     double w1, w2;
//     u1 = (line2->estimate()).first.col(0);
//     u2 = (line2->estimate()).first.col(1);
//     w1 = ((line2->estimate()).second)(0,0);
//     w2 = ((line2->estimate()).second)(1,0);

//     Eigen::Matrix<double,6,4> der_Lk_thetak;
//     Eigen::Matrix<double, 3, 3> Mat_tmp;
//     Mat_tmp << 0, w1 * u1(2), -w1 * u1(1),
//                 -w1 * u1(2), 0, w1 * u1(0),
//                 w1 * u1(1), -w1 * u1(0), 0;
//     der_Lk_thetak.block<3,3>(0,0) = Mat_tmp;
//     Mat_tmp << 0, w2 * u2(2), -w2 * u2(1),
//                 -w2 * u2(2), 0, w2 * u2(0),
//                 w2 * u2(1), -w2 * u2(0), 0;
//     der_Lk_thetak.block<3,3>(3,0) = Mat_tmp;
//     der_Lk_thetak.block<3,1>(0,3) = -w2 * u1;
//     der_Lk_thetak.block<3,1>(3,3) = w1 * u2;

//     _jacobianOplus[1] = der_e_Lk * der_Lk_thetak;


// //     Vector3 invHl2 = H->estimate().inverse()*point2->estimate();

// //     // jacobian wrt H
// //     J(0,4) =  invHl2(2);
// //     J(0,5) = -invHl2(1);
// //     J(1,3) = -invHl2(2);
// //     J(1,5) =  invHl2(0);
// //     J(2,3) =  invHl2(1);
// //     J(2,4) = -invHl2(0);

// //     Eigen::Matrix<number_t,3,6,Eigen::ColMajor> Jhom = J;

// //     _jacobianOplus[0] = Matrix3::Identity();
// //     _jacobianOplus[1] = -H->estimate().inverse().rotation();
// //     _jacobianOplus[2] = Jhom;
// // }



void LineLandmarkMotionTernaryEdge::linearizeOplus()
{
    //Line for time k-1
    VertexLine* line1 = dynamic_cast<VertexLine*>(_vertices[0]);
    //for time k
    VertexLine* line2 = dynamic_cast<VertexLine*>(_vertices[1]);
    VertexSE3* H = static_cast<VertexSE3*>(_vertices[2]);

    Eigen::Matrix<double, 6, 1> Plucker1 = orthonormal2plucker(line1->estimate());
    Eigen::Matrix<double, 6, 1> Plucker2 = orthonormal2plucker(line2->estimate());
    Eigen::Vector3d translation = H->estimate().translation();
    Eigen::Matrix3d rotation = H->estimate().rotation();
    Eigen::Matrix<double, 6, 6> LineTransformation;
    LineTransformation.block<3,3>(0,0) = rotation;
    LineTransformation.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
    LineTransformation.block<3,3>(3,3) = rotation;
    Eigen::Matrix<double, 3, 3> skew_translation;
    skew_translation << 0, -translation(2), translation(1),
                        translation(2), 0, -translation(0),
                        -translation(1), translation(0), 0;
    LineTransformation.block<3,3>(0,3) = skew_translation * rotation;
    
    Eigen::Matrix<double, 6, 1> plucker_transformed1 = LineTransformation * Plucker1;
    double n21, n22, n23, u21, u22, u23, n11, n12, n13, u11, u12, u13;
    n21 = plucker_transformed1(0);
    n22 = plucker_transformed1(1);
    n23 = plucker_transformed1(2);
    u21 = plucker_transformed1(3);
    u22 = plucker_transformed1(4);
    u23 = plucker_transformed1(5);
    n11 = Plucker2(0);
    n12 = Plucker2(1);
    n13 = Plucker2(2);
    u11 = Plucker2(3);
    u12 = Plucker2(4);
    u13 = Plucker2(5);
    
    Eigen::Matrix<double, 2, 6> der_e_Lk;
    // der_e_Lk.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    // der_e_Lk.block<3,3>(0,3) = Eigen::Matrix3d::Zero();
    // der_e_Lk.block<1,3>(3,0) = Eigen::Vector3d::Zero();
    
    const double &u_1x = u11;
    const double &u_1y = u12;
    const double &u_1z = u13;
    const double &u_2x = u21;
    const double &u_2y = u22;
    const double &u_2z = u23;
    const double &n_1x = n11;
    const double &n_1y = n12;
    const double &n_1z = n13;
    const double &n_2x = n21;
    const double &n_2y = n22;
    const double &n_2z = n23;


    //Distance related error
    Eigen::Vector3d u1_, u2_;
    u1_ << u_1x, u_1y, u_1z;
    u2_ << u_2x, u_2y, u_2z;
    double s;
    s = ((u2_(0) / u1_(0)) + (u2_(1) / u1_(1) )+ (u2_(2) / u1_(2))) / 3;
    Eigen::Vector3d u3 = u1_.cross(u2_);
    
    //if the below becomes true then if we are in the first case (lines not collinear) thenwe need to take in negative of the  derivative der_e_Lk and der_e_Lkh
    bool negat = false;

    if (u11 * n21 + u12 * n22 + u13 * n23 + n11 * u21 + n12 * u22 + n13 * u23 < 0)
    {
        negat = true;
    }
    if (abs(u3.norm()) < 1e-6)
    {
        //First case: u1 and u2 are collinear
        //wrt to nx
        der_e_Lk(0, 0) = ((u12*(u12*(n11-n21/s)-u11*(n12-n22/s))*2.0+u13*(u13*(n11-n21/s)-u11*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0)))/((u11*u11)*2.0+(u12*u12)*2.0+(u13*u13)*2.0);
        //wrt to ny
        der_e_Lk(0, 1) = ((u11*(u12*(n11-n21/s)-u11*(n12-n22/s))*2.0-u13*(u13*(n12-n22/s)-u12*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(-1.0/2.0))/(u11*u11+u12*u12+u13*u13);
        //wrt to nz
        der_e_Lk(0, 2) = ((u11*(u13*(n11-n21/s)-u11*(n13-n23/s))*2.0+u12*(u13*(n12-n22/s)-u12*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(-1.0/2.0))/(u11*u11+u12*u12+u13*u13);
        //wtr to ux
        der_e_Lk(0, 3) = -(u11*sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*2.0+(((n12-n22/s)*(u12*(n11-n21/s)-u11*(n12-n22/s))*2.0+(n13-n23/s)*(u13*(n11-n21/s)-u11*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(u11*u11+u12*u12+u13*u13))/2.0)*1.0/pow(u11*u11+u12*u12+u13*u13,2.0);
        //wrt to uy
        der_e_Lk(0, 4) = -(u12*sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*2.0-(((n11-n21/s)*(u12*(n11-n21/s)-u11*(n12-n22/s))*2.0-(n13-n23/s)*(u13*(n12-n22/s)-u12*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(u11*u11+u12*u12+u13*u13))/2.0)*1.0/pow(u11*u11+u12*u12+u13*u13,2.0);
        //wrt to uz 
        der_e_Lk(0, 5) = -(u13*sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*2.0-(((n11-n21/s)*(u13*(n11-n21/s)-u11*(n13-n23/s))*2.0+(n12-n22/s)*(u13*(n12-n22/s)-u12*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(u11*u11+u12*u12+u13*u13))/2.0)*1.0/pow(u11*u11+u12*u12+u13*u13,2.0);
    }
    else
    {
        //Second case: u1 and u2 are not collinear
        //wrt to nx
        der_e_Lk(0, 0) = u21*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to ny
        der_e_Lk(0, 1) = u22*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to nz
        der_e_Lk(0, 2) = u23*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to ux 
        der_e_Lk(0, 3) = (n21*sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))-((u22*(u11*u22-u12*u21)*2.0+u23*(u11*u23-u13*u21)*2.0)*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))*(n11*u21+n21*u11+n12*u22+n22*u12+n13*u23+n23*u13))/2.0)/(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to uy
        der_e_Lk(0, 4) = (n22*sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))+((u21*(u11*u22-u12*u21)*2.0-u23*(u12*u23-u13*u22)*2.0)*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))*(n11*u21+n21*u11+n12*u22+n22*u12+n13*u23+n23*u13))/2.0)/(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to uz
        der_e_Lk(0, 5) = (n23*sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))+((u21*(u11*u23-u13*u21)*2.0+u22*(u12*u23-u13*u22)*2.0)*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))*(n11*u21+n21*u11+n12*u22+n22*u12+n13*u23+n23*u13))/2.0)/(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
    
        if (negat)
        {
            der_e_Lk.block<1,6>(0,0) = -der_e_Lk.block<1,6>(0,0);
        }
    }


    // Angle related Error
    der_e_Lk.block<1, 3>(1,0) = Eigen::Vector3d::Zero();

    double nom, tmp, u1_norm, u2_norm, u1u2;
    u1_norm = sqrt(u_1x * u_1x + u_1y * u_1y + u_1z * u_1z);
    u2_norm = sqrt(u_2x * u_2x + u_2y * u_2y + u_2z * u_2z);
    u1u2 = u_1x * u_2x + u_1y * u_2y + u_1z * u_2z;
    if (u1u2/(u1_norm * u2_norm) < 0)
    {
        nom =  1 + (u1u2/(u1_norm * u2_norm));
    }
    else
    {
        nom =  1 - (u1u2/(u1_norm * u2_norm));
    }
    tmp = (u_2x - u_1x * (u_2y * u_1y + u_2z * u_1z)) / (u2_norm * pow((u_1x * u_1x + 1), 1.5));
    der_e_Lk(1,3) = nom * tmp;
    tmp = (u_2y - u_1y * (u_2x * u_1x + u_2z * u_1z)) / (u2_norm * pow((u_1y * u_1y + 1), 1.5));
    der_e_Lk(1,4) = nom * tmp;
    tmp = (u_2z - u_1z * (u_2x * u_1x + u_2y * u_1y)) / (u2_norm * pow((u_1z * u_1z + 1), 1.5));
    der_e_Lk(1,5) = nom * tmp;

    Eigen::Vector3d u1, u2;
    double w1, w2;
    u1 = (line2->estimate()).first.col(0);
    u2 = (line2->estimate()).first.col(1);
    w1 = ((line2->estimate()).second)(0,0);
    w2 = ((line2->estimate()).second)(1,0);

    Eigen::Matrix<double,6,4> der_Lk_thetak;
    Eigen::Matrix<double, 3, 3> Mat_tmp;
    Mat_tmp << 0, w1 * u1(2), -w1 * u1(1),
                -w1 * u1(2), 0, w1 * u1(0),
                w1 * u1(1), -w1 * u1(0), 0;
    der_Lk_thetak.block<3,3>(0,0) = Mat_tmp;
    Mat_tmp << 0, w2 * u2(2), -w2 * u2(1),
                -w2 * u2(2), 0, w2 * u2(0),
                w2 * u2(1), -w2 * u2(0), 0;
    der_Lk_thetak.block<3,3>(3,0) = Mat_tmp;
    der_Lk_thetak.block<3,1>(0,3) = -w2 * u1;
    der_Lk_thetak.block<3,1>(3,3) = w1 * u2;

    _jacobianOplus[1] = der_e_Lk * der_Lk_thetak;

    //Jacobian wrt to the parameters of line at time k-1
    Eigen::Matrix<double, 2, 6> der_e_Lkh;

    //Distance related error
    if (abs(u3.norm()) < 1e-6)
    {
        //second case
        //wrt to nx
        der_e_Lkh(0,0) = ((u12*(u12*(n11-n21/s)-u11*(n12-n22/s))*2.0+u13*(u13*(n11-n21/s)-u11*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0)))/((u11*u11)*2.0+(u12*u12)*2.0+(u13*u13)*2.0);
        //wrt to ny
        der_e_Lkh(0,1)= ((u11*(u12*(n11-n21/s)-u11*(n12-n22/s))*2.0-u13*(u13*(n12-n22/s)-u12*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(-1.0/2.0))/(u11*u11+u12*u12+u13*u13);
        //wrt to nz
        der_e_Lkh(0,2) = ((u11*(u13*(n11-n21/s)-u11*(n13-n23/s))*2.0+u12*(u13*(n12-n22/s)-u12*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(-1.0/2.0))/(u11*u11+u12*u12+u13*u13);
        //wrt to ux
        der_e_Lkh(0,3) = ((u12*(u12*(n11-n21/s)-u11*(n12-n22/s))*2.0+u13*(u13*(n11-n21/s)-u11*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0)))/((u11*u11)*2.0+(u12*u12)*2.0+(u13*u13)*2.0);
        //wrt to uy
        der_e_Lkh(0,4) = ((u11*(u12*(n11-n21/s)-u11*(n12-n22/s))*2.0-u13*(u13*(n12-n22/s)-u12*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(-1.0/2.0))/(u11*u11+u12*u12+u13*u13);
        //wrt to uz
        der_e_Lkh(0,5) = ((u11*(u13*(n11-n21/s)-u11*(n13-n23/s))*2.0+u12*(u13*(n12-n22/s)-u12*(n13-n23/s))*2.0)*1.0/sqrt(pow(u12*(n11-n21/s)-u11*(n12-n22/s),2.0)+pow(u13*(n11-n21/s)-u11*(n13-n23/s),2.0)+pow(u13*(n12-n22/s)-u12*(n13-n23/s),2.0))*(-1.0/2.0))/(u11*u11+u12*u12+u13*u13);
    }
    else {
        //first case
        //wrt to nx
        der_e_Lkh(0,0) = u11*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to ny
        der_e_Lkh(0,1) = u12*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to nz
        der_e_Lkh(0,2) = u13*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to ux
        der_e_Lkh(0,3)= (n11*sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))+((u12*(u11*u22-u12*u21)*2.0+u13*(u11*u23-u13*u21)*2.0)*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))*(n11*u21+n21*u11+n12*u22+n22*u12+n13*u23+n23*u13))/2.0)/(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to uy
        der_e_Lkh(0,4) = (n12*sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))-((u11*(u11*u22-u12*u21)*2.0-u13*(u12*u23-u13*u22)*2.0)*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))*(n11*u21+n21*u11+n12*u22+n22*u12+n13*u23+n23*u13))/2.0)/(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0));
        //wrt to uz
        der_e_Lkh(0,5) = (n13*sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))-((u11*(u11*u23-u13*u21)*2.0+u12*(u12*u23-u13*u22)*2.0)*1.0/sqrt(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0))*(n11*u21+n21*u11+n12*u22+n22*u12+n13*u23+n23*u13))/2.0)/(pow(u11*u22-u12*u21,2.0)+pow(u11*u23-u13*u21,2.0)+pow(u12*u23-u13*u22,2.0)); 
        if (negat)
        {
            der_e_Lkh.block<1,6>(0,0) = -der_e_Lkh.block<1,6>(0,0);
        }
    }
    //Angle related error
    der_e_Lkh.block<1, 3>(1,0) = Eigen::Vector3d::Zero();
    tmp = (u_1x - u_2x * (u_2y * u_1y + u_2z * u_1z)) / (u1_norm * pow((u_2x * u_2x + 1), 1.5));
    der_e_Lkh(1,3) = nom * tmp;
    tmp = (u_1y - u_2y * (u_2x * u_1x + u_2z * u_1z)) / (u1_norm * pow((u_2y * u_2y + 1), 1.5));
    der_e_Lkh(1,4) = nom * tmp;
    tmp = (u_1z - u_2z * (u_2x * u_1x + u_2y * u_1y)) / (u1_norm * pow((u_2z * u_2z + 1), 1.5));
    der_e_Lkh(1,5) = nom * tmp;

    Eigen::Matrix<double, 6, 6> der_Lkh_Lk1;
    der_Lkh_Lk1 = LineTransformation;

    //orthonormal representation of line at k-1 in woorld coordinate frame
    u1 = (line1->estimate()).first.col(0);
    u2 = (line1->estimate()).first.col(1);
    w1 = ((line1->estimate()).second)(0,0);
    w2 = ((line1->estimate()).second)(1,0);

    Eigen::Matrix<double,6,4> der_Lk1_thetak1;
    Mat_tmp << 0, w1 * u1(2), -w1 * u1(1),
                -w1 * u1(2), 0, w1 * u1(0),
                w1 * u1(1), -w1 * u1(0), 0;
    der_Lk1_thetak1.block<3,3>(0,0) = Mat_tmp;
    Mat_tmp << 0, w2 * u2(2), -w2 * u2(1),
                -w2 * u2(2), 0, w2 * u2(0),
                w2 * u2(1), -w2 * u2(0), 0;
    der_Lk1_thetak1.block<3,3>(3,0) = Mat_tmp;
    der_Lk1_thetak1.block<3,1>(0,3) = -w2 * u1;
    der_Lk1_thetak1.block<3,1>(3,3) = w1 * u2;

    _jacobianOplus[0] = der_e_Lkh * der_Lkh_Lk1  * der_Lk1_thetak1;

    // Jacobian wrt to the parameters of the transformation matrix H
    Eigen::Matrix<double, 6, 6> der_Lkh_xi;
    Eigen::Matrix<double, 3, 1> vector_tmp;
    vector_tmp = rotation * Plucker1.tail<3>();
    Mat_tmp << 0, vector_tmp(2), -vector_tmp(1),
                -vector_tmp(2), 0, vector_tmp(0),
                vector_tmp(1), -vector_tmp(0), 0;
    der_Lkh_xi.block<3,3>(3,0) = Mat_tmp;
    der_Lkh_xi.block<3,3>(0,3) = Mat_tmp;
    der_Lkh_xi.block<3,3>(3,3) = Eigen::Matrix3d::Zero();
    
    //for the up left block
    Mat_tmp << 0, -translation(2), translation(1),
                translation(2), 0, -translation(0),
                -translation(1), translation(0), 0;
    vector_tmp = Mat_tmp * vector_tmp;
    Mat_tmp <<  0, -vector_tmp(2), vector_tmp(1),
                vector_tmp(2), 0, -vector_tmp(0),
                -vector_tmp(1), vector_tmp(0), 0;
    vector_tmp = rotation * Plucker1.head<3>();
    Eigen::Matrix<double, 3, 3> Mat_tmp1;
    Mat_tmp1 << 0, vector_tmp(2), -vector_tmp(1),
                -vector_tmp(2), 0, vector_tmp(0),
                vector_tmp(1), -vector_tmp(0), 0;
    Mat_tmp = Mat_tmp1 - Mat_tmp;        
    
    der_Lkh_xi.block<3, 3>(0, 0) = Mat_tmp;


    _jacobianOplus[2] = der_e_Lkh * der_Lkh_xi;
}



Eigen::Matrix<double, 6, 1> LineLandmarkMotionTernaryEdge::orthonormal2plucker(std::pair<Eigen::Matrix3d, Eigen::Matrix2d> line){
    Eigen::Matrix<double, 6, 1> plucker;
    plucker.head<3>() = line.second(0,0) * line.first.col(0);
    plucker.tail<3>() = line.second(1,0) * line.first.col(1);
    return plucker;
}

// ************************************************************************************************


EdgeSE3Altitude::EdgeSE3Altitude() : BaseUnaryEdge<1, double, VertexSE3>()
{
   resize(1);
   J.fill(0);
   J[1] = 1;
}

bool EdgeSE3Altitude::read(std::istream& is)
{
    double meas;
    is >> meas;
    setMeasurement(meas);
    is >> information()(0);
    return true;
}

bool EdgeSE3Altitude::write(std::ostream& os) const
{
    os  << measurement() << " ";
    os << information()(0) << " ";
    return os.good();
}

void EdgeSE3Altitude::computeError()
{
    VertexSE3* v = static_cast<VertexSE3*>(_vertices[0]);
    _error << v->estimate().translation()[1] - _measurement;
}

void EdgeSE3Altitude::linearizeOplus()
{
    Eigen::Matrix<number_t,1,6,Eigen::RowMajor> Jhom = J;
    _jacobianOplusXi = Jhom;
}

// ************************************************************************************************

} // end namespace
