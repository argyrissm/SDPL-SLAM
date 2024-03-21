#include "edge_se3_ortho_line.h"
#include "parameter_se3_offset.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <iostream>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/stuff/opengl_primitives.h"
#endif

namespace g2o {
using namespace std;

    bool EdgeSE3OrthoLine::read(std::istream& is){
        for (int i=0; i<6; i++) 
            is >> _measurement[i];
        for (int i=0; i<2; ++i)
        {
            for (int j=i; j<2; ++j)
            {
                is >>  information()(i, j);
                if (i!=j)
                    information()(j,i) = information()(i,j);
            }
        }
        return true;
    }

    bool EdgeSE3OrthoLine::write(std::ostream& os) const{
        for (int i=0; i<6; i++) 
            os << _measurement[i] << " ";
        for (int i=0; i<2; ++i)
        {
            for (int j=i; j<2; ++j)
                os <<  information()(i, j) << " ";
        }
        return os.good();
    }

    Vector2d EdgeSE3OrthoLine::returnError(){
        VertexSE3* v1 = static_cast<VertexSE3*>(_vertices[0]);
        VertexLine* v2 = static_cast<VertexLine*>(_vertices[1]);
        //Measurement (Observation)
        Eigen::Vector3d start_point = _measurement.head<3>();
        Eigen::Vector3d end_point = _measurement.tail<3>(); 
        Eigen::Matrix<double, 6, 1> plucker = orthonormal2plucker(v2->estimate());
        Eigen::Vector3d translation = v1->estimate().translation();
        Eigen::Matrix3d rotation = v1->estimate().rotation();
        rotation.transposeInPlace();
        translation = -rotation * translation;
        // std::cout << "EdgeSE3OrthoLine: Observation start point " << start_point << std::endl;
        // std::cout << "EdgeSE3OrthoLine: Observation end point " << end_point << std::endl;
        // std::cout << "EdgeSE3OrthoLine: Rotation w->c " << rotation << std::endl;
        // std::cout << "EdgeSE3OrthoLine: Translation w->c " << translation << std::endl;
        // std::cout << "EdgeSE3OrthoLine: plucker in world coordinates" << plucker << std::endl;
        Eigen::Matrix<double, 6, 6> LineTransformation;
        LineTransformation.block<3,3>(0,0) = rotation;
        LineTransformation.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
        LineTransformation.block<3,3>(3,3) = rotation;
        Eigen::Matrix<double, 3, 3> skew_translation;
        skew_translation << 0, -translation(2), translation(1),
                            translation(2), 0, -translation(0),
                            -translation(1), translation(0), 0;
        LineTransformation.block<3,3>(0,3) = skew_translation * rotation;
        Eigen::Matrix<double, 6, 1> plucker_transformed = LineTransformation * plucker;
        //std::cout << "EdgeSE3OrthoLine: plucker in camera coordinates" << plucker_transformed << std::endl;
        Eigen::Vector3d plucker_n_c = plucker_transformed.head<3>();
        Eigen::Vector3d plucker_u_c = plucker_transformed.tail<3>(); 
        double dist1 = (start_point.cross(plucker_u_c) - plucker_n_c).norm();
        double dist2 = (end_point.cross(plucker_u_c) - plucker_n_c).norm();
        // std::cout << "EdgeSE3OrthoLine: dist1 " << dist1 << std::endl;
        // std::cout << "EdgeSE3OrthoLine: dist2 " << dist2 << std::endl;
        
        Vector2d tmp_error;
        tmp_error << dist1, dist2;
        if (std::isnan(tmp_error(0)) || std::isnan(tmp_error(1)))
        {
            std::exit(EXIT_FAILURE);
        }
        return tmp_error;
    }
    
    //Compute error
    void EdgeSE3OrthoLine::computeError(){
        VertexSE3* v1 = static_cast<VertexSE3*>(_vertices[0]);
        VertexLine* v2 = static_cast<VertexLine*>(_vertices[1]);
        //Measurement (Observation)
        Eigen::Vector3d start_point = _measurement.head<3>();
        Eigen::Vector3d end_point = _measurement.tail<3>(); 
        Eigen::Matrix<double, 6, 1> plucker = orthonormal2plucker(v2->estimate());
        Eigen::Vector3d translation = v1->estimate().translation();
        Eigen::Matrix3d rotation = v1->estimate().rotation();
        rotation.transposeInPlace();
        translation = -rotation * translation;
        Eigen::Matrix<double, 6, 6> LineTransformation;
        LineTransformation.block<3,3>(0,0) = rotation;
        LineTransformation.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
        LineTransformation.block<3,3>(3,3) = rotation;
        Eigen::Matrix<double, 3, 3> skew_translation;
        skew_translation << 0, -translation(2), translation(1),
                            translation(2), 0, -translation(0),
                            -translation(1), translation(0), 0;
        LineTransformation.block<3,3>(0,3) = skew_translation * rotation;
        Eigen::Matrix<double, 6, 1> plucker_transformed = LineTransformation * plucker;
        Eigen::Vector3d plucker_n_c = plucker_transformed.head<3>();
        Eigen::Vector3d plucker_u_c = plucker_transformed.tail<3>(); 
        double dist1 = (start_point.cross(plucker_u_c) - plucker_n_c).norm();
        double dist2 = (end_point.cross(plucker_u_c) - plucker_n_c).norm();
        _error << dist1, dist2;
        // if (std::isnan(_error(0)) || std::isnan(_error(1)))
        // {
        //     // std::cout << "v2->estimate(): " << v2->estimate().first << std::endl;
        //     // std::cout << "VertexSE3: start_point: " << start_point.transpose() << std::endl;
        //     // std::cout << "VertexSE3: end_point: " << end_point.transpose() << std::endl;
        //     // std::cout << "VertexSE3: plucker_n_c: " << plucker_n_c.transpose() << std::endl;
        //     // std::cout << "VertexSE3: plucker_u_c: " << plucker_u_c.transpose() << std::endl;
        //     // std::cout << "VertexSE3: plucker: " << plucker.transpose() << std::endl;
        //     // std::cout << "VertexSE3: plucker_transformed: " << plucker_transformed.transpose() << std::endl;
        //     // std::cout << "VertexSE3: translation: " << translation.transpose() << std::endl;
        //     // std::cout << "VertexSE3: rotation: " << rotation << std::endl;
        //     // std::cout << "VertexSE3: LineTransformation: " << LineTransformation << std::endl;
        //     // std::cout << "VertexSE3: plucker_transformed: " << plucker_transformed.transpose() << std::endl;
        //     // std::cout << "VertexSE3: plucker_n_c: " << plucker_n_c.transpose() << std::endl;
        //     // std::cout << "VertexSE3: plucker_u_c: " << plucker_u_c.transpose() << std::endl;
        //     // std::cout << "VertexSE3: dist1: " << dist1 << std::endl;
        //     // std::cout << "VertexSE3: dist2: " << dist2 << std::endl;
        //     // std::cout << "VertexSE3: _error: " << _error.transpose() << std::endl;
        //     std::cout << "Error in EdgeSE3OrthoLine::computeError() has nan" << std::endl;
        //     std::exit(EXIT_FAILURE);
        // }
    }


    //Lineariseoplus
    void EdgeSE3OrthoLine::linearizeOplus()
    {
        VertexSE3 *vi = static_cast<VertexSE3 *>(_vertices[0]);
        VertexLine *vj = static_cast<VertexLine *>(_vertices[1]);
        Eigen::Vector3d start_point = _measurement.head<3>();
        Eigen::Vector3d end_point = _measurement.tail<3>(); 
        Eigen::Matrix<double, 6, 1> plucker = orthonormal2plucker(vj->estimate());
        Eigen::Vector3d translation = vi->estimate().translation();
        Eigen::Matrix3d rotation = vi->estimate().rotation();

        //Rotation from camera to world frame
        double Rwc1_1 = rotation(0,0);
        double Rwc1_2 = rotation(0,1);
        double Rwc1_3 = rotation(0,2);
        double Rwc2_1 = rotation(1,0);
        double Rwc2_2 = rotation(1,1);
        double Rwc2_3 = rotation(1,2);
        double Rwc3_1 = rotation(2,0);
        double Rwc3_2 = rotation(2,1);
        double Rwc3_3 = rotation(2,2);
        double twc1 = translation(0);
        double twc2 = translation(1);
        double twc3 = translation(2);
        double n_w1 = plucker(0);
        double n_w2 = plucker(1);
        double n_w3 = plucker(2);
        double u_w1 = plucker(3);
        double u_w2 = plucker(4);
        double u_w3 = plucker(5);

        rotation.transposeInPlace();
        translation = -rotation * translation;
        Eigen::Matrix<double, 6, 6> LineTransformation;
        LineTransformation.block<3,3>(0,0) = rotation;
        LineTransformation.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
        LineTransformation.block<3,3>(3,3) = rotation;
        Eigen::Matrix<double, 3, 3> skew_translation;
        skew_translation << 0, -translation(2), translation(1),
                            translation(2), 0, -translation(0),
                            -translation(1), translation(0), 0;
        LineTransformation.block<3,3>(0,3) = skew_translation * rotation;
        Eigen::Matrix<double, 6, 1> plucker_transformed = LineTransformation * plucker;
        double u_c1, u_c2, u_c3, n_c1, n_c2, n_c3, x_start1, x_start2, x_start3, x_end1, x_end2, x_end3;
        n_c1 = plucker_transformed(0);
        n_c2 = plucker_transformed(1);
        n_c3 = plucker_transformed(2);
        u_c1 = plucker_transformed(3);
        u_c2 = plucker_transformed(4);
        u_c3 = plucker_transformed(5);
        x_start1 = start_point(0);
        x_start2 = start_point(1);
        x_start3 = start_point(2);
        x_end1 = end_point(0);
        x_end2 = end_point(1);
        x_end3 = end_point(2);

        // Derivative of error with respect to line parameters 2x6
        Eigen::Matrix<double, 2, 6> der_e_Lc;
        der_e_Lc(0, 0) = (1.0/sqrt(pow(n_c3+u_c1*x_start2-u_c2*x_start1,2.0)+pow(n_c2-u_c1*x_start3+u_c3*x_start1,2.0)+pow(n_c1+u_c2*x_start3-u_c3*x_start2,2.0))*(n_c1*2.0+u_c2*x_start3*2.0-u_c3*x_start2*2.0))/2.0;
        der_e_Lc(0, 1) = (1.0/sqrt(pow(n_c3+u_c1*x_start2-u_c2*x_start1,2.0)+pow(n_c2-u_c1*x_start3+u_c3*x_start1,2.0)+pow(n_c1+u_c2*x_start3-u_c3*x_start2,2.0))*(n_c2*2.0-u_c1*x_start3*2.0+u_c3*x_start1*2.0))/2.0;
        der_e_Lc(0, 2) = (1.0/sqrt(pow(n_c3+u_c1*x_start2-u_c2*x_start1,2.0)+pow(n_c2-u_c1*x_start3+u_c3*x_start1,2.0)+pow(n_c1+u_c2*x_start3-u_c3*x_start2,2.0))*(n_c3*2.0+u_c1*x_start2*2.0-u_c2*x_start1*2.0))/2.0;
        der_e_Lc(0, 3) = ((x_start2*(n_c3+u_c1*x_start2-u_c2*x_start1)*2.0-x_start3*(n_c2-u_c1*x_start3+u_c3*x_start1)*2.0)*1.0/sqrt(pow(n_c3+u_c1*x_start2-u_c2*x_start1,2.0)+pow(n_c2-u_c1*x_start3+u_c3*x_start1,2.0)+pow(n_c1+u_c2*x_start3-u_c3*x_start2,2.0)))/2.0;
        der_e_Lc(0, 4) = (x_start1*(n_c3+u_c1*x_start2-u_c2*x_start1)*2.0-x_start3*(n_c1+u_c2*x_start3-u_c3*x_start2)*2.0)*1.0/sqrt(pow(n_c3+u_c1*x_start2-u_c2*x_start1,2.0)+pow(n_c2-u_c1*x_start3+u_c3*x_start1,2.0)+pow(n_c1+u_c2*x_start3-u_c3*x_start2,2.0))*(-1.0/2.0);
        der_e_Lc(0, 5) = ((x_start1*(n_c2-u_c1*x_start3+u_c3*x_start1)*2.0-x_start2*(n_c1+u_c2*x_start3-u_c3*x_start2)*2.0)*1.0/sqrt(pow(n_c3+u_c1*x_start2-u_c2*x_start1,2.0)+pow(n_c2-u_c1*x_start3+u_c3*x_start1,2.0)+pow(n_c1+u_c2*x_start3-u_c3*x_start2,2.0)))/2.0;
        
        der_e_Lc(1, 0) = (1.0/sqrt(pow(n_c3+u_c1*x_end2-u_c2*x_end1,2.0)+pow(n_c2-u_c1*x_end3+u_c3*x_end1,2.0)+pow(n_c1+u_c2*x_end3-u_c3*x_end2,2.0))*(n_c1*2.0+u_c2*x_end3*2.0-u_c3*x_end2*2.0))/2.0;
        der_e_Lc(1, 1) = (1.0/sqrt(pow(n_c3+u_c1*x_end2-u_c2*x_end1,2.0)+pow(n_c2-u_c1*x_end3+u_c3*x_end1,2.0)+pow(n_c1+u_c2*x_end3-u_c3*x_end2,2.0))*(n_c2*2.0-u_c1*x_end3*2.0+u_c3*x_end1*2.0))/2.0;
        der_e_Lc(1, 2) = (1.0/sqrt(pow(n_c3+u_c1*x_end2-u_c2*x_end1,2.0)+pow(n_c2-u_c1*x_end3+u_c3*x_end1,2.0)+pow(n_c1+u_c2*x_end3-u_c3*x_end2,2.0))*(n_c3*2.0+u_c1*x_end2*2.0-u_c2*x_end1*2.0))/2.0;
        der_e_Lc(1, 3) = ((x_end2*(n_c3+u_c1*x_end2-u_c2*x_end1)*2.0-x_end3*(n_c2-u_c1*x_end3+u_c3*x_end1)*2.0)*1.0/sqrt(pow(n_c3+u_c1*x_end2-u_c2*x_end1,2.0)+pow(n_c2-u_c1*x_end3+u_c3*x_end1,2.0)+pow(n_c1+u_c2*x_end3-u_c3*x_end2,2.0)))/2.0;
        der_e_Lc(1, 4) = (x_end1*(n_c3+u_c1*x_end2-u_c2*x_end1)*2.0-x_end3*(n_c1+u_c2*x_end3-u_c3*x_end2)*2.0)*1.0/sqrt(pow(n_c3+u_c1*x_end2-u_c2*x_end1,2.0)+pow(n_c2-u_c1*x_end3+u_c3*x_end1,2.0)+pow(n_c1+u_c2*x_end3-u_c3*x_end2,2.0))*(-1.0/2.0);
        der_e_Lc(1, 5) = ((x_end1*(n_c2-u_c1*x_end3+u_c3*x_end1)*2.0-x_end2*(n_c1+u_c2*x_end3-u_c3*x_end2)*2.0)*1.0/sqrt(pow(n_c3+u_c1*x_end2-u_c2*x_end1,2.0)+pow(n_c2-u_c1*x_end3+u_c3*x_end1,2.0)+pow(n_c1+u_c2*x_end3-u_c3*x_end2,2.0)))/2.0;
        
        //------------------------------------
        //Derivative of line in camera coordinates wrt to line parameters in world coordinates is just the Transormation matrix
        Eigen::Matrix<double, 6, 6> der_Lc_Lw;
        der_Lc_Lw = LineTransformation;
        // -----------------------------------
        
        double w1, w2;
        Eigen::Vector3d u1, u2;
        u1 = (vj->estimate()).first.col(0);
        u2 = (vj->estimate()).first.col(1);
        w1 = ((vj->estimate()).second)(0, 0);
        w2 = ((vj->estimate()).second)(1, 0);

        //Derivative of line in world coordinates wrt to the parameters of the orthonormal representation
        Eigen::Matrix<double,6,4> der_Lw_theta;
        Eigen::Matrix<double, 3, 3> Mat_tmp;
        Mat_tmp << 0, w1 * u1(2), -w1 * u1(1),
                   -w1 * u1(2), 0, w1 * u1(0),
                   w1 * u1(1), -w1 * u1(0), 0;
        der_Lw_theta.block<3,3>(0,0) = Mat_tmp;
        Mat_tmp << 0, w2 * u2(2), -w2 * u2(1),
                   -w2 * u2(2), 0, w2 * u2(0),
                   w2 * u2(1), -w2 * u2(0), 0;
        der_Lw_theta.block<3,3>(3,0) = Mat_tmp;
        der_Lw_theta.block<3,1>(0,3) = -w2 * u1;
        der_Lw_theta.block<3,1>(3,3) = w1 * u2;

        //_jacobianOplusXj <- Jacobian wrt line parameters
        //2x6 * 6x6 * 6x4 = 2x4
        _jacobianOplusXj = der_e_Lc * der_Lc_Lw * der_Lw_theta;
         
        //_jacobianOplusXi  <- Jacobian wrt pose parameters

        Eigen::Matrix<double, 6, 6> der_Lc_xi;
        // Eigen::Matrix<double, 3, 1> vector_tmp;
        // vector_tmp = rotation * plucker.tail<3>();
        // Mat_tmp << 0, vector_tmp(2), -vector_tmp(1),
        //            -vector_tmp(2), 0, vector_tmp(0),
        //            vector_tmp(1), -vector_tmp(0), 0;
        // der_Lc_xi.block<3,3>(0,0) = Mat_tmp;
        // der_Lc_xi.block<3,3>(3,3) = Mat_tmp;
        // der_Lc_xi.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
        
        // //for the up left block
        // Mat_tmp << 0, -translation(2), translation(1),
        //            translation(2), 0, -translation(0),
        //            -translation(1), translation(0), 0;
        // vector_tmp = Mat_tmp * vector_tmp;
        // Mat_tmp <<  0, -vector_tmp(2), vector_tmp(1),
        //            vector_tmp(2), 0, -vector_tmp(0),
        //            -vector_tmp(1), vector_tmp(0), 0;
        // vector_tmp = rotation * plucker.head<3>();
        // Eigen::Matrix<double, 3, 3> Mat_tmp1;
        // Mat_tmp1 << 0, vector_tmp(2), -vector_tmp(1),
        //            -vector_tmp(2), 0, vector_tmp(0),
        //            vector_tmp(1), -vector_tmp(0), 0;
        // Mat_tmp = Mat_tmp1 - Mat_tmp;        
        
        // der_Lc_xi.block<3, 3>(0, 3) = Mat_tmp;

        Eigen::Matrix<double, 6, 3> der_Lc_drho;
        //for tranlation
        der_Lc_drho(0,0) = -u_w2*(Rwc1_2*Rwc2_3-Rwc1_3*Rwc2_2)-u_w3*(Rwc1_2*Rwc3_3-Rwc1_3*Rwc3_2);
        der_Lc_drho(0,1) = u_w1*(Rwc1_2*Rwc2_3-Rwc1_3*Rwc2_2)-u_w3*(Rwc2_2*Rwc3_3-Rwc2_3*Rwc3_2);
        der_Lc_drho(0,2) = u_w1*(Rwc1_2*Rwc3_3-Rwc1_3*Rwc3_2)+u_w2*(Rwc2_2*Rwc3_3-Rwc2_3*Rwc3_2);
        der_Lc_drho(1,0) = u_w2*(Rwc1_1*Rwc2_3-Rwc1_3*Rwc2_1)+u_w3*(Rwc1_1*Rwc3_3-Rwc1_3*Rwc3_1);
        der_Lc_drho(1,1) = -u_w1*(Rwc1_1*Rwc2_3-Rwc1_3*Rwc2_1)+u_w3*(Rwc2_1*Rwc3_3-Rwc2_3*Rwc3_1);
        der_Lc_drho(1,2) = -u_w1*(Rwc1_1*Rwc3_3-Rwc1_3*Rwc3_1)-u_w2*(Rwc2_1*Rwc3_3-Rwc2_3*Rwc3_1);
        der_Lc_drho(2,0) = -u_w2*(Rwc1_1*Rwc2_2-Rwc1_2*Rwc2_1)-u_w3*(Rwc1_1*Rwc3_2-Rwc1_2*Rwc3_1);
        der_Lc_drho(2,1) = u_w1*(Rwc1_1*Rwc2_2-Rwc1_2*Rwc2_1)-u_w3*(Rwc2_1*Rwc3_2-Rwc2_2*Rwc3_1);
        der_Lc_drho(2,2) = u_w1*(Rwc1_1*Rwc3_2-Rwc1_2*Rwc3_1)+u_w2*(Rwc2_1*Rwc3_2-Rwc2_2*Rwc3_1);
        der_Lc_drho.block<3,3>(3,0) = Eigen::Matrix3d::Zero();

        //for rotation
        Eigen::Matrix<double, 6, 3> der_Lc_dphi;

        der_Lc_dphi(0,0) = Rwc2_1*n_w3-Rwc3_1*n_w2*1.0+u_w1*(Rwc2_1*twc2+Rwc3_1*twc3)-Rwc2_1*twc1*u_w2*1.0-Rwc3_1*twc1*u_w3*1.0;
        der_Lc_dphi(0,1) = Rwc1_1*n_w3*-1.0+Rwc3_1*n_w1+u_w2*(Rwc1_1*twc1+Rwc3_1*twc3)-Rwc1_1*twc2*u_w1*1.0-Rwc3_1*twc2*u_w3*1.0;
        der_Lc_dphi(0,2) = Rwc1_1*n_w2-Rwc2_1*n_w1*1.0+u_w3*(Rwc1_1*twc1+Rwc2_1*twc2)-Rwc1_1*twc3*u_w1*1.0-Rwc2_1*twc3*u_w2*1.0;
        der_Lc_dphi(1,0) = Rwc2_2*n_w3-Rwc3_2*n_w2*1.0+u_w1*(Rwc2_2*twc2+Rwc3_2*twc3)-Rwc2_2*twc1*u_w2*1.0-Rwc3_2*twc1*u_w3*1.0;
        der_Lc_dphi(1,1) = Rwc1_2*n_w3*-1.0+Rwc3_2*n_w1+u_w2*(Rwc1_2*twc1+Rwc3_2*twc3)-Rwc1_2*twc2*u_w1*1.0-Rwc3_2*twc2*u_w3*1.0;
        der_Lc_dphi(1,2) = Rwc1_2*n_w2-Rwc2_2*n_w1*1.0+u_w3*(Rwc1_2*twc1+Rwc2_2*twc2)-Rwc1_2*twc3*u_w1*1.0-Rwc2_2*twc3*u_w2*1.0;
        der_Lc_dphi(2,0) = Rwc2_3*n_w3-Rwc3_3*n_w2*1.0+u_w1*(Rwc2_3*twc2+Rwc3_3*twc3)-Rwc2_3*twc1*u_w2*1.0-Rwc3_3*twc1*u_w3*1.0;
        der_Lc_dphi(2,1) = Rwc1_3*n_w3*-1.0+Rwc3_3*n_w1+u_w2*(Rwc1_3*twc1+Rwc3_3*twc3)-Rwc1_3*twc2*u_w1*1.0-Rwc3_3*twc2*u_w3*1.0;
        der_Lc_dphi(2,2) = Rwc1_3*n_w2-Rwc2_3*n_w1*1.0+u_w3*(Rwc1_3*twc1+Rwc2_3*twc2)-Rwc1_3*twc3*u_w1*1.0-Rwc2_3*twc3*u_w2*1.0;
        der_Lc_dphi(3,0) = Rwc2_1*u_w3-Rwc3_1*u_w2*1.0;
        der_Lc_dphi(3,1) = Rwc1_1*u_w3*-1.0+Rwc3_1*u_w1;
        der_Lc_dphi(3,2) = Rwc1_1*u_w2-Rwc2_1*u_w1*1.0;
        der_Lc_dphi(4,0) = Rwc2_2*u_w3-Rwc3_2*u_w2*1.0;
        der_Lc_dphi(4,1) = Rwc1_2*u_w3*-1.0+Rwc3_2*u_w1;
        der_Lc_dphi(4,2) = Rwc1_2*u_w2-Rwc2_2*u_w1*1.0;
        der_Lc_dphi(5,0) = Rwc2_3*u_w3-Rwc3_3*u_w2*1.0;
        der_Lc_dphi(5,1) = Rwc1_3*u_w3*-1.0+Rwc3_3*u_w1;
        der_Lc_dphi(5,2)= Rwc1_3*u_w2-Rwc2_3*u_w1*1.0;

        der_Lc_xi.block<6,3>(0,0) = der_Lc_drho;
        der_Lc_xi.block<6,3>(0,3) = der_Lc_dphi;
        // 2x6 * 6x6 * 6x6 = 2 x 6
        //_jacobianOplusXi = der_e_Lc * der_Lc_Lw * der_Lw_xi;
        _jacobianOplusXi = der_e_Lc * der_Lc_xi;
        
    }

    Eigen::Matrix<double, 6, 1> EdgeSE3OrthoLine::orthonormal2plucker(std::pair<Eigen::Matrix3d, Eigen::Matrix2d> line){
        Eigen::Matrix<double, 6, 1> plucker;
        plucker.head<3>() = line.second(0,0) * line.first.col(0);
        plucker.tail<3>() = line.second(1,0) * line.first.col(1);
        return plucker;
    }

}