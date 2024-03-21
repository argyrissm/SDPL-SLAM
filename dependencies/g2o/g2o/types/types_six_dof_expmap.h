// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Modified by Raúl Mur Artal (2014)
// Added EdgeSE3ProjectXYZ (project using focal_length in x,y directions)
// Modified by Raúl Mur Artal (2016)
// Added EdgeStereoSE3ProjectXYZ (project using focal_length in x,y directions)
// Added EdgeSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)
// Added EdgeStereoSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)

// Modified by Jun Zhang (2019)
// Added EdgeSE3ProjectFlowDepth
// Added EdgeSE3ProjectDepth
// Added EdgeSE3ProjectFlow
// Added EdgeFlowDepthPrior
// Added EdgeDepthPrior

#ifndef G2O_SIX_DOF_TYPES_EXPMAP
#define G2O_SIX_DOF_TYPES_EXPMAP

#include "../core/base_vertex.h"
#include "../core/base_binary_edge.h"
#include "../core/base_unary_edge.h"
#include "../core/base_multi_edge.h"
#include "se3_ops.h"
#include "se3quat.h"
#include "types_sba.h"
#include <Eigen/Geometry>
#include <cmath>



namespace g2o {
namespace types_six_dof_expmap {
void init();
}

using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix6d;


/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */
class  VertexSE3Expmap : public BaseVertex<6, SE3Quat>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSE3Expmap();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  virtual void setToOriginImpl() {
    _estimate = SE3Quat();
  }

  virtual void oplusImpl(const double* update_)  {
    Eigen::Map<const Vector6d> update(update_);
    setEstimate(SE3Quat::exp(update)*estimate());
  }
};


class  EdgeSE3ProjectXYZ: public  BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZ();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs-cam_project(v1->estimate().map(v2->estimate()));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
};


class  EdgeStereoSE3ProjectXYZ: public  BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectXYZ();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(v2->estimate()),bf);
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector3d cam_project(const Vector3d & trans_xyz, const float &bf) const;

  double fx, fy, cx, cy, bf;
};

class  EdgeSE3ProjectXYZOnlyPose: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs-cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  Vector3d Xw;
  double fx, fy, cx, cy;
};

//Edge for lines. Only pose will be optimised
class EdgeSE3ProjectXYZLineOnlyPose: public BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZLineOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError() 
  {
    const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    Eigen::Vector3d obs(_measurement);
    Eigen::Matrix<double,2,3> Matrix23d;
    //start_point 
    Matrix23d(0,0) = cam_project(v1->estimate().map(Xw_s))(0); 
    Matrix23d(0,1) = cam_project(v1->estimate().map(Xw_s))(1);
    Matrix23d(0,2) = 1.0;
    //end__point
    Matrix23d(1,0) = cam_project(v1->estimate().map(Xw_e))(0); 
    Matrix23d(1,1) = cam_project(v1->estimate().map(Xw_e))(1);
    Matrix23d(1,2) = 1.0;
    //point * infinite line

    // if (std::isnan(obs(0)) || std::isnan(obs(1)) || std::isnan(obs(2)) || std::isnan(Matrix23d(0,0)) || std::isnan(Matrix23d(0,1)) || std::isnan(Matrix23d(0,2)) || std::isnan(Matrix23d(1,0)) || std::isnan(Matrix23d(1,1)) || std::isnan(Matrix23d(1,2))) {
    //   // At least one element is NaN, stop execution
    //   std::cout << "One or more elements of dist_tot is NaN. Stopping execution." << std::endl;
    //   //std::cout << "fx fy etc " << fx << " " << fy << " " << cx << " " << cy << std::endl;
    //   std::exit(EXIT_FAILURE);  // Exit the program with a failure status
    // }

    bool foundNaN = false;

    // for (int i = 0; i < 3; i++) {
    //     if (std::isnan(obs(i))) {
    //         std::cout << "obs(" << i << ") is NaN." << std::endl;
    //         foundNaN = true;
    //     }
    // }

    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         if (std::isnan(Matrix23d(i, j))) {
    //             std::cout << "Matrix23d(" << i << "," << j << ") is NaN." << std::endl;
    //             foundNaN = true;
    //         }
    //     }
    // }

    // if (foundNaN) {
    //     std::cout << "One or more elements are NaN. Stopping execution." << std::endl;
    //     std::exit(EXIT_FAILURE);
    // }

    _error = Matrix23d * obs;

    //std::cout << "Line only pose " << _error << std::endl;
  }

  bool isDepthPositive() {
    const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw_s))(2)>0.0 && (v1->estimate().map(Xw_e))(2)>0.0;
  }

  void setMeasurement(const Vector3d & m){
    _measurement = m;
  }

  virtual void linearizeOplus();

  Eigen::Vector2d cam_project(const Eigen::Vector3d & trans_xyz) const;

  //Starting and ending points of line segment 3D
  Eigen::Vector3d Xw_s;
  Eigen::Vector3d Xw_e;
  Eigen::Vector3d obs_temp;
  double fx, fy, cx, cy;

  private:
    Eigen::Matrix<double, 3, 1> _measurement;
};


class  EdgeStereoSE3ProjectXYZOnlyPose: public  BaseUnaryEdge<3, Vector3d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectXYZOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector3d obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector3d cam_project(const Vector3d & trans_xyz) const;

  Vector3d Xw;
  double fx, fy, cx, cy, bf;
};

// **************************************************************************************************

class  EdgeSE3ProjectXYZOnlyObjMotion: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZOnlyObjMotion(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs-cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  Vector3d Xw;

  // NEW: projection matrix
  Matrix<double, 3, 4> P;

};

//The above but for lines
class  EdgeSE3ProjectXYZOnlyObjMotionLine: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZOnlyObjMotionLine(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    Eigen::Vector3d obs(_measurement);
    //Eigen::Matrix<double,2,3> Matrix23d;
    //start_point 
    // Matrix23d(0,0) = cam_project(v1->estimate().map(Xw_s))(0); 
    // Matrix23d(0,1) = cam_project(v1->estimate().map(Xw_s))(1);
    // Matrix23d(0,2) = 1.0;
    // //end__point
    // Matrix23d(1,0) = cam_project(v1->estimate().map(Xw_e))(0); 
    // Matrix23d(1,1) = cam_project(v1->estimate().map(Xw_e))(1);
    // Matrix23d(1,2) = 1.0;

    Vector2d proj_start = cam_project(v1->estimate().map(Xw_s));
    Vector2d proj_end = cam_project(v1->estimate().map(Xw_e));

    Vector3d hom_proj_start, hom_proj_end;
    hom_proj_start << proj_start(0), proj_start(1), 1;
    hom_proj_end << proj_end(0), proj_end(1), 1;

    Eigen::Matrix<double,2,1> dist_tot;
    dist_tot << obs.dot(hom_proj_start), obs.dot(hom_proj_end);
    _error = dist_tot;
    //point * infinite line
    //_error = Matrix23d * obs;
  }

  bool isDepthPositive() {
    const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw_s))(2)>0.0 && (v1->estimate().map(Xw_e))(2)>0.0;
  }
  void setMeasurement(const Vector3d& m) {
    _measurement = m;
  }
  Eigen::Matrix<double, 3, 1> _measurement;

  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  Eigen::Vector3d Xw_s;
  Eigen::Vector3d Xw_e;
  
  // NEW: projection matrix
  Matrix<double, 3, 4> P;
  double fx, fy, cx, cy;

};

class  EdgeXYZPrior2: public  BaseUnaryEdge<3, Vector3d, VertexSBAPointXYZ>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeXYZPrior2(){}

  bool read(std::istream& is);
  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSBAPointXYZ* v1 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d obs(_measurement);
    _error = obs-v1->estimate();
  }

  bool isDepthPositive() {
    const VertexSBAPointXYZ* v1 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return v1->estimate()(2)>0.0;
  }

  virtual void linearizeOplus();

};

class  EdgeSE3ProjectXYZOnlyPoseBack: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZOnlyPoseBack(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector2d obs(_measurement);
    // SE3Quat est_inv = v1->estimate().inverse();
    // _error = obs-cam_project(est_inv.map(Xw));
    _error = obs-cam_project(v1->estimate().map_2(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map_2(Xw))(2)>0.0;
  }

  // virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  Vector3d Xw;
  double fx, fy, cx, cy;
};

// **********************************************************************************************************

class  EdgeSE3ProjectFlowDepth: public  BaseBinaryEdge<2, Vector2d, VertexSBAFlowDepth, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectFlowDepth(){};

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlowDepth* v2 = static_cast<const VertexSBAFlowDepth*>(_vertices[0]);
    Vector2d obs(_measurement);
    Vector3d est = v2->estimate();
    Vector3d Xw;
    Xw << (meas(0)-est(0)-cx)*est(2)/fx, (meas(1)-est(1)-cy)*est(2)/fy, est(2);
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    _error = obs-cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlowDepth* v2 = static_cast<const VertexSBAFlowDepth*>(_vertices[0]);
    Vector3d est = v2->estimate();
    Vector3d Xw;
    Xw << (meas(0)-est(0)-cx)*est(2)/fx, (meas(1)-est(1)-cy)*est(2)/fy, est(2);
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
  Matrix<double,2,1> meas;
  Matrix<double,4,4> Twl;
};

class  EdgeFlowDepthPrior: public  BaseUnaryEdge<3, Vector3d, VertexSBAFlowDepth>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeFlowDepthPrior(){}

  bool read(std::istream& is);
  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSBAFlowDepth* v1 = static_cast<const VertexSBAFlowDepth*>(_vertices[0]);
    Vector3d obs(_measurement);
    _error = obs-v1->estimate();
  }

  bool isDepthPositive() {
    const VertexSBAFlowDepth* v1 = static_cast<const VertexSBAFlowDepth*>(_vertices[0]);
    return v1->estimate()(2)>0.0;
  }

  virtual void linearizeOplus();

};

// **********************************************************************************************************

class  EdgeSE3ProjectFlow: public  BaseBinaryEdge<2, Vector2d, VertexSBAFlow, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectFlow(){};

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlow* v2 = static_cast<const VertexSBAFlow*>(_vertices[0]);
    Vector2d obs(_measurement);
    Vector2d est = v2->estimate();
    Vector3d Xw;
    Xw << (meas(0)-est(0)-cx)*depth/fx, (meas(1)-est(1)-cy)*depth/fy, depth;
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    _error = obs-cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlow* v2 = static_cast<const VertexSBAFlow*>(_vertices[0]);
    Vector2d est = v2->estimate();
    Vector3d Xw;
    Xw << (meas(0)-est(0)-cx)*depth/fx, (meas(1)-est(1)-cy)*depth/fy, depth;
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
  double depth;
  Matrix<double,2,1> meas;
  Matrix<double,4,4> Twl;
};

class  EdgeFlowPrior: public  BaseUnaryEdge<2, Vector2d, VertexSBAFlow>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeFlowPrior(){}

  bool read(std::istream& is);
  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSBAFlow* v1 = static_cast<const VertexSBAFlow*>(_vertices[0]);
    Vector2d obs(_measurement);
    // _error = obs-v1->estimate();
    _error = v1->estimate()-obs;
  }

  virtual void linearizeOplus();

};

class  EdgeFlowPriorLine: public  BaseUnaryEdge<4, Vector4d, VertexSBAFlowLine>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeFlowPriorLine(){}

  bool read(std::istream& is);
  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSBAFlowLine* v1 = static_cast<const VertexSBAFlowLine*>(_vertices[0]);
    Vector4d obs(_measurement);
    // _error = obs-v1->estimate();
    _error = v1->estimate()-obs;
  }

  virtual void linearizeOplus();

};

// **********************************************************************************************************

class  EdgeSE3ProjectFlow2: public  BaseBinaryEdge<2, Vector2d, VertexSBAFlow, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectFlow2(){};

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlow* v2 = static_cast<const VertexSBAFlow*>(_vertices[0]);
    Vector2d obs(_measurement);
    Vector2d est = v2->estimate();
    Vector3d Xw;
    Xw << (obs(0)-cx)*depth/fx, (obs(1)-cy)*depth/fy, depth;
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    _error = (obs+est) - cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlow* v2 = static_cast<const VertexSBAFlow*>(_vertices[0]);
    Vector2d obs(_measurement);
    Vector2d est = v2->estimate();
    Vector3d Xw;
    Xw << (obs(0)-cx)*depth/fx, (obs(1)-cy)*depth/fy, depth;
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
  double depth;
  Matrix<double,4,4> Twl;
};

//***********************************************************************************************************

//Edge with two optical flows and lines
class  EdgeSE3ProjectFlow2_Line2: public  BaseBinaryEdge<2, Vector2d, VertexSBAFlowLine, VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectFlow2_Line2(){};
    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    // void computeError() {
    //   Eigen::Matrix<double,4,1> dist_tot;
    //   dist_tot << 1, 1, 1, 1;
    //   _error = dist_tot;
    // }

    void check_values() {
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
        const VertexSBAFlowLine* v0 = static_cast<const VertexSBAFlowLine*>(_vertices[0]);
        Vector4d obs(_measurement);
        Vector4d est = v0->estimate();
        Vector3d Xw_start, Xw_end;
        std::cout << "Depth start: " << depth_start << std::endl;
        std::cout << "Depth end: " << depth_end << std::endl;
        Xw_start << (obs(0)-cx)*depth_start/fx, (obs(1)-cy)*depth_start/fy, depth_start;
        Xw_end << (obs(2)-cx)*depth_end/fx, (obs(3)-cy)*depth_end/fy, depth_end;
        Xw_start = Twl.block(0,0,3,3)*Xw_start + Twl.col(3).head(3);
        Xw_end = Twl.block(0,0,3,3)*Xw_end + Twl.col(3).head(3);
        Vector3d Xw_start_trans, Xw_end_trans;
        Xw_start_trans = v1->estimate().map(Xw_start);
        Xw_end_trans = v1->estimate().map(Xw_end);
        Vector2d proj_start = cam_project(v1->estimate().map(Xw_start));
        Vector2d proj_end = cam_project(v1->estimate().map(Xw_end));
        //correspondences
        Eigen::Matrix<double,2,1> P;
        P << obs(0) + v0->estimate()(0), obs(1) + v0->estimate()(1);
        Eigen::Matrix<double,2,1> Q;
        Q  << obs(2) + v0->estimate()(2), obs(3) + v0->estimate()(3);

        std::cout << "Flow estimate is " << est(0) << " " << est(1) << " " << est(2) << " " << est(3) << std::endl;
        std::cout << "P is " << P(0) << " " << P(1) << " Q is " << Q(0) << " " << Q(1) << std::endl;
        std::cout << "Projection of P is " << proj_start(0) << " " << proj_start(1) << " Projection of Q is " << proj_end(0) << " " << proj_end(1) << std::endl; 

    }
    void computeError()  {
        //v2 for pose, v1 for flow start point, v0 for flow end point
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
        const VertexSBAFlowLine* v0 = static_cast<const VertexSBAFlowLine*>(_vertices[0]);
        Vector4d obs(_measurement);
        //Vector2d est = v2->estimate();
        Vector3d Xw_start, Xw_end;
        //std::cout << "Depth start: " << depth_start << std::endl;
        //std::cout << "Depth end: " << depth_end << std::endl;
        Xw_start << (obs(0)-cx)*depth_start/fx, (obs(1)-cy)*depth_start/fy, depth_start;
        Xw_end << (obs(2)-cx)*depth_end/fx, (obs(3)-cy)*depth_end/fy, depth_end;
        Xw_start = Twl.block(0,0,3,3)*Xw_start + Twl.col(3).head(3);
        Xw_end = Twl.block(0,0,3,3)*Xw_end + Twl.col(3).head(3);

        Vector2d proj_start = cam_project(v1->estimate().map(Xw_start));
        Vector2d proj_end = cam_project(v1->estimate().map(Xw_end));
        //std::cout << "proj_start " << proj_start << std::endl;
        //std::cout << "proj_end " << proj_end << std::endl;
        //std::cout << "Observed line insinde compute error " << proj_start << " " << proj_end << std::endl;
        //correspondences
        Eigen::Matrix<double,3,1> P;
        P << obs(0) + v0->estimate()(0), obs(1) + v0->estimate()(1), 1;
        Eigen::Matrix<double,3,1> Q;
        Q  << obs(2) + v0->estimate()(2), obs(3) + v0->estimate()(3), 1;

        Vector3d line = P.cross(Q) / (P.cross(Q)).norm();

        Vector3d P_hom, Q_hom;
        P_hom << proj_start(0), proj_start(1), 1;
        Q_hom << proj_end(0), proj_end(1), 1;
        //std::cout << "Estimated line inside compute error " << P << " " << Q << std::endl;

        //distance of projection and line
        Eigen::Matrix<double,2,1> dist_tot;
        dist_tot << line.dot(P_hom), line.dot(Q); 

        // if (std::isnan(dist_tot(0)) || std::isnan(dist_tot(1))) {
        //     // At least one element is NaN, stop execution
        //     std::cout << "v1 estimate " << v1->estimate() << std::endl;
        //     std::cout << "obs " << obs << std::endl;
        //     std::cout << "Xw_start " << Xw_start << std::endl;
        //     std::cout << "Xw_end " << Xw_end << std::endl;
        //     std::cout << "test 1 " << v1->estimate().map(Xw_start) << std::endl;
        //     std::cout << "test 2 " << v1->estimate().map(Xw_end) << std::endl;
        //     std::cout << proj_start << " " << proj_end << std::endl;
        //     std::cout << "P is " << P << " Q is " << Q << std::endl;
        //     std::cout << "dist_tot " << dist_tot << std::endl;
        //     std::cout << "One or more elements of dist_tot is NaN. Stopping execution." << std::endl;
        //     //std::cout << "fx fy etc " << fx << " " << fy << " " << cx << " " << cy << std::endl;
        //     std::exit(EXIT_FAILURE);  // Exit the program with a failure status
        //} else {
            //std::cout << "P is " << P << " Q is " << Q << std::endl;
            //std::cout << "dist_tot " << dist_tot(0) << " " << dist_tot(1) << " " << dist_tot(2) << " " << dist_tot(3) << std::endl;
            _error = dist_tot;
            //_error << 1, 1, 1, 1;
        //}

        //std::cout << "EdgeFlow2_Line2 error " << _error << std::endl;

    }

    virtual void linearizeOplus();
    
    void setMeasurement(const Vector4d& m) {
      _measurement = m;
    }

    Vector2d cam_project(const Vector3d & trans_xyz) const;
    double fx, fy, cx, cy;
    double depth_start, depth_end;
    Matrix<double,4,4> Twl;
    Eigen::Matrix<double, 4, 1> _measurement;

};

//Edge with two optical flows and lines
// class  EdgeSE3ProjectFlow2_Line: public  BaseMultiEdge<4, Vector4d>{
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//     EdgeSE3ProjectFlow2_Line();
//     bool read(std::istream& is);

//     bool write(std::ostream& os) const;

//     void computeError()  {
//         //v2 for pose, v1 for flow start point, v0 for flow end point
//         // const VertexSE3Expmap* v2 = static_cast<const VertexSE3Expmap*>(_vertices[2]);
//         // const VertexSBAFlow* v1 = static_cast<const VertexSBAFlow*>(_vertices[1]);
//         // const VertexSBAFlow* v0 = static_cast<const VertexSBAFlow*>(_vertices[0]);
//         // Vector4d obs(_measurement);
//         // //Vector2d est = v2->estimate();
//         // Vector3d Xw_start, Xw_end;
//         // std::cout << "I dont know " << std::endl;
//         // std::cout << "Depth start: " << depth_start << std::endl;
//         // std::cout << "Depth end: " << depth_end << std::endl;
//         // Xw_start << (obs(0)-cx)*depth_start/fx, (obs(1)-cy)*depth_start/fy, depth_start;
//         // Xw_end << (obs(2)-cx)*depth_end/fx, (obs(3)-cy)*depth_end/fy, depth_end;
//         // Xw_start = Twl.block(0,0,3,3)*Xw_start + Twl.col(3).head(3);
//         // Xw_end = Twl.block(0,0,3,3)*Xw_end + Twl.col(3).head(3);
//         // Vector2d proj_start = cam_project(v2->estimate().map(Xw_start));
//         // Vector2d proj_end = cam_project(v2->estimate().map(Xw_end));
//         // //correspondences
//         // Eigen::Matrix<double,2,1> P;
//         // P << obs(0) + v0->estimate()(0), obs(1) + v0->estimate()(1);
//         // Eigen::Matrix<double,2,1> Q;
//         // Q  << obs(2) + v1->estimate()(0), obs(3) + v1->estimate()(1);
//         // //direction of line
//         // Eigen::Matrix<double,2,1> dir = P - Q;
//         // //distance of projection and line
//         // Eigen::Matrix<double,2,1> dist_start = (proj_start - P) - ((proj_start - P).dot(dir))*dir;
//         // Eigen::Matrix<double,2,1> dist_end = (proj_end - P) - ((proj_end - P).dot(dir))*dir;
//         // double _error_start = dist_start.norm();
//         // double _error_end = dist_end.norm();
//         //_error << _error_start, _error_end, 1, 1;
//         _error << 1, 1, 1, 1;
//     }

//   //   bool isDepthPositive() {
//   //       const VertexSE3Expmap* v2 = static_cast<const VertexSE3Expmap*>(_vertices[2]);
//   //       const VertexSBAFlow* v1 = static_cast<const VertexSBAFlow*>(_vertices[1]);
//   //       const VertexSBAFlow* v0 = static_cast<const VertexSBAFlow*>(_vertices[0]);
//   //       Vector4d obs(_measurement);
//   //       //Vector2d est = v2->estimate();
//   //       Vector3d Xw_start, Xw_end;
//   //       std::cout << "Depth start: " << depth_start << std::endl;
//   //       std::cout << "Depth end: " << depth_end << std::endl;
//   //       Xw_start << (obs(0)-cx)*depth_start/fx, (obs(1)-cy)*depth_start/fy, depth_start;
//   //       Xw_end << (obs(2)-cx)*depth_end/fx, (obs(3)-cy)*depth_end/fy, depth_end;

//   //       //To transform the points from the local coordinates of the last frame to the world frame (So Twl is the transformation as i know it T^0_{i-1})
//   //       Xw_start = Twl.block(0,0,3,3)*Xw_start + Twl.col(3).head(3);
//   //       Xw_end = Twl.block(0,0,3,3)*Xw_end + Twl.col(3).head(3);
//   //       return ((v2->estimate().map(Xw_start))(2)>0.0 && (v2->estimate().map(Xw_end))(2)>0.0);
//   // }

//     //virtual void linearizeOplus();

//     //Vector2d cam_project(const Vector3d & trans_xyz) const;
//     //Eigen::Matrix<double, 4, 1> _error;
//     double fx, fy, cx, cy;
//     double depth_start, depth_end;
//     Matrix<double,4,4> Twl;
// };


// **********************************************************************************************************

class  EdgeSE3ProjectFlowDepth2: public  BaseBinaryEdge<2, Vector2d, VertexSBAFlowDepth, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectFlowDepth2(){};

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlowDepth* v2 = static_cast<const VertexSBAFlowDepth*>(_vertices[0]);
    Vector2d obs(_measurement);
    Vector3d est = v2->estimate();
    Vector3d Xw;
    Xw << (obs(0)-cx)*est(2)/fx, (obs(1)-cy)*est(2)/fy, est(2);
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    _error = (obs+est.head(2))-cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlowDepth* v2 = static_cast<const VertexSBAFlowDepth*>(_vertices[0]);
    Vector2d obs(_measurement);
    Vector3d est = v2->estimate();
    Vector3d Xw;
    Xw << (obs(0)-cx)*est(2)/fx, (obs(1)-cy)*est(2)/fy, est(2);
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
  Matrix<double,4,4> Twl;
};

// **********************************************************************************************************

class  EdgeSE3ProjectDepth: public  BaseBinaryEdge<2, Vector2d, VertexSBADepth, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectDepth(){};

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBADepth* v2 = static_cast<const VertexSBADepth*>(_vertices[0]);
    Vector2d obs(_measurement);
    Matrix<double, 1, 1> est = v2->estimate();
    Vector3d Xw;
    Xw << (obs(0)-cx)*est(0)/fx, (obs(1)-cy)*est(0)/fy, est(0);
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    _error = (obs+flow)-cam_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBADepth* v2 = static_cast<const VertexSBADepth*>(_vertices[0]);
    Vector2d obs(_measurement);
    Matrix<double, 1, 1> est = v2->estimate();
    Vector3d Xw;
    Xw << (obs(0)-cx)*est(0)/fx, (obs(1)-cy)*est(0)/fy, est(0);
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
  Vector2d flow;
  Matrix<double,4,4> Twl;
};

class  EdgeDepthPrior: public  BaseUnaryEdge<1, Matrix<double, 1, 1>, VertexSBADepth>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeDepthPrior(){};

  bool read(std::istream& is);
  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSBADepth* v1 = static_cast<const VertexSBADepth*>(_vertices[0]);
    Matrix<double, 1, 1> obs(_measurement);
    _error = obs-v1->estimate();
  }

  virtual void linearizeOplus();

};

// **********************************************************************************************************


class  EdgeSE3ProjectFlowDepth3: public  BaseMultiEdge<2, Vector2d>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectFlowDepth3();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAFlow* v2 = static_cast<const VertexSBAFlow*>(_vertices[0]);
    const VertexSBADepth* v3 = static_cast<const VertexSBADepth*>(_vertices[2]);
    Vector2d obs(_measurement);
    Vector2d flow = v2->estimate();
    Matrix<double, 1, 1> depth = v3->estimate();
    Vector3d Xw;
    Xw << (obs(0)-cx)*depth(0)/fx, (obs(1)-cy)*depth(0)/fy, depth(0);
    Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
    _error = (obs+flow)-cam_project(v1->estimate().map(Xw));
  }

  // bool isDepthPositive() {
  //   const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
  //   const VertexSBADepth* v3 = static_cast<const VertexSBADepth*>(_vertices[2]);
  //   Vector2d obs(_measurement);
  //   Matrix<double, 1, 1> depth = v3->estimate();
  //   Vector3d Xw;
  //   Xw << (obs(0)-cx)*depth(0)/fx, (obs(1)-cy)*depth(0)/fy, depth(0);
  //   Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
  //   return (v1->estimate().map(Xw))(2)>0.0;
  // }

  // virtual void linearizeOplus();

  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
  Matrix<double,4,4> Twl;
};

// **********************************************************************************************************


// **********************************************************************************************************

} // end namespace

#endif
