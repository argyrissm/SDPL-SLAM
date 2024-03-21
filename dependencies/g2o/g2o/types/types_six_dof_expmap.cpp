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

#include "types_six_dof_expmap.h"

#include "../core/factory.h"
#include "../stuff/macros.h"

namespace g2o {

using namespace std;


Vector2d project2d(const Vector3d& v)  {
  Vector2d res;
  if (v(2) < 0 ) {
    //std::cout << "I am going to divide with 0 check the depth or negative depth" << v(2) << std::endl;
    std::exit(EXIT_FAILURE);
  }
  res(0) = v(0)/v(2);
  res(1) = v(1)/v(2);
  return res;
}

Vector3d unproject2d(const Vector2d& v)  {
  Vector3d res;
  res(0) = v(0);
  res(1) = v(1);
  res(2) = 1;
  return res;
}

VertexSE3Expmap::VertexSE3Expmap() : BaseVertex<6, SE3Quat>() {
}

bool VertexSE3Expmap::read(std::istream& is) {
  Vector7d est;
  for (int i=0; i<7; i++)
    is  >> est[i];
  SE3Quat cam2world;
  cam2world.fromVector(est);
  setEstimate(cam2world.inverse());
  return true;
}

bool VertexSE3Expmap::write(std::ostream& os) const {
  SE3Quat cam2world(estimate().inverse());
  for (int i=0; i<7; i++)
    os << cam2world[i] << " ";
  return os.good();
}


EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeSE3ProjectXYZ::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZ::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


void EdgeSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;

  Matrix<double,2,3> tmp;
  tmp(0,0) = fx;
  tmp(0,1) = 0;
  tmp(0,2) = -x/z*fx;

  tmp(1,0) = 0;
  tmp(1,1) = fy;
  tmp(1,2) = -y/z*fy;

  _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;
}

Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}


Vector3d EdgeStereoSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz, const float &bf) const{
  const float invz = 1.0f/trans_xyz[2];
  Vector3d res;
  res[0] = trans_xyz[0]*invz*fx + cx;
  res[1] = trans_xyz[1]*invz*fy + cy;
  res[2] = res[0] - bf*invz;
  return res;
}

EdgeStereoSE3ProjectXYZ::EdgeStereoSE3ProjectXYZ() : BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeStereoSE3ProjectXYZ::read(std::istream& is){
  for (int i=0; i<=3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZ::write(std::ostream& os) const {

  for (int i=0; i<=3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeStereoSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  const Matrix3d R =  T.rotation().toRotationMatrix();

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;

  _jacobianOplusXi(0,0) = -fx*R(0,0)/z+fx*x*R(2,0)/z_2;
  _jacobianOplusXi(0,1) = -fx*R(0,1)/z+fx*x*R(2,1)/z_2;
  _jacobianOplusXi(0,2) = -fx*R(0,2)/z+fx*x*R(2,2)/z_2;

  _jacobianOplusXi(1,0) = -fy*R(1,0)/z+fy*y*R(2,0)/z_2;
  _jacobianOplusXi(1,1) = -fy*R(1,1)/z+fy*y*R(2,1)/z_2;
  _jacobianOplusXi(1,2) = -fy*R(1,2)/z+fy*y*R(2,2)/z_2;

  _jacobianOplusXi(2,0) = _jacobianOplusXi(0,0)-bf*R(2,0)/z_2;
  _jacobianOplusXi(2,1) = _jacobianOplusXi(0,1)-bf*R(2,1)/z_2;
  _jacobianOplusXi(2,2) = _jacobianOplusXi(0,2)-bf*R(2,2)/z_2;

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;

  _jacobianOplusXj(2,0) = _jacobianOplusXj(0,0)-bf*y/z_2;
  _jacobianOplusXj(2,1) = _jacobianOplusXj(0,1)+bf*x/z_2;
  _jacobianOplusXj(2,2) = _jacobianOplusXj(0,2);
  _jacobianOplusXj(2,3) = _jacobianOplusXj(0,3);
  _jacobianOplusXj(2,4) = 0;
  _jacobianOplusXj(2,5) = _jacobianOplusXj(0,5)-bf/z_2;
}


//Only Pose

bool EdgeSE3ProjectXYZOnlyPose::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0/xyz_trans[2];
  double invz_2 = invz*invz;

  _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
  _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  _jacobianOplusXi(0,2) = y*invz *fx;
  _jacobianOplusXi(0,3) = -invz *fx;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x*invz_2 *fx;

  _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  _jacobianOplusXi(1,2) = -x*invz *fy;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -invz *fy;
  _jacobianOplusXi(1,5) = y*invz_2 *fy;
}

Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

Vector2d EdgeSE3ProjectXYZLineOnlyPose::cam_project(const Eigen::Vector3d & trans_xyz) const{
  const float invz = 1.0f/trans_xyz[2];
  float epsilon = 0.0000001;
  if (invz < epsilon) {
    std::cout << "I am going to divide with 0 check the depth or negative depth" << invz << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Eigen::Vector2d res;
  res[0] = trans_xyz[0]*invz*fx + cx;
  res[1] = trans_xyz[1]*invz*fy + cy;
  return res;
}

bool EdgeSE3ProjectXYZLineOnlyPose::read(std::istream& is){
  for (int i=0; i<3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
        if (i!=j)
          information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZLineOnlyPose::write(std::ostream& os) const {
    for (int i=0; i<3; i++){
        os << measurement()[i] << " ";
    }

    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++){
            os << " " <<  information()(i,j);
        }
    return os.good();
}

void EdgeSE3ProjectXYZLineOnlyPose::linearizeOplus() {
    g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
    Eigen::Vector3d xyz_trans_s = vi->estimate().map(Xw_s);  //m
    Eigen::Vector3d xyz_trans_e = vi->estimate().map(Xw_e);  //g

    double x_s = xyz_trans_s[0];  //mx
    double y_s = xyz_trans_s[1];  //my
    double invz_s = 1.0/xyz_trans_s[2];  // 1/mz
    double invz_s_2 = invz_s*invz_s;  // 1/mz^2

    double x_e = xyz_trans_e[0];  //gx
    double y_e = xyz_trans_e[1];  //gy
    double invz_e = 1.0/xyz_trans_e[2];  //  1/gz
    double invz_e_2 = invz_e*invz_e;    //  1/gz^2

    double l0 = obs_temp(0);
    double l1 = obs_temp(1);
    // Jacobian = [wrt theta1, wrt theta2, wrt theta3, wrt t1, wrt t2, wrt t3]
    _jacobianOplusXi(0,0) = -fx*x_s*y_s*invz_s_2*l0-fy*(1+y_s*y_s*invz_s_2)*l1;
    _jacobianOplusXi(0,1) = fx*(1+x_s*x_s*invz_s_2)*l0+fy*x_s*y_s*invz_s_2*l1;
    _jacobianOplusXi(0,2) = -fx*y_s*invz_s*l0+fy*x_s*invz_s*l1; 
    _jacobianOplusXi(0,3) = fx*invz_s*l0;
    _jacobianOplusXi(0,4) = fy*invz_s*l1;
    _jacobianOplusXi(0,5) = (-fx*x_s*l0-fy*y_s*l1)*invz_s_2;

    _jacobianOplusXi(1,0) = -fx*x_e*y_e*invz_e_2*l0-fy*(1+y_e*y_e*invz_e_2)*l1;
    _jacobianOplusXi(1,1) = fx*(1+x_e*x_e*invz_e_2)*l0+fy*x_e*y_e*invz_e_2*l1;
    _jacobianOplusXi(1,2) = -fx*y_e*invz_e*l0+fy*x_e*invz_e*l1;  
    _jacobianOplusXi(1,3) = fx*invz_e*l0;
    _jacobianOplusXi(1,4) = fy*invz_e*l1;
    _jacobianOplusXi(1,5) = (-fx*x_e*l0-fy*y_e*l1)*invz_e_2;
}

Vector3d EdgeStereoSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  const float invz = 1.0f/trans_xyz[2];
  Vector3d res;
  res[0] = trans_xyz[0]*invz*fx + cx;
  res[1] = trans_xyz[1]*invz*fy + cy;
  res[2] = res[0] - bf*invz;
  return res;
}


bool EdgeStereoSE3ProjectXYZOnlyPose::read(std::istream& is){
  for (int i=0; i<=3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<=3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0/xyz_trans[2];
  double invz_2 = invz*invz;

  _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
  _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  _jacobianOplusXi(0,2) = y*invz *fx;
  _jacobianOplusXi(0,3) = -invz *fx;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x*invz_2 *fx;

  _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  _jacobianOplusXi(1,2) = -x*invz *fy;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -invz *fy;
  _jacobianOplusXi(1,5) = y*invz_2 *fy;

  _jacobianOplusXi(2,0) = _jacobianOplusXi(0,0)-bf*y*invz_2;
  _jacobianOplusXi(2,1) = _jacobianOplusXi(0,1)+bf*x*invz_2;
  _jacobianOplusXi(2,2) = _jacobianOplusXi(0,2);
  _jacobianOplusXi(2,3) = _jacobianOplusXi(0,3);
  _jacobianOplusXi(2,4) = 0;
  _jacobianOplusXi(2,5) = _jacobianOplusXi(0,5)-bf*invz_2;
}

// ************************************************************************************************

bool EdgeSE3ProjectXYZOnlyObjMotion::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyObjMotion::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectXYZOnlyObjMotion::cam_project(const Vector3d & trans_xyz) const{

  double m1 = P(0,0)*trans_xyz[0] + P(0,1)*trans_xyz[1] + P(0,2)*trans_xyz[2] + P(0,3);
  double m2 = P(1,0)*trans_xyz[0] + P(1,1)*trans_xyz[1] + P(1,2)*trans_xyz[2] + P(1,3);
  double m3 = P(2,0)*trans_xyz[0] + P(2,1)*trans_xyz[1] + P(2,2)*trans_xyz[2] + P(2,3);
  double invm3 = 1.0/m3;

  Vector2d res;
  res[0] = m1*invm3;
  res[1] = m2*invm3;

  return res;
}

void EdgeSE3ProjectXYZOnlyObjMotion::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];

  double m1 = P(0,0)*x + P(0,1)*y + P(0,2)*z + P(0,3);
  double m2 = P(1,0)*x + P(1,1)*y + P(1,2)*z + P(1,3);
  double m3 = P(2,0)*x + P(2,1)*y + P(2,2)*z + P(2,3);
  double invm3 = 1.0/m3;
  double invm3_2 = invm3*invm3;

  Matrix<double,2,3> tmp;
  tmp(0,0) = invm3_2*(P(0,0)*m3-P(2,0)*m1);
  tmp(0,1) = invm3_2*(P(0,1)*m3-P(2,1)*m1);
  tmp(0,2) = invm3_2*(P(0,2)*m3-P(2,2)*m1);
  tmp(1,0) = invm3_2*(P(1,0)*m3-P(2,0)*m2);
  tmp(1,1) = invm3_2*(P(1,1)*m3-P(2,1)*m2);
  tmp(1,2) = invm3_2*(P(1,2)*m3-P(2,2)*m2);

  _jacobianOplusXi(0,0) = -1.0*( y*tmp(0,2)-z*tmp(0,1) );
  _jacobianOplusXi(0,1) = -1.0*( z*tmp(0,0)-x*tmp(0,2) );
  _jacobianOplusXi(0,2) = -1.0*( x*tmp(0,1)-y*tmp(0,0) );
  _jacobianOplusXi(0,3) = -1.0*tmp(0,0);
  _jacobianOplusXi(0,4) = -1.0*tmp(0,1);
  _jacobianOplusXi(0,5) = -1.0*tmp(0,2);

  _jacobianOplusXi(1,0) = -1.0*( y*tmp(1,2)-z*tmp(1,1) );
  _jacobianOplusXi(1,1) = -1.0*( z*tmp(1,0)-x*tmp(1,2) );
  _jacobianOplusXi(1,2) = -1.0*( x*tmp(1,1)-y*tmp(1,0) );
  _jacobianOplusXi(1,3) = -1.0*tmp(1,0);
  _jacobianOplusXi(1,4) = -1.0*tmp(1,1);
  _jacobianOplusXi(1,5) = -1.0*tmp(1,2);
}

// ************************************************************************************************
bool EdgeSE3ProjectXYZOnlyObjMotionLine::read(std::istream& is){
  for (int i=0; i<3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyObjMotionLine::write(std::ostream& os) const {

  for (int i=0; i<3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}
Vector2d EdgeSE3ProjectXYZOnlyObjMotionLine::cam_project(const Vector3d & trans_xyz) const{

  double m1 = P(0,0)*trans_xyz[0] + P(0,1)*trans_xyz[1] + P(0,2)*trans_xyz[2] + P(0,3);
  double m2 = P(1,0)*trans_xyz[0] + P(1,1)*trans_xyz[1] + P(1,2)*trans_xyz[2] + P(1,3);
  double m3 = P(2,0)*trans_xyz[0] + P(2,1)*trans_xyz[1] + P(2,2)*trans_xyz[2] + P(2,3);
  double invm3 = 1.0/m3;

  Vector2d res;
  res[0] = m1*invm3;
  res[1] = m2*invm3;

  return res;
}

void EdgeSE3ProjectXYZOnlyObjMotionLine::linearizeOplus() {

  //_jacobianOplusXi will have size 2x6

  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  SE3Quat T(vi->estimate());
  Vector3d obs(_measurement);

  Vector3d xyz_trans_start = T.map(Xw_s);
  Vector3d xyz_trans_end = T.map(Xw_e);
  double mx, my, mz, mz2, gx, gy, gz, gz2;

  mx = xyz_trans_start[0];
  my = xyz_trans_start[1];
  mz = xyz_trans_start[2];
  mz2 = mz*mz;
  gx = xyz_trans_end[0];
  gy = xyz_trans_end[1];
  gz = xyz_trans_end[2];
  gz2 = gz*gz;
  Vector2d proj_start = cam_project(vi->estimate().map(Xw_s));
  Vector2d proj_end = cam_project(vi->estimate().map(Xw_e));
  
  
  Eigen::Matrix<double, 2, 6> der_proj_start_xi, der_proj_end_xi;
  der_proj_start_xi << -fx*mx*my/mz2, fx*(1+(std::pow(mx, 2) / mz2)), -fx*(my/mz), fx/mz, 0, -fx*mx/mz2, 
    -fy*(1+(std::pow(my, 2)/mz2)), fy*mx*my/mz2, fy*mx/mz, 0, fy/mz, -fy*my/mz2;
  der_proj_end_xi << -fx*gx*gy/gz2, fx*(1+(std::pow(gx, 2) / gz2)), -fx*(gy/gz), fx/gz, 0, -fx*gx/gz2, 
    -fy*(1+(std::pow(my, 2)/mz2)), fy*gx*gy/gz2, fy*gx/gz, 0, fy/gz, -fy*gy/gz2;
  

  Eigen::Matrix<double, 2, 6> der_e_xi;
  Eigen::Matrix<double, 1, 6> der_e_xi_start, der_e_xi_end;
  Eigen::Matrix<double, 1, 2> line;
  line << obs(0), obs(1);
  der_e_xi_start = line * der_proj_start_xi;
  der_e_xi_end = line * der_proj_end_xi;
  der_e_xi.block<1, 6>(0, 0) = der_e_xi_start;
  der_e_xi.block<1, 6>(1, 0) = der_e_xi_end;

  _jacobianOplusXi = der_e_xi;
}

// ************************************************************************************************

bool EdgeXYZPrior2::read(std::istream& is){
  for (int i=0; i<3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<3; i++)
    for (int j=i; j<3; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeXYZPrior2::write(std::ostream& os) const {

  for (int i=0; i<3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<3; i++)
    for (int j=i; j<3; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeXYZPrior2::linearizeOplus(){
    _jacobianOplusXi = -1.0*Matrix3d::Identity();
}


// ************************************************************************************************

bool EdgeSE3ProjectXYZOnlyPoseBack::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPoseBack::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectXYZOnlyPoseBack::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

// void EdgeSE3ProjectXYZOnlyPoseBack::linearizeOplus() {
//   VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
//   Vector3d xyz_trans = vi->estimate().map_2(Xw);

//   double x = xyz_trans[0];
//   double y = xyz_trans[1];
//   double invz = 1.0/xyz_trans[2];
//   double invz_2 = invz*invz;

//   _jacobianOplusXi(0,0) =  -1.0*x*y*invz_2 *fx;
//   _jacobianOplusXi(0,1) = (1.0+(x*x*invz_2)) *fx;
//   _jacobianOplusXi(0,2) = -1.0*y*invz *fx;
//   _jacobianOplusXi(0,3) = invz *fx;
//   _jacobianOplusXi(0,4) = 0;
//   _jacobianOplusXi(0,5) = -1.0*x*invz_2 *fx;

//   _jacobianOplusXi(1,0) = -1.0*(1.0+y*y*invz_2) *fy;
//   _jacobianOplusXi(1,1) = x*y*invz_2 *fy;
//   _jacobianOplusXi(1,2) = x*invz *fy;
//   _jacobianOplusXi(1,3) = 0;
//   _jacobianOplusXi(1,4) = invz *fy;
//   _jacobianOplusXi(1,5) = -1.0*y*invz_2 *fy;
// }

// ************************************************************************************************

bool EdgeSE3ProjectFlowDepth::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectFlowDepth::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectFlowDepth::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

void EdgeSE3ProjectFlowDepth::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAFlowDepth* vi = static_cast<VertexSBAFlowDepth*>(_vertices[0]);
  Vector3d uvd = vi->estimate();
  Vector3d Xw;
  Xw << (meas(0)-uvd(0)-cx)*uvd(2)/fx, (meas(1)-uvd(1)-cy)*uvd(2)/fy, uvd(2);
  Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
  Vector3d xyz_trans = T.map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;
  double invfx = 1.0/fx;
  double invfy = 1.0/fy;

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;

  Matrix<double,2,3> K;
  K(0,0) = fx; K(0,1) = 0;  K(0,2) = cx;
  K(1,0) = 0;  K(1,1) = fy; K(1,2) = cy;

  Matrix<double,3,4> T_mat;
  T_mat.block(0,0,3,3) = T.rotation().toRotationMatrix();
  T_mat.col(3) = T.translation();

  Matrix<double,2,4> A;
  A = K*T_mat*Twl;

  _jacobianOplusXi(0,0) = A(0,0)*uvd(2)*invfx;
  _jacobianOplusXi(0,1) = A(0,1)*uvd(2)*invfy;
  _jacobianOplusXi(0,2) = -1.0*( A(0,0)*(meas(0)-uvd(0)-cx)*invfx + A(0,1)*(meas(1)-uvd(1)-cy)*invfy + A(0,2) );

  _jacobianOplusXi(1,0) = A(1,0)*uvd(2)*invfx;
  _jacobianOplusXi(1,1) = A(1,1)*uvd(2)*invfy;
  _jacobianOplusXi(1,2) = -1.0*( A(1,0)*(meas(0)-uvd(0)-cx)*invfx + A(1,1)*(meas(1)-uvd(1)-cy)*invfy + A(1,2) );

}

// ************************************************************************************************

bool EdgeFlowDepthPrior::read(std::istream& is){
  for (int i=0; i<3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<3; i++)
    for (int j=i; j<3; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeFlowDepthPrior::write(std::ostream& os) const {

  for (int i=0; i<3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<3; i++)
    for (int j=i; j<3; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeFlowDepthPrior::linearizeOplus(){
    _jacobianOplusXi = -1.0*Matrix3d::Identity();
}

// *****************************************************************************************************************

bool EdgeSE3ProjectFlow::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectFlow::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectFlow::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

void EdgeSE3ProjectFlow::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAFlow* vi = static_cast<VertexSBAFlow*>(_vertices[0]);
  Vector2d f_uv = vi->estimate();
  Vector3d Xw;
  Xw << (meas(0)-f_uv(0)-cx)*depth/fx, (meas(1)-f_uv(1)-cy)*depth/fy, depth;
  Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
  Vector3d xyz_trans = T.map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;
  double invfx = 1.0/fx;
  double invfy = 1.0/fy;

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;

  Matrix<double,2,3> K;
  K(0,0) = fx; K(0,1) = 0;  K(0,2) = cx;
  K(1,0) = 0;  K(1,1) = fy; K(1,2) = cy;

  Matrix<double,3,4> T_mat;
  T_mat.block(0,0,3,3) = T.rotation().toRotationMatrix();
  T_mat.col(3) = T.translation();

  Matrix<double,2,4> A;
  A = K*T_mat*Twl;

  _jacobianOplusXi(0,0) = A(0,0)*depth*invfx;
  _jacobianOplusXi(0,1) = A(0,1)*depth*invfy;

  _jacobianOplusXi(1,0) = A(1,0)*depth*invfx;
  _jacobianOplusXi(1,1) = A(1,1)*depth*invfy;

}

// ************************************************************************************************

bool EdgeFlowPrior::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeFlowPrior::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeFlowPrior::linearizeOplus(){
  // _jacobianOplusXi = -1.0*Matrix2d::Identity();
  _jacobianOplusXi = Matrix2d::Identity();
}

// ************************************************************************************************


// ************************************************************************************************

bool EdgeFlowPriorLine::read(std::istream& is){
  for (int i=0; i<4; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<4; i++)
    for (int j=i; j<4; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeFlowPriorLine::write(std::ostream& os) const {

  for (int i=0; i<4; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<4; i++)
    for (int j=i; j<4; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeFlowPriorLine::linearizeOplus(){
  // _jacobianOplusXi = -1.0*Matrix2d::Identity();
  _jacobianOplusXi = Matrix4d::Identity();
}

// ************************************************************************************************

bool EdgeSE3ProjectFlow2::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectFlow2::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectFlow2::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}


//Jacobian with respect to flow is just set to identity
void EdgeSE3ProjectFlow2::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAFlow* vi = static_cast<VertexSBAFlow*>(_vertices[0]);
  Vector2d obs(_measurement);
  Vector2d f_uv = vi->estimate();
  Vector3d Xw;
  Xw << (obs(0)-cx)*depth/fx, (obs(1)-cy)*depth/fy, depth;
  Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
  Vector3d xyz_trans = T.map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;

  _jacobianOplusXi = Matrix2d::Identity();

}

// ************************************************************************************************


bool EdgeSE3ProjectFlow2_Line2::read(std::istream& is){
  for (int i=0; i<4; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectFlow2_Line2::write(std::ostream& os) const {

  for (int i=0; i<4; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectFlow2_Line2::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}


void EdgeSE3ProjectFlow2_Line2::linearizeOplus() {
  //std::cout << "Test line oplus1" << std::endl;
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAFlowLine* vi = static_cast<VertexSBAFlowLine*>(_vertices[0]);
  Vector4d obs(_measurement);
  Vector3d Xw_start, Xw_end;

  //std::cout << "Test line oplus2" << std::endl;
  Xw_start << (obs(0)-cx)*depth_start/fx, (obs(1)-cy)*depth_start/fy, depth_start;
  Xw_end << (obs(2)-cx)*depth_end/fx, (obs(3)-cy)*depth_end/fy, depth_end;
  Xw_start = Twl.block(0,0,3,3)*Xw_start + Twl.col(3).head(3);
  Xw_end = Twl.block(0,0,3,3)*Xw_end + Twl.col(3).head(3);
  Vector3d xyz_trans_start = T.map(Xw_start);
  Vector3d xyz_trans_end = T.map(Xw_end);
  double mx, my, mz, mz2, gx, gy, gz, gz2;
  mx = xyz_trans_start[0];
  my = xyz_trans_start[1];
  mz = xyz_trans_start[2];
  mz2 = mz*mz;
  gx = xyz_trans_end[0];
  gy = xyz_trans_end[1];
  gz = xyz_trans_end[2];
  gz2 = gz*gz;
  // std::cout << "Xw_start " << Xw_start << std::endl;
  // std::cout << "Xw_end " << Xw_end << std::endl;
  // std::cout << "xyz_trans_start " << xyz_trans_start << std::endl;
  // std::cout << "xyz_trans_end " << xyz_trans_end << std::endl;
  // std::cout << "obs " << obs << std::endl;
  Vector2d K_start = cam_project(xyz_trans_start);
  Vector2d K_end = cam_project(xyz_trans_end);
  //apo ta panw tha parw ta k 
  //mallon tha prepei na kanw jexwrista tis 2 grammes tou jacobian

  //obs has the start and and point of the lines in previous frame (so prevP and prevQ)

  double prevP0 = obs(0);
  double prevP1 = obs(1);
  double prevQ0 = obs(2);
  double prevQ1 = obs(3);
  Vector2d flow_start, flow_end;
  //std::cout << "estimate of vi si " << vi->estimate() << std::endl;
  
  flow_start  << vi->estimate()(0), vi->estimate()(1);
  flow_end << vi->estimate()(2), vi->estimate()(3);
  double fp0 = flow_start.x();
  double fp1 = flow_start.y();
  double fq0 = flow_end.x();
  double fq1 = flow_end.y();

  //The following are the observations in the current frame
  double P0 = prevP0 + flow_start[0];
  double P1 = prevP1 + flow_start[1];
  double Q0 = prevQ0 + flow_end[0];
  double Q1 = prevQ1 + flow_end[1];

  Eigen::Matrix<double, 3, 1> P, Q;
  P << P0, P1, 1;
  Q << Q0, Q1, 1;
  Eigen::Matrix<double, 3, 1> tmp_line = P.cross(Q) / (P.cross(Q)).norm();
  Eigen::Matrix<double, 1, 2> line;
  line << tmp_line(0), tmp_line(1);
  double Kp1 = K_start.x();
  double Kp2 = K_start.y();
  double Kq1 = K_end.x();
  double Kq2 = K_end.y();
  // if (P1 - Q1 <  1e-10)
  // {
  //   P1 = P1 + 1e-10;
  // }

  _jacobianOplusXi(0, 0) = 1.0/pow(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0)+pow((fp0+prevP0)*(fq1+prevQ1)-(fp1+prevP1)*(fq0+prevQ0),2.0),3.0/2.0)*(fp1-fq1+prevP1-prevQ1)*(-Kp1*fp0-Kp2*fp1+Kp1*fq0+Kp2*fq1-Kp1*prevP0+Kp1*prevQ0-Kp2*prevP1+Kp2*prevQ1+fp0*fq0+fp1*fq1+fp0*prevQ0+fp1*prevQ1+fq0*prevP0-fq0*prevQ0*2.0+fq1*prevP1-fq1*prevQ1*2.0+prevP0*prevQ0+prevP1*prevQ1-fq0*fq0-fq1*fq1-prevQ0*prevQ0-prevQ1*prevQ1-Kp1*fp0*(fq1*fq1)-Kp2*fp1*(fq0*fq0)-Kp1*fp0*(prevQ1*prevQ1)-Kp2*fp1*(prevQ0*prevQ0)-Kp1*(fq1*fq1)*prevP0-Kp2*(fq0*fq0)*prevP1-Kp1*prevP0*(prevQ1*prevQ1)-Kp2*prevP1*(prevQ0*prevQ0)+Kp1*fp1*fq0*fq1+Kp2*fp0*fq0*fq1-Kp1*fp0*fq1*prevQ1*2.0+Kp1*fp1*fq0*prevQ1+Kp1*fp1*fq1*prevQ0+Kp2*fp0*fq0*prevQ1+Kp2*fp0*fq1*prevQ0-Kp2*fp1*fq0*prevQ0*2.0+Kp1*fq0*fq1*prevP1+Kp2*fq0*fq1*prevP0+Kp1*fp1*prevQ0*prevQ1+Kp2*fp0*prevQ0*prevQ1+Kp1*fq0*prevP1*prevQ1-Kp1*fq1*prevP0*prevQ1*2.0+Kp1*fq1*prevP1*prevQ0+Kp2*fq0*prevP0*prevQ1-Kp2*fq0*prevP1*prevQ0*2.0+Kp2*fq1*prevP0*prevQ0+Kp1*prevP1*prevQ0*prevQ1+Kp2*prevP0*prevQ0*prevQ1);
  _jacobianOplusXi(0, 1) = -1.0/pow(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0)+pow((fp0+prevP0)*(fq1+prevQ1)-(fp1+prevP1)*(fq0+prevQ0),2.0),3.0/2.0)*(fp0-fq0+prevP0-prevQ0)*(-Kp1*fp0-Kp2*fp1+Kp1*fq0+Kp2*fq1-Kp1*prevP0+Kp1*prevQ0-Kp2*prevP1+Kp2*prevQ1+fp0*fq0+fp1*fq1+fp0*prevQ0+fp1*prevQ1+fq0*prevP0-fq0*prevQ0*2.0+fq1*prevP1-fq1*prevQ1*2.0+prevP0*prevQ0+prevP1*prevQ1-fq0*fq0-fq1*fq1-prevQ0*prevQ0-prevQ1*prevQ1-Kp1*fp0*(fq1*fq1)-Kp2*fp1*(fq0*fq0)-Kp1*fp0*(prevQ1*prevQ1)-Kp2*fp1*(prevQ0*prevQ0)-Kp1*(fq1*fq1)*prevP0-Kp2*(fq0*fq0)*prevP1-Kp1*prevP0*(prevQ1*prevQ1)-Kp2*prevP1*(prevQ0*prevQ0)+Kp1*fp1*fq0*fq1+Kp2*fp0*fq0*fq1-Kp1*fp0*fq1*prevQ1*2.0+Kp1*fp1*fq0*prevQ1+Kp1*fp1*fq1*prevQ0+Kp2*fp0*fq0*prevQ1+Kp2*fp0*fq1*prevQ0-Kp2*fp1*fq0*prevQ0*2.0+Kp1*fq0*fq1*prevP1+Kp2*fq0*fq1*prevP0+Kp1*fp1*prevQ0*prevQ1+Kp2*fp0*prevQ0*prevQ1+Kp1*fq0*prevP1*prevQ1-Kp1*fq1*prevP0*prevQ1*2.0+Kp1*fq1*prevP1*prevQ0+Kp2*fq0*prevP0*prevQ1-Kp2*fq0*prevP1*prevQ0*2.0+Kp2*fq1*prevP0*prevQ0+Kp1*prevP1*prevQ0*prevQ1+Kp2*prevP0*prevQ0*prevQ1);
  _jacobianOplusXi(0, 2) = 1.0/pow(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0)+pow((fp0+prevP0)*(fq1+prevQ1)-(fp1+prevP1)*(fq0+prevQ0),2.0),3.0/2.0)*(fp1-fq1+prevP1-prevQ1)*(Kp1*fp0+Kp2*fp1-Kp1*fq0-Kp2*fq1+Kp1*prevP0-Kp1*prevQ0+Kp2*prevP1-Kp2*prevQ1+fp0*fq0+fp1*fq1-fp0*prevP0*2.0+fp0*prevQ0-fp1*prevP1*2.0+fp1*prevQ1+fq0*prevP0+fq1*prevP1+prevP0*prevQ0+prevP1*prevQ1-fp0*fp0-fp1*fp1-prevP0*prevP0-prevP1*prevP1-Kp1*(fp1*fp1)*fq0-Kp2*(fp0*fp0)*fq1-Kp1*(fp1*fp1)*prevQ0-Kp2*(fp0*fp0)*prevQ1-Kp1*fq0*(prevP1*prevP1)-Kp2*fq1*(prevP0*prevP0)-Kp1*(prevP1*prevP1)*prevQ0-Kp2*(prevP0*prevP0)*prevQ1+Kp1*fp0*fp1*fq1+Kp2*fp0*fp1*fq0+Kp1*fp0*fp1*prevQ1+Kp2*fp0*fp1*prevQ0+Kp1*fp0*fq1*prevP1-Kp1*fp1*fq0*prevP1*2.0+Kp1*fp1*fq1*prevP0+Kp2*fp0*fq0*prevP1-Kp2*fp0*fq1*prevP0*2.0+Kp2*fp1*fq0*prevP0+Kp1*fp0*prevP1*prevQ1+Kp1*fp1*prevP0*prevQ1-Kp1*fp1*prevP1*prevQ0*2.0-Kp2*fp0*prevP0*prevQ1*2.0+Kp2*fp0*prevP1*prevQ0+Kp2*fp1*prevP0*prevQ0+Kp1*fq1*prevP0*prevP1+Kp2*fq0*prevP0*prevP1+Kp1*prevP0*prevP1*prevQ1+Kp2*prevP0*prevP1*prevQ0);
  _jacobianOplusXi(0, 3) = -1.0/pow(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0)+pow((fp0+prevP0)*(fq1+prevQ1)-(fp1+prevP1)*(fq0+prevQ0),2.0),3.0/2.0)*(fp0-fq0+prevP0-prevQ0)*(Kp1*fp0+Kp2*fp1-Kp1*fq0-Kp2*fq1+Kp1*prevP0-Kp1*prevQ0+Kp2*prevP1-Kp2*prevQ1+fp0*fq0+fp1*fq1-fp0*prevP0*2.0+fp0*prevQ0-fp1*prevP1*2.0+fp1*prevQ1+fq0*prevP0+fq1*prevP1+prevP0*prevQ0+prevP1*prevQ1-fp0*fp0-fp1*fp1-prevP0*prevP0-prevP1*prevP1-Kp1*(fp1*fp1)*fq0-Kp2*(fp0*fp0)*fq1-Kp1*(fp1*fp1)*prevQ0-Kp2*(fp0*fp0)*prevQ1-Kp1*fq0*(prevP1*prevP1)-Kp2*fq1*(prevP0*prevP0)-Kp1*(prevP1*prevP1)*prevQ0-Kp2*(prevP0*prevP0)*prevQ1+Kp1*fp0*fp1*fq1+Kp2*fp0*fp1*fq0+Kp1*fp0*fp1*prevQ1+Kp2*fp0*fp1*prevQ0+Kp1*fp0*fq1*prevP1-Kp1*fp1*fq0*prevP1*2.0+Kp1*fp1*fq1*prevP0+Kp2*fp0*fq0*prevP1-Kp2*fp0*fq1*prevP0*2.0+Kp2*fp1*fq0*prevP0+Kp1*fp0*prevP1*prevQ1+Kp1*fp1*prevP0*prevQ1-Kp1*fp1*prevP1*prevQ0*2.0-Kp2*fp0*prevP0*prevQ1*2.0+Kp2*fp0*prevP1*prevQ0+Kp2*fp1*prevP0*prevQ0+Kp1*fq1*prevP0*prevP1+Kp2*fq0*prevP0*prevP1+Kp1*prevP0*prevP1*prevQ1+Kp2*prevP0*prevP1*prevQ0);
  _jacobianOplusXi(1, 0) = 1.0/pow(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0)+pow((fp0+prevP0)*(fq1+prevQ1)-(fp1+prevP1)*(fq0+prevQ0),2.0),3.0/2.0)*(fp1-fq1+prevP1-prevQ1)*(-Kq1*fp0-Kq2*fp1+Kq1*fq0+Kq2*fq1-Kq1*prevP0+Kq1*prevQ0-Kq2*prevP1+Kq2*prevQ1+fp0*fq0+fp1*fq1+fp0*prevQ0+fp1*prevQ1+fq0*prevP0-fq0*prevQ0*2.0+fq1*prevP1-fq1*prevQ1*2.0+prevP0*prevQ0+prevP1*prevQ1-fq0*fq0-fq1*fq1-prevQ0*prevQ0-prevQ1*prevQ1-Kq1*fp0*(fq1*fq1)-Kq2*fp1*(fq0*fq0)-Kq1*fp0*(prevQ1*prevQ1)-Kq2*fp1*(prevQ0*prevQ0)-Kq1*(fq1*fq1)*prevP0-Kq2*(fq0*fq0)*prevP1-Kq1*prevP0*(prevQ1*prevQ1)-Kq2*prevP1*(prevQ0*prevQ0)+Kq1*fp1*fq0*fq1+Kq2*fp0*fq0*fq1-Kq1*fp0*fq1*prevQ1*2.0+Kq1*fp1*fq0*prevQ1+Kq1*fp1*fq1*prevQ0+Kq2*fp0*fq0*prevQ1+Kq2*fp0*fq1*prevQ0-Kq2*fp1*fq0*prevQ0*2.0+Kq1*fq0*fq1*prevP1+Kq2*fq0*fq1*prevP0+Kq1*fp1*prevQ0*prevQ1+Kq2*fp0*prevQ0*prevQ1+Kq1*fq0*prevP1*prevQ1-Kq1*fq1*prevP0*prevQ1*2.0+Kq1*fq1*prevP1*prevQ0+Kq2*fq0*prevP0*prevQ1-Kq2*fq0*prevP1*prevQ0*2.0+Kq2*fq1*prevP0*prevQ0+Kq1*prevP1*prevQ0*prevQ1+Kq2*prevP0*prevQ0*prevQ1);
  _jacobianOplusXi(1, 1) = -1.0/pow(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0)+pow((fp0+prevP0)*(fq1+prevQ1)-(fp1+prevP1)*(fq0+prevQ0),2.0),3.0/2.0)*(fp0-fq0+prevP0-prevQ0)*(-Kq1*fp0-Kq2*fp1+Kq1*fq0+Kq2*fq1-Kq1*prevP0+Kq1*prevQ0-Kq2*prevP1+Kq2*prevQ1+fp0*fq0+fp1*fq1+fp0*prevQ0+fp1*prevQ1+fq0*prevP0-fq0*prevQ0*2.0+fq1*prevP1-fq1*prevQ1*2.0+prevP0*prevQ0+prevP1*prevQ1-fq0*fq0-fq1*fq1-prevQ0*prevQ0-prevQ1*prevQ1-Kq1*fp0*(fq1*fq1)-Kq2*fp1*(fq0*fq0)-Kq1*fp0*(prevQ1*prevQ1)-Kq2*fp1*(prevQ0*prevQ0)-Kq1*(fq1*fq1)*prevP0-Kq2*(fq0*fq0)*prevP1-Kq1*prevP0*(prevQ1*prevQ1)-Kq2*prevP1*(prevQ0*prevQ0)+Kq1*fp1*fq0*fq1+Kq2*fp0*fq0*fq1-Kq1*fp0*fq1*prevQ1*2.0+Kq1*fp1*fq0*prevQ1+Kq1*fp1*fq1*prevQ0+Kq2*fp0*fq0*prevQ1+Kq2*fp0*fq1*prevQ0-Kq2*fp1*fq0*prevQ0*2.0+Kq1*fq0*fq1*prevP1+Kq2*fq0*fq1*prevP0+Kq1*fp1*prevQ0*prevQ1+Kq2*fp0*prevQ0*prevQ1+Kq1*fq0*prevP1*prevQ1-Kq1*fq1*prevP0*prevQ1*2.0+Kq1*fq1*prevP1*prevQ0+Kq2*fq0*prevP0*prevQ1-Kq2*fq0*prevP1*prevQ0*2.0+Kq2*fq1*prevP0*prevQ0+Kq1*prevP1*prevQ0*prevQ1+Kq2*prevP0*prevQ0*prevQ1);
  _jacobianOplusXi(1, 2) = 1.0/pow(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0)+pow((fp0+prevP0)*(fq1+prevQ1)-(fp1+prevP1)*(fq0+prevQ0),2.0),3.0/2.0)*(fp1-fq1+prevP1-prevQ1)*(Kq1*fp0+Kq2*fp1-Kq1*fq0-Kq2*fq1+Kq1*prevP0-Kq1*prevQ0+Kq2*prevP1-Kq2*prevQ1+fp0*fq0+fp1*fq1-fp0*prevP0*2.0+fp0*prevQ0-fp1*prevP1*2.0+fp1*prevQ1+fq0*prevP0+fq1*prevP1+prevP0*prevQ0+prevP1*prevQ1-fp0*fp0-fp1*fp1-prevP0*prevP0-prevP1*prevP1-Kq1*(fp1*fp1)*fq0-Kq2*(fp0*fp0)*fq1-Kq1*(fp1*fp1)*prevQ0-Kq2*(fp0*fp0)*prevQ1-Kq1*fq0*(prevP1*prevP1)-Kq2*fq1*(prevP0*prevP0)-Kq1*(prevP1*prevP1)*prevQ0-Kq2*(prevP0*prevP0)*prevQ1+Kq1*fp0*fp1*fq1+Kq2*fp0*fp1*fq0+Kq1*fp0*fp1*prevQ1+Kq2*fp0*fp1*prevQ0+Kq1*fp0*fq1*prevP1-Kq1*fp1*fq0*prevP1*2.0+Kq1*fp1*fq1*prevP0+Kq2*fp0*fq0*prevP1-Kq2*fp0*fq1*prevP0*2.0+Kq2*fp1*fq0*prevP0+Kq1*fp0*prevP1*prevQ1+Kq1*fp1*prevP0*prevQ1-Kq1*fp1*prevP1*prevQ0*2.0-Kq2*fp0*prevP0*prevQ1*2.0+Kq2*fp0*prevP1*prevQ0+Kq2*fp1*prevP0*prevQ0+Kq1*fq1*prevP0*prevP1+Kq2*fq0*prevP0*prevP1+Kq1*prevP0*prevP1*prevQ1+Kq2*prevP0*prevP1*prevQ0);
  _jacobianOplusXi(1, 3) = -1.0/pow(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0)+pow((fp0+prevP0)*(fq1+prevQ1)-(fp1+prevP1)*(fq0+prevQ0),2.0),3.0/2.0)*(fp0-fq0+prevP0-prevQ0)*(Kq1*fp0+Kq2*fp1-Kq1*fq0-Kq2*fq1+Kq1*prevP0-Kq1*prevQ0+Kq2*prevP1-Kq2*prevQ1+fp0*fq0+fp1*fq1-fp0*prevP0*2.0+fp0*prevQ0-fp1*prevP1*2.0+fp1*prevQ1+fq0*prevP0+fq1*prevP1+prevP0*prevQ0+prevP1*prevQ1-fp0*fp0-fp1*fp1-prevP0*prevP0-prevP1*prevP1-Kq1*(fp1*fp1)*fq0-Kq2*(fp0*fp0)*fq1-Kq1*(fp1*fp1)*prevQ0-Kq2*(fp0*fp0)*prevQ1-Kq1*fq0*(prevP1*prevP1)-Kq2*fq1*(prevP0*prevP0)-Kq1*(prevP1*prevP1)*prevQ0-Kq2*(prevP0*prevP0)*prevQ1+Kq1*fp0*fp1*fq1+Kq2*fp0*fp1*fq0+Kq1*fp0*fp1*prevQ1+Kq2*fp0*fp1*prevQ0+Kq1*fp0*fq1*prevP1-Kq1*fp1*fq0*prevP1*2.0+Kq1*fp1*fq1*prevP0+Kq2*fp0*fq0*prevP1-Kq2*fp0*fq1*prevP0*2.0+Kq2*fp1*fq0*prevP0+Kq1*fp0*prevP1*prevQ1+Kq1*fp1*prevP0*prevQ1-Kq1*fp1*prevP1*prevQ0*2.0-Kq2*fp0*prevP0*prevQ1*2.0+Kq2*fp0*prevP1*prevQ0+Kq2*fp1*prevP0*prevQ0+Kq1*fq1*prevP0*prevP1+Kq2*fq0*prevP0*prevP1+Kq1*prevP0*prevP1*prevQ1+Kq2*prevP0*prevP1*prevQ0);

  //check if any of the values of the jacobian are nan
  // if (std::isnan(_jacobianOplusXi(0,0)) || std::isnan(_jacobianOplusXi(0,1)) || std::isnan(_jacobianOplusXi(0,2)) || std::isnan(_jacobianOplusXi(0,3)) ||
  //     std::isnan(_jacobianOplusXi(1,0)) || std::isnan(_jacobianOplusXi(1,1)) || std::isnan(_jacobianOplusXi(1,2)) || std::isnan(_jacobianOplusXi(1,3)))
  //   {
  //   std::cout << "Jacobian is nan" << std::endl;
  //   std::cout << "P0: " << P0 << " P1: " << P1 << " Q0: " << Q0 << " Q1: " << Q1 << std::endl;
  //   std::cout << "Kq1: " << Kq1 << " Kq2: " << Kq2 << std::endl;
  //   std::cout << "Kp1: " << Kp1 << " Kp2: " << Kp2 << std::endl;
  //   std::cout << "_jacobianOplusXi " << _jacobianOplusXi << std::endl;
  // }

  Eigen::Matrix<double, 2, 6> der_proj_start_xi, der_proj_end_xi;
  der_proj_start_xi << -fx*mx*my/mz2, fx*(1+(std::pow(mx, 2) / mz2)), -fx*(my/mz), fx/mz, 0, -fx*mx/mz2, 
    -fy*(1+(std::pow(my, 2)/mz2)), fy*mx*my/mz2, fy*mx/mz, 0, fy/mz, -fy*my/mz2;
  der_proj_end_xi << -fx*gx*gy/gz2, fx*(1+(std::pow(gx, 2) / gz2)), -fx*(gy/gz), fx/gz, 0, -fx*gx/gz2, 
    -fy*(1+(std::pow(my, 2)/mz2)), fy*gx*gy/gz2, fy*gx/gz, 0, fy/gz, -fy*gy/gz2;
  

  Eigen::Matrix<double, 2, 6> der_e_xi;
  Eigen::Matrix<double, 1, 6> der_e_xi_start, der_e_xi_end;
  der_e_xi_start = line * der_proj_start_xi;
  der_e_xi_end = line * der_proj_end_xi;
  der_e_xi.block<1, 6>(0, 0) = der_e_xi_start;
  der_e_xi.block<1, 6>(1, 0) = der_e_xi_end;

  //check if any of der_e_xi is nan
  // if (std::isnan(der_e_xi(0,0)) || std::isnan(der_e_xi(0,1)) || std::isnan(der_e_xi(0,2)) || std::isnan(der_e_xi(0,3)) || std::isnan(der_e_xi(0,4)) || std::isnan(der_e_xi(0,5)) ||
  //     std::isnan(der_e_xi(1,0)) || std::isnan(der_e_xi(1,1)) || std::isnan(der_e_xi(1,2)) || std::isnan(der_e_xi(1,3)) || std::isnan(der_e_xi(1,4)) || std::isnan(der_e_xi(1,5)))
  // {
  //   std::cout << "der_e_xi is nan" << std::endl;
  //   std::cout << "P0: " << P0 << " P1: " << P1 << " Q0: " << Q0 << " Q1: " << Q1 << std::endl;
  //   std::cout << "Kq1: " << Kq1 << " Kq2: " << Kq2 << std::endl;
  //   std::cout << "der_e_xi: " << std::endl << der_e_xi << std::endl;
  // }

  _jacobianOplusXj = der_e_xi;
  // //Here we are going to calculate the jacobian of the error of the start point wrt the pose
  // Eigen::Matrix<double,2,6> ProjJac_s;
  

  // double deriv_wrt_proj11 =   pow(fp1-fq1+prevP1-prevQ1,2.0)/(fp0*fq0*-2.0-fp1*fq1*2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP0*prevQ0*2.0-prevP1*prevQ1*2.0+fp0*fp0+fp1*fp1+fq0*fq0+fq1*fq1+prevP0*prevP0+prevP1*prevP1+prevQ0*prevQ0+prevQ1*prevQ1);
  // double deriv_wrt_proj21 = -1.0/sqrt(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0))*(fp0-fq0+prevP0-prevQ0)*(fp1-fq1+prevP1-prevQ1)*1.0/sqrt(fp0*fq0*-2.0-fp1*fq1*2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP0*prevQ0*2.0-prevP1*prevQ1*2.0+fp0*fp0+fp1*fp1+fq0*fq0+fq1*fq1+prevP0*prevP0+prevP1*prevP1+prevQ0*prevQ0+prevQ1*prevQ1);
  // double deriv_wrt_proj12 = -1.0/sqrt(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0))*(fp0-fq0+prevP0-prevQ0)*(fp1-fq1+prevP1-prevQ1)*1.0/sqrt(fp0*fq0*-2.0-fp1*fq1*2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP0*prevQ0*2.0-prevP1*prevQ1*2.0+fp0*fp0+fp1*fp1+fq0*fq0+fq1*fq1+prevP0*prevP0+prevP1*prevP1+prevQ0*prevQ0+prevQ1*prevQ1);
  // double deriv_wrt_proj22 =   pow(fp0-fq0+prevP0-prevQ0,2.0)/(fp0*fq0*-2.0-fp1*fq1*2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP0*prevQ0*2.0-prevP1*prevQ1*2.0+fp0*fp0+fp1*fp1+fq0*fq0+fq1*fq1+prevP0*prevP0+prevP1*prevP1+prevQ0*prevQ0+prevQ1*prevQ1);

  // Eigen::Matrix<double, 2, 2> deriv_wrt_proj;
  // deriv_wrt_proj << deriv_wrt_proj11, deriv_wrt_proj12,
  //                   deriv_wrt_proj21, deriv_wrt_proj22;
  
  // ProjJac_s << -fx*mx*my/mz2, fx*(1+(std::pow(mx, 2) / mz2)), -fx*(my/mz), fx/mz, 0, -fx*mx/mz2, 
  //   -fy*(1+(std::pow(my, 2)/mz2)), fy*mx*my/mz2, fy*mx/mz, 0, fy/mz, -fy*my/mz2;

  // Eigen::Matrix<double, 2, 6> deriv_wrt_s = deriv_wrt_proj * ProjJac_s;

  // // double x_start = xyz_trans_start[0];
  // // double y_start = xyz_trans_start[1];
  // // double z_start = xyz_trans_start[2];
  // // double z_2_start =  z_start*z_start;

  // // double x_end = xyz_trans_end[0];
  // // double y_end = xyz_trans_end[1];
  // // double z_end = xyz_trans_end[2];
  // // double z_2_end =  z_end*z_end;

  // //Jabobian second row (error term for the end point)
  // // K0 = K_end.x();
  // // K1 = K_end.y();



  // //Now for the jacobian of the error for the end point wrt pose
  // deriv_wrt_proj11 = pow(fp1-fq1+prevP1-prevQ1,2.0)/(fp0*fq0*-2.0-fp1*fq1*2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP0*prevQ0*2.0-prevP1*prevQ1*2.0+fp0*fp0+fp1*fp1+fq0*fq0+fq1*fq1+prevP0*prevP0+prevP1*prevP1+prevQ0*prevQ0+prevQ1*prevQ1);
  // deriv_wrt_proj21 = -1.0/sqrt(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0))*(fp0-fq0+prevP0-prevQ0)*(fp1-fq1+prevP1-prevQ1)*1.0/sqrt(fp0*fq0*-2.0-fp1*fq1*2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP0*prevQ0*2.0-prevP1*prevQ1*2.0+fp0*fp0+fp1*fp1+fq0*fq0+fq1*fq1+prevP0*prevP0+prevP1*prevP1+prevQ0*prevQ0+prevQ1*prevQ1);
  // deriv_wrt_proj12 = -1.0/sqrt(pow(fp0-fq0+prevP0-prevQ0,2.0)+pow(fp1-fq1+prevP1-prevQ1,2.0))*(fp0-fq0+prevP0-prevQ0)*(fp1-fq1+prevP1-prevQ1)*1.0/sqrt(fp0*fq0*-2.0-fp1*fq1*2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP0*prevQ0*2.0-prevP1*prevQ1*2.0+fp0*fp0+fp1*fp1+fq0*fq0+fq1*fq1+prevP0*prevP0+prevP1*prevP1+prevQ0*prevQ0+prevQ1*prevQ1);
  // deriv_wrt_proj22 = pow(fp0-fq0+prevP0-prevQ0,2.0)/(fp0*fq0*-2.0-fp1*fq1*2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP0*prevQ0*2.0-prevP1*prevQ1*2.0+fp0*fp0+fp1*fp1+fq0*fq0+fq1*fq1+prevP0*prevP0+prevP1*prevP1+prevQ0*prevQ0+prevQ1*prevQ1);

  // deriv_wrt_proj << deriv_wrt_proj11, deriv_wrt_proj12,
  //                   deriv_wrt_proj21, deriv_wrt_proj22;

  // Eigen::Matrix<double, 2, 6> ProjJac_e;
  // ProjJac_s << -fx*gx*gy/gz2, fx*(1+(std::pow(gx, 2) / gz2)), -fx*(gy/gz), fx/gz, 0, -fx*gx/gz2, 
  //   -fy*(1+(std::pow(my, 2)/mz2)), fy*gx*gy/gz2, fy*gx/gz, 0, fy/gz, -fy*gy/gz2;

  // Eigen::Matrix<double, 2, 6> deriv_wrt_e = deriv_wrt_proj * ProjJac_e;
  // Eigen::Matrix<double, 4, 6> Jacobian;
  // Jacobian << deriv_wrt_s,
  //               deriv_wrt_e;

  // _jacobianOplusXj = Jacobian;


}


//   EdgeSE3ProjectFlow2_Line::EdgeSE3ProjectFlow2_Line(): BaseMultiEdge<4, Vector4d>()
//   {
//     resize(3);
//     // _jacobianOplus[0].resize(2,2);
//     // _jacobianOplus[1].resize(2,2);
//     // _jacobianOplus[2].resize(2,6);
//     // _measurement = Vector4d::Zero();
//     //_error = Vector2d::Zero();
//   };


// bool EdgeSE3ProjectFlow2_Line::read(std::istream& is){
//   for (int i=0; i<4; i++){
//     is >> _measurement[i];
//   }
//   for (int i=0; i<4; i++)
//     for (int j=i; j<4; j++) {
//       is >> information()(i,j);
//       if (i!=j)
//         information()(j,i)=information()(i,j);
//     }
//   return true;
// }

// bool EdgeSE3ProjectFlow2_Line::write(std::ostream& os) const {

//   for (int i=0; i<4; i++){
//     os << measurement()[i] << " ";
//   }

//   for (int i=0; i<4; i++)
//     for (int j=i; j<4; j++){
//       os << " " <<  information()(i,j);
//     }
//   return os.good();
// }

// Vector2d EdgeSE3ProjectFlow2_Line::cam_project(const Vector3d & trans_xyz) const{
//   Vector2d proj = project2d(trans_xyz);
//   Vector2d res;
//   res[0] = proj[0]*fx + cx;
//   res[1] = proj[1]*fy + cy;
//   return res;
// }


// void EdgeSE3ProjectFlow2_Line::linearizeOplus() {
//   std::cout << "Test line oplus1" << std::endl;
//   VertexSE3Expmap * vk = static_cast<VertexSE3Expmap *>(_vertices[2]);
//   SE3Quat T(vk->estimate());
//   VertexSBAFlow* vi = static_cast<VertexSBAFlow*>(_vertices[0]);
//   VertexSBAFlow* vj = static_cast<VertexSBAFlow*>(_vertices[1]);
//   Vector4d obs(_measurement);
//   Vector3d Xw_start, Xw_end;

//   std::cout << "Test line oplus2" << std::endl;
//   Xw_start << (obs(0)-cx)*depth_start/fx, (obs(1)-cy)*depth_start/fy, depth_start;
//   Xw_end << (obs(2)-cx)*depth_end/fx, (obs(3)-cy)*depth_end/fy, depth_end;
//   Xw_start = Twl.block(0,0,3,3)*Xw_start + Twl.col(3).head(3);
//   Xw_end = Twl.block(0,0,3,3)*Xw_end + Twl.col(3).head(3);
//   Vector3d xyz_trans_start = T.map(Xw_start);
//   Vector3d xyz_trans_end = T.map(Xw_end);
//   double mx, my, mz, mz2, gx, gy, gz, gz2;
//   mx = xyz_trans_start[0];
//   my = xyz_trans_start[1];
//   mz = xyz_trans_start[2];
//   mz2 = mz*mz;
//   gx = xyz_trans_end[0];
//   gy = xyz_trans_end[1];
//   gz = xyz_trans_end[2];
//   gz2 = gz*gz;

//   Vector2d K_start = cam_project(xyz_trans_start);
//   Vector2d K_end = cam_project(xyz_trans_end);
//   //apo ta panw tha parw ta k 
//   //mallon tha prepei na kanw jexwrista tis 2 grammes tou jacobian

//   //obs has the start and and point of the lines in previous frame (so prevP and prevQ)

//   double prevP0 = obs(0);
//   double prevP1 = obs(1);
//   double prevQ0 = obs(2);
//   double prevQ1 = obs(3);
//   Vector2d flow_start = vi->estimate();
//   Vector2d flow_end = vj->estimate();
//   double fp0 = flow_start.x();
//   double fp1 = flow_start.y();
//   double fq0 = flow_end.x();
//   double fq1 = flow_end.y();

//   //The following are the observations in the current frame
//   double P0 = prevP0 + flow_start[0];
//   double P1 = prevP1 + flow_start[1];
//   double Q0 = prevQ0 + flow_end[0];
//   double Q1 = prevQ1 + flow_end[1];

//   double K0 = K_start.x();
//   double K1 = K_start.y();

//   //Here we are going to calculate the jacobian of the error of the start point wrt the pose
//   Eigen::Matrix<double,2,6> ProjJac_s;
  

//   double deriv_wrt_proj0 = 1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((fp0*fq0*-2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-prevP0*prevQ0*2.0+fp0*fp0+fq0*fq0+prevP0*prevP0+prevQ0*prevQ0-1.0)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0+(fp0-fq0+prevP0-prevQ0)*(fp1-fq1+prevP1-prevQ1)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0)*(-1.0/2.0);
//   double deriv_wrt_proj1 = 1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((fp1*fq1*-2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP1*prevQ1*2.0+fp1*fp1+fq1*fq1+prevP1*prevP1+prevQ1*prevQ1-1.0)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0+(fp0-fq0+prevP0-prevQ0)*(fp1-fq1+prevP1-prevQ1)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0)*(-1.0/2.0);

//   Eigen::Matrix<double, 1, 2> deriv_wrt_proj;
//   deriv_wrt_proj << deriv_wrt_proj0, deriv_wrt_proj1;
  
//   ProjJac_s << fx*(1+(std::pow(mx, 2) / mz2)), -fx*(my/mz), fx/mz, 0, -fx*mx/mz2, -fx*mx*my/mz2,
//     -fy*(1+(std::pow(my, 2)/mz2)), fy*mx*my/mz2, fy*mx/mz, 0, fy/mz, -fy*my/mz2;

//   Eigen::Matrix<double, 1, 6> deriv_wrt_s = deriv_wrt_proj * ProjJac_s;

//   double x_start = xyz_trans_start[0];
//   double y_start = xyz_trans_start[1];
//   double z_start = xyz_trans_start[2];
//   double z_2_start =  z_start*z_start;

//   double x_end = xyz_trans_end[0];
//   double y_end = xyz_trans_end[1];
//   double z_end = xyz_trans_end[2];
//   double z_2_end =  z_end*z_end;

//   //Jacobian with respect to fp0
//   double _jac_start1 = 1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((K0*fp0*2.0+K1*fp1-K0*fq0*2.0-K1*fq1+K0*prevP0*2.0-K0*prevQ0*2.0+K1*prevP1-K1*prevQ1+fp0*fq0*4.0+fp1*fq1-fp0*prevP0*6.0+fp0*prevQ0*4.0-fp1*prevP1*2.0+fp1*prevQ1+fq0*prevP0*4.0-fq0*prevQ0*2.0+fq1*prevP1+prevP0*prevQ0*4.0+prevP1*prevQ1-(fp0*fp0)*3.0-fp1*fp1-fq0*fq0-(prevP0*prevP0)*3.0-prevP1*prevP1-prevQ0*prevQ0+1.0)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0+(fp1-fq1+prevP1-prevQ1)*(K0-fp0*2.0+fq0-prevP0*2.0+prevQ0)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0)*(-1.0/2.0);


//   //Jacobian with respect to fp1
//   double _jac_start2 = 1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((K0*fp0+K1*fp1*2.0-K0*fq0-K1*fq1*2.0+K0*prevP0-K0*prevQ0+K1*prevP1*2.0-K1*prevQ1*2.0+fp0*fq0+fp1*fq1*4.0-fp0*prevP0*2.0+fp0*prevQ0-fp1*prevP1*6.0+fp1*prevQ1*4.0+fq0*prevP0+fq1*prevP1*4.0-fq1*prevQ1*2.0+prevP0*prevQ0+prevP1*prevQ1*4.0-fp0*fp0-(fp1*fp1)*3.0-fq1*fq1-prevP0*prevP0-(prevP1*prevP1)*3.0-prevQ1*prevQ1+1.0)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0+(fp0-fq0+prevP0-prevQ0)*(K1-fp1*2.0+fq1-prevP1*2.0+prevQ1)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0)*(-1.0/2.0);

//   //Jacobian with respect to fq0
//   double _jac_end1 = (1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((K0*fp0*2.0+K1*fp1-K0*fq0*2.0-K1*fq1+K0*prevP0*2.0-K0*prevQ0*2.0+K1*prevP1-K1*prevQ1+fp0*fq0*2.0+fp1*fq1-fp0*prevP0*4.0+fp0*prevQ0*2.0-fp1*prevP1*2.0+fp1*prevQ1+fq0*prevP0*2.0+fq1*prevP1+prevP0*prevQ0*2.0+prevP1*prevQ1-(fp0*fp0)*2.0-fp1*fp1-(prevP0*prevP0)*2.0-prevP1*prevP1)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0-(-K0+fp0+prevP0)*(fp1-fq1+prevP1-prevQ1)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0))/2.0;

//   //Jacobian with respect to fq1
//   double _jac_end2 = (1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((K0*fp0+K1*fp1*2.0-K0*fq0-K1*fq1*2.0+K0*prevP0-K0*prevQ0+K1*prevP1*2.0-K1*prevQ1*2.0+fp0*fq0+fp1*fq1*2.0-fp0*prevP0*2.0+fp0*prevQ0-fp1*prevP1*4.0+fp1*prevQ1*2.0+fq0*prevP0+fq1*prevP1*2.0+prevP0*prevQ0+prevP1*prevQ1*2.0-fp0*fp0-(fp1*fp1)*2.0-prevP0*prevP0-(prevP1*prevP1)*2.0)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0-(-K1+fp1+prevP1)*(fp0-fq0+prevP0-prevQ0)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0))/2.0;

//   //Jabobian second row (error term for the end point)
//   K0 = K_end.x();
//   K1 = K_end.y();



//   //Now for the jacobian of the error for the end point wrt pose
//   deriv_wrt_proj0 = 1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((fp0*fq0*-2.0+fp0*prevP0*2.0-fp0*prevQ0*2.0-fq0*prevP0*2.0+fq0*prevQ0*2.0-prevP0*prevQ0*2.0+fp0*fp0+fq0*fq0+prevP0*prevP0+prevQ0*prevQ0-1.0)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0+(fp0-fq0+prevP0-prevQ0)*(fp1-fq1+prevP1-prevQ1)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0)*(-1.0/2.0);
//   deriv_wrt_proj1 = 1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((fp1*fq1*-2.0+fp1*prevP1*2.0-fp1*prevQ1*2.0-fq1*prevP1*2.0+fq1*prevQ1*2.0-prevP1*prevQ1*2.0+fp1*fp1+fq1*fq1+prevP1*prevP1+prevQ1*prevQ1-1.0)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0+(fp0-fq0+prevP0-prevQ0)*(fp1-fq1+prevP1-prevQ1)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0)*(-1.0/2.0);

//   deriv_wrt_proj << deriv_wrt_proj0, deriv_wrt_proj1;

//   Eigen::Matrix<double, 2, 6> ProjJac_e;
//   ProjJac_s << fx*(1+(std::pow(gx, 2) / gz2)), -fx*(gy/gz), fx/gz, 0, -fx*gx/gz2, -fx*gx*gy/gz2,
//     -fy*(1+(std::pow(my, 2)/mz2)), fy*gx*gy/gz2, fy*gx/gz, 0, fy/gz, -fy*gy/gz2;

//   Eigen::Matrix<double, 1, 6> deriv_wrt_e = deriv_wrt_proj * ProjJac_e;
//   Eigen::Matrix<double, 2, 6> Jacobian;
//   Jacobian << deriv_wrt_s,
//                 deriv_wrt_e;

//   _jacobianOplus[2] = Jacobian;
//     //Jacobian with respect to fp0
//   double _jac_start3 = 1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((K0*fp0*2.0+K1*fp1-K0*fq0*2.0-K1*fq1+K0*prevP0*2.0-K0*prevQ0*2.0+K1*prevP1-K1*prevQ1+fp0*fq0*4.0+fp1*fq1-fp0*prevP0*6.0+fp0*prevQ0*4.0-fp1*prevP1*2.0+fp1*prevQ1+fq0*prevP0*4.0-fq0*prevQ0*2.0+fq1*prevP1+prevP0*prevQ0*4.0+prevP1*prevQ1-(fp0*fp0)*3.0-fp1*fp1-fq0*fq0-(prevP0*prevP0)*3.0-prevP1*prevP1-prevQ0*prevQ0+1.0)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0+(fp1-fq1+prevP1-prevQ1)*(K0-fp0*2.0+fq0-prevP0*2.0+prevQ0)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0)*(-1.0/2.0);

//   //Jacobian with respect to fp1
//   double _jac_start4 = 1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((K0*fp0+K1*fp1*2.0-K0*fq0-K1*fq1*2.0+K0*prevP0-K0*prevQ0+K1*prevP1*2.0-K1*prevQ1*2.0+fp0*fq0+fp1*fq1*4.0-fp0*prevP0*2.0+fp0*prevQ0-fp1*prevP1*6.0+fp1*prevQ1*4.0+fq0*prevP0+fq1*prevP1*4.0-fq1*prevQ1*2.0+prevP0*prevQ0+prevP1*prevQ1*4.0-fp0*fp0-(fp1*fp1)*3.0-fq1*fq1-prevP0*prevP0-(prevP1*prevP1)*3.0-prevQ1*prevQ1+1.0)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0+(fp0-fq0+prevP0-prevQ0)*(K1-fp1*2.0+fq1-prevP1*2.0+prevQ1)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0)*(-1.0/2.0);

//   //Jacobian with respect to fq0
//   double _jac_end3 = (1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((K0*fp0*2.0+K1*fp1-K0*fq0*2.0-K1*fq1+K0*prevP0*2.0-K0*prevQ0*2.0+K1*prevP1-K1*prevQ1+fp0*fq0*2.0+fp1*fq1-fp0*prevP0*4.0+fp0*prevQ0*2.0-fp1*prevP1*2.0+fp1*prevQ1+fq0*prevP0*2.0+fq1*prevP1+prevP0*prevQ0*2.0+prevP1*prevQ1-(fp0*fp0)*2.0-fp1*fp1-(prevP0*prevP0)*2.0-prevP1*prevP1)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0-(-K0+fp0+prevP0)*(fp1-fq1+prevP1-prevQ1)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0))/2.0;

//   //Jacobian with respect to fq1
//   double _jac_end4 = (1.0/sqrt(pow(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1,2.0)+pow(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1,2.0))*((K0*fp0+K1*fp1*2.0-K0*fq0-K1*fq1*2.0+K0*prevP0-K0*prevQ0+K1*prevP1*2.0-K1*prevQ1*2.0+fp0*fq0+fp1*fq1*2.0-fp0*prevP0*2.0+fp0*prevQ0-fp1*prevP1*4.0+fp1*prevQ1*2.0+fq0*prevP0+fq1*prevP1*2.0+prevP0*prevQ0+prevP1*prevQ1*2.0-fp0*fp0-(fp1*fp1)*2.0-prevP0*prevP0-(prevP1*prevP1)*2.0)*(K1-fp1-prevP1-K1*(fp1*fp1)-K1*(fq1*fq1)-K1*(prevP1*prevP1)-K1*(prevQ1*prevQ1)+(fp0*fp0)*fp1-(fp0*fp0)*fq1+fp1*(fq1*fq1)-(fp1*fp1)*fq1*2.0+fp1*(prevP0*prevP0)+(fp0*fp0)*prevP1+fp1*(prevP1*prevP1)*3.0-(fp0*fp0)*prevQ1+(fp1*fp1)*prevP1*3.0+fp1*(prevQ1*prevQ1)-(fp1*fp1)*prevQ1*2.0-fq1*(prevP0*prevP0)-fq1*(prevP1*prevP1)*2.0+(fq1*fq1)*prevP1+(prevP0*prevP0)*prevP1-(prevP0*prevP0)*prevQ1+prevP1*(prevQ1*prevQ1)-(prevP1*prevP1)*prevQ1*2.0+fp1*fp1*fp1+prevP1*prevP1*prevP1-K0*fp0*fp1+K0*fp0*fq1+K0*fp1*fq0+K1*fp1*fq1*2.0-K0*fq0*fq1-K0*fp0*prevP1-K0*fp1*prevP0+K0*fp0*prevQ1+K0*fp1*prevQ0-K1*fp1*prevP1*2.0+K1*fp1*prevQ1*2.0+K0*fq0*prevP1+K0*fq1*prevP0-K0*fq0*prevQ1-K0*fq1*prevQ0+K1*fq1*prevP1*2.0-K1*fq1*prevQ1*2.0-K0*prevP0*prevP1+K0*prevP0*prevQ1+K0*prevP1*prevQ0-K0*prevQ0*prevQ1+K1*prevP1*prevQ1*2.0-fp0*fp1*fq0+fp0*fq0*fq1+fp0*fp1*prevP0*2.0-fp0*fp1*prevQ0-fp0*fq0*prevP1-fp0*fq1*prevP0*2.0-fp1*fq0*prevP0+fp0*fq0*prevQ1+fp0*fq1*prevQ0-fp1*fq1*prevP1*4.0+fp1*fq1*prevQ1*2.0+fq0*fq1*prevP0+fp0*prevP0*prevP1*2.0-fp0*prevP0*prevQ1*2.0-fp0*prevP1*prevQ0-fp1*prevP0*prevQ0+fp0*prevQ0*prevQ1-fp1*prevP1*prevQ1*4.0-fq0*prevP0*prevP1+fq0*prevP0*prevQ1+fq1*prevP0*prevQ0+fq1*prevP1*prevQ1*2.0-prevP0*prevP1*prevQ0+prevP0*prevQ0*prevQ1)*2.0-(-K1+fp1+prevP1)*(fp0-fq0+prevP0-prevQ0)*(K0-fp0-prevP0-K0*(fp0*fp0)-K0*(fq0*fq0)-K0*(prevP0*prevP0)-K0*(prevQ0*prevQ0)+fp0*(fp1*fp1)+fp0*(fq0*fq0)-(fp0*fp0)*fq0*2.0-(fp1*fp1)*fq0+fp0*(prevP0*prevP0)*3.0+(fp0*fp0)*prevP0*3.0+fp0*(prevP1*prevP1)+fp0*(prevQ0*prevQ0)-(fp0*fp0)*prevQ0*2.0+(fp1*fp1)*prevP0-(fp1*fp1)*prevQ0-fq0*(prevP0*prevP0)*2.0+(fq0*fq0)*prevP0-fq0*(prevP1*prevP1)+prevP0*(prevP1*prevP1)+prevP0*(prevQ0*prevQ0)-(prevP0*prevP0)*prevQ0*2.0-(prevP1*prevP1)*prevQ0+fp0*fp0*fp0+prevP0*prevP0*prevP0-K1*fp0*fp1+K0*fp0*fq0*2.0+K1*fp0*fq1+K1*fp1*fq0-K1*fq0*fq1-K0*fp0*prevP0*2.0+K0*fp0*prevQ0*2.0-K1*fp0*prevP1-K1*fp1*prevP0+K1*fp0*prevQ1+K1*fp1*prevQ0+K0*fq0*prevP0*2.0-K0*fq0*prevQ0*2.0+K1*fq0*prevP1+K1*fq1*prevP0-K1*fq0*prevQ1-K1*fq1*prevQ0+K0*prevP0*prevQ0*2.0-K1*prevP0*prevP1+K1*prevP0*prevQ1+K1*prevP1*prevQ0-K1*prevQ0*prevQ1-fp0*fp1*fq1+fp1*fq0*fq1+fp0*fp1*prevP1*2.0-fp0*fp1*prevQ1-fp0*fq0*prevP0*4.0+fp0*fq0*prevQ0*2.0-fp0*fq1*prevP1-fp1*fq0*prevP1*2.0-fp1*fq1*prevP0+fp1*fq0*prevQ1+fp1*fq1*prevQ0+fq0*fq1*prevP1-fp0*prevP0*prevQ0*4.0+fp1*prevP0*prevP1*2.0-fp0*prevP1*prevQ1-fp1*prevP0*prevQ1-fp1*prevP1*prevQ0*2.0+fp1*prevQ0*prevQ1+fq0*prevP0*prevQ0*2.0-fq1*prevP0*prevP1+fq0*prevP1*prevQ1+fq1*prevP1*prevQ0-prevP0*prevP1*prevQ1+prevP1*prevQ0*prevQ1)*2.0))/2.0;

//   Eigen::Matrix2d jac_Start;
//   Eigen::Matrix2d jac_End;
//   jac_Start << _jac_start1, _jac_start2,
//                 _jac_start3, _jac_start4;
//   jac_End << _jac_end1, _jac_end2,
//                 _jac_end3, _jac_end4;
//   _jacobianOplus[0] = jac_Start;
//   _jacobianOplus[1] = jac_End;

// }





//************************************************************************************************
bool EdgeSE3ProjectFlowDepth2::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectFlowDepth2::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectFlowDepth2::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

void EdgeSE3ProjectFlowDepth2::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAFlowDepth* vi = static_cast<VertexSBAFlowDepth*>(_vertices[0]);
  Vector2d obs(_measurement);
  Vector3d est = vi->estimate();
  Vector3d Xw;
  Xw << (obs(0)-cx)*est(2)/fx, (obs(1)-cy)*est(2)/fy, est(2);
  Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
  Vector3d xyz_trans = T.map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;
  double invfx = 1.0/fx;
  double invfy = 1.0/fy;

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;

  Matrix<double,2,3> K;
  K(0,0) = fx; K(0,1) = 0;  K(0,2) = cx;
  K(1,0) = 0;  K(1,1) = fy; K(1,2) = cy;

  Matrix<double,3,4> T_mat;
  T_mat.block(0,0,3,3) = T.rotation().toRotationMatrix();
  T_mat.col(3) = T.translation();

  Matrix<double,2,4> A;
  A = K*T_mat*Twl;

  _jacobianOplusXi(0,0) = 1.0;
  _jacobianOplusXi(0,1) = 0.0;
  _jacobianOplusXi(0,2) = -1.0*( A(0,0)*(obs(0)-cx)*invfx + A(0,1)*(obs(1)-cy)*invfy + A(0,2) );

  _jacobianOplusXi(1,0) = 0.0;
  _jacobianOplusXi(1,1) = 1.0;
  _jacobianOplusXi(1,2) = -1.0*( A(1,0)*(obs(0)-cx)*invfx + A(1,1)*(obs(1)-cy)*invfy + A(1,2) );

}

// ************************************************************************************************

bool EdgeSE3ProjectDepth::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectDepth::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectDepth::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

void EdgeSE3ProjectDepth::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBADepth* vi = static_cast<VertexSBADepth*>(_vertices[0]);
  Vector2d obs(_measurement);
  Matrix<double, 1, 1> est = vi->estimate();
  Vector3d Xw;
  Xw << (obs(0)-cx)*est(0)/fx, (obs(1)-cy)*est(0)/fy, est(0);
  Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
  Vector3d xyz_trans = T.map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;
  double invfx = 1.0/fx;
  double invfy = 1.0/fy;

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;

  Matrix<double,2,3> K;
  K(0,0) = fx; K(0,1) = 0;  K(0,2) = cx;
  K(1,0) = 0;  K(1,1) = fy; K(1,2) = cy;

  Matrix<double,3,4> T_mat;
  T_mat.block(0,0,3,3) = T.rotation().toRotationMatrix();
  T_mat.col(3) = T.translation();

  Matrix<double,2,4> A;
  A = K*T_mat*Twl;

  _jacobianOplusXi(0,0) = -1.0*( A(0,0)*(obs(0)-cx)*invfx + A(0,1)*(obs(1)-cy)*invfy + A(0,2) );
  _jacobianOplusXi(1,0) = -1.0*( A(1,0)*(obs(0)-cx)*invfx + A(1,1)*(obs(1)-cy)*invfy + A(1,2) );

}

// ************************************************************************************************

bool EdgeDepthPrior::read(std::istream& is){
  for (int i=0; i<1; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<1; i++)
    for (int j=i; j<1; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeDepthPrior::write(std::ostream& os) const {

  for (int i=0; i<1; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<1; i++)
    for (int j=i; j<1; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeDepthPrior::linearizeOplus(){
    Matrix<double, 1, 1> jac(-1.0);
    _jacobianOplusXi = jac;
}

// ************************************************************************************************

EdgeSE3ProjectFlowDepth3::EdgeSE3ProjectFlowDepth3() : BaseMultiEdge<2, Vector2d>()
{
  resize(3);
  _jacobianOplus[0].resize(2,2);
  _jacobianOplus[1].resize(2,6);
  _jacobianOplus[2].resize(2,1);
}

bool EdgeSE3ProjectFlowDepth3::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectFlowDepth3::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d EdgeSE3ProjectFlowDepth3::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

// void EdgeSE3ProjectFlowDepth3::linearizeOplus() {
//   VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
//   SE3Quat T(vj->estimate());
//   VertexSBAFlow* vi = static_cast<VertexSBAFlow*>(_vertices[0]);
//   VertexSBADepth* vk = static_cast<VertexSBADepth*>(_vertices[2]);
//   Vector2d obs(_measurement);
//   Vector2d flow = vi->estimate();
//   Matrix<double, 1, 1> depth = vk->estimate();
//   Vector3d Xw;
//   Xw << (obs(0)-cx)*depth(0)/fx, (obs(1)-cy)*depth(0)/fy, depth(0);
//   Xw = Twl.block(0,0,3,3)*Xw + Twl.col(3).head(3);
//   Vector3d xyz_trans = T.map(Xw);

//   double x = xyz_trans[0];
//   double y = xyz_trans[1];
//   double z = xyz_trans[2];
//   double z_2 = z*z;
//   double invfx = 1.0/fx;
//   double invfy = 1.0/fy;

//   Matrix<double,2,6> J_1;

//   J_1(0,0) =  x*y/z_2 *fx;
//   J_1(0,1) = -(1+(x*x/z_2)) *fx;
//   J_1(0,2) = y/z *fx;
//   J_1(0,3) = -1./z *fx;
//   J_1(0,4) = 0;
//   J_1(0,5) = x/z_2 *fx;

//   J_1(1,0) = (1+y*y/z_2) *fy;
//   J_1(1,1) = -x*y/z_2 *fy;
//   J_1(1,2) = -x/z *fy;
//   J_1(1,3) = 0;
//   J_1(1,4) = -1./z *fy;
//   J_1(1,5) = y/z_2 *fy;

//   _jacobianOplus[1] = J_1;

//   Matrix<double,2,3> K;
//   K(0,0) = fx; K(0,1) = 0;  K(0,2) = cx;
//   K(1,0) = 0;  K(1,1) = fy; K(1,2) = cy;

//   Matrix<double,3,4> T_mat;
//   T_mat.block(0,0,3,3) = T.rotation().toRotationMatrix();
//   T_mat.col(3) = T.translation();

//   Matrix<double,2,4> A;
//   A = K*T_mat*Twl;

//   Matrix<double,2,2> J_0;

//   J_0(0,0) = 1.0;
//   J_0(0,1) = 0.0;

//   J_0(1,0) = 0.0;
//   J_0(1,1) = 1.0;

//   _jacobianOplus[0] = J_0;

//   Matrix<double,2,1> J_2;

//   J_2(0,0) = -1.0*( A(0,0)*(obs(0)-cx)*invfx + A(0,1)*(obs(1)-cy)*invfy + A(0,2) );
//   J_2(1,0) = -1.0*( A(1,0)*(obs(0)-cx)*invfx + A(1,1)*(obs(1)-cy)*invfy + A(1,2) );

//   _jacobianOplus[2] = J_2;

// }

// ************************************************************************************************


// ************************************************************************************************

} // end namespace
