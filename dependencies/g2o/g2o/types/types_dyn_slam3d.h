/**
* This file is part of VDO-SLAM.
*
* Copyright (C) 2019-2020 Jun Zhang <jun doc zhang2 at anu dot edu doc au> (The Australian National University)
* For more information see <https://github.com/halajun/DynamicObjectSLAM>
*
**/

#ifndef G2O_DYNAMIC_SLAM3D
#define G2O_DYNAMIC_SLAM3D

#include "vertex_se3.h"
#include "edge_se3.h"
#include "vertex_pointxyz.h"
#include "vertex_line.h"
#include "../core/base_multi_edge.h"
#include "../core/base_unary_edge.h"

namespace g2o {

namespace types_dyn_slam3d {
void init();
}

class LandmarkMotionTernaryEdge: public BaseMultiEdge<3,Vector3>
{
    public:
     LandmarkMotionTernaryEdge();

    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    void computeError();
    void linearizeOplus();

    virtual void setMeasurement(const Vector3& m){
      _measurement = m;
    }

    Vector3 returnError();

private:
    Eigen::Matrix<number_t,3,6,Eigen::ColMajor> J;

};

class LineLandmarkMotionTernaryEdge: public BaseMultiEdge<2,Vector2>
{
    public:
     LineLandmarkMotionTernaryEdge();

    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    void computeError();
    void linearizeOplus();

    virtual void setMeasurement(const Vector2& m){
      _measurement = m;
    }

    Vector2 returnError();

    Eigen::Matrix<double, 6, 1> orthonormal2plucker(std::pair<Eigen::Matrix3d, Eigen::Matrix2d> line);

};


class EdgeSE3Altitude: public BaseUnaryEdge<1, double, VertexSE3>
{
    public:
        EdgeSE3Altitude();

        virtual bool read(std::istream& is);
        virtual bool write(std::ostream& os) const;
        void computeError();
        void linearizeOplus();

        virtual void setMeasurement(const double& m){
            _measurement = m;
        }

    private:
        Eigen::Matrix<number_t,1,6,Eigen::RowMajor> J;
};


} // end namespace

#endif
