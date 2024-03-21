#ifndef G2O_EDGE_SE3_ORTHO_LINE_H_
#define G2O_EDGE_SE3_ORTHO_LINE_H_

#include "../core/base_binary_edge.h"
#include "vertex_se3.h"
#include "vertex_line.h"
#include "parameter_se3_offset.h"
#include "g2o_types_slam3d_api.h"

namespace g2o {

class G2O_TYPES_SLAM3D_API EdgeSE3OrthoLine : public BaseBinaryEdge<2, Vector2d, VertexSE3, VertexLine> {
  public:
    //TODO: check if the below is needed
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3OrthoLine(){};
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    Vector2d returnError();
    // return the error estimate as a 3-vector
    void computeError();
    // jacobian
    void linearizeOplus();
    
    void setMeasurement(const Vector6& m) {
      _measurement = m;
    }
    
  private:
    Eigen::Matrix<double, 6, 1> orthonormal2plucker(std::pair<Eigen::Matrix3d, Eigen::Matrix2d> line);
    Eigen::Matrix<double, 6, 1> _measurement;
};

} // end namespace g2o

#endif