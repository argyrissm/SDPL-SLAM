#ifndef G2O_VERTEX_LINE_
#define G2O_VERTEX_LINE_

#include "../../config.h"
#include "../core/base_vertex.h"
#include "../core/hyper_graph_action.h"
#include "isometry3d_mappings.h"
#include "g2o_types_slam3d_api.h"


namespace g2o {


    class G2O_TYPES_SLAM3D_API VertexLine : public BaseVertex<4, std::pair<Eigen::Matrix<double, 3, 3>, Eigen::Matrix<double, 2, 2>>>
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            VertexLine(){};

            virtual void setToOriginImpl() {
                Eigen::Matrix3d mat3_tmp = Eigen::Matrix3d::Zero();
                Eigen::Matrix2d mat2_tmp = Eigen::Matrix2d::Zero();
    
                _estimate = std::make_pair(mat3_tmp, mat2_tmp);
            }

            virtual bool read(std::istream& is);
            virtual bool write(std::ostream& os) const;

            // est_(0, 1, 2, 3) = theta_1, theta_2, theta_3 for U, theta for W
            virtual void oplusImpl(const number_t* est) //a 4d vector
            {
                Eigen::Map<const Vector4> _est(est);
                Eigen::Matrix3d U_matrix = _estimate.first;
                Eigen::Matrix2d W_matrix = _estimate.second;
                Eigen::Matrix3d Rx, Ry, Rz;
                Rx << 1, 0, 0,
                    0, cos(_est[0]), -sin(_est[0]),
                    0, sin(_est[0]), cos(est[0]);
                Ry << cos(_est[1]), 0, sin(_est[1]),
                    0, 1, 0,
                    -sin(_est[1]), 0, cos(_est[1]);
                Rz << cos(_est[2]), -sin(_est[2]), 0,
                    sin(_est[2]), cos(_est[2]), 0,
                    0, 0, 1;
                Eigen::Matrix3d update_u = Rx * Ry * Rz;

                //for update of w
                Eigen::Matrix2d update_w;
                update_w << cos(_est[3]), -sin(_est[3]),
                        sin(_est[3]), cos(_est[3]);

                U_matrix = U_matrix * update_u;
                W_matrix = W_matrix * update_w;

                _estimate = std::make_pair(U_matrix, W_matrix);
            }

            virtual bool getEstimateData(std::pair<Eigen::Matrix<double, 3, 3>, Eigen::Matrix<double, 2, 2>> &est) const{
                est = _estimate;
                return true;
            }
            
    };

} //end namespace
#endif