#include "vertex_line.h"
#include <stdio.h>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_primitives.h"
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <typeinfo>

namespace g2o {
  bool VertexLine::read(std::istream& is) {
    Eigen::Matrix3d tmp_mat1;
    Eigen::Matrix2d tmp_mat2;
    for (int i =0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            is >> tmp_mat1(i, j);
        }
    }
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            is >> tmp_mat2(i, j);
        }
    }
    setEstimate(std::make_pair(tmp_mat1, tmp_mat2));
    return true;
  }

  bool VertexLine::write(std::ostream& os) const {
    std::pair<Eigen::Matrix3d, Eigen::Matrix2d> lv=estimate();
    for (int i =0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            os << lv.first(i, j) << " ";
        }
    }
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            os << lv.second(i, j) << " ";
        }
    }
    return os.good();
  }
}