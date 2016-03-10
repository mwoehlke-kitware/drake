#include "drake/util/drakeGeometryUtil.h"
#include "drake/systems/plants/RigidBodyTree.h"

#include "EigenTypes.h"

using namespace std;
using namespace Eigen;

template <typename Scalar, typename DerivedPoints>
Matrix<Scalar, Dynamic, Dynamic>
RigidBodyTree::transformPointsJacobian(
    const KinematicsCache<Scalar>& cache,
    const MatrixBase<DerivedPoints>& points, int from_body_or_frame_ind,
    int to_body_or_frame_ind, bool in_terms_of_qdot) const {
  int cols = in_terms_of_qdot ? num_positions : num_velocities;
  int npoints = static_cast<int>(points.cols());

  auto points_base = transformPoints(cache, points, from_body_or_frame_ind,
                                     to_body_or_frame_ind);

  int body_ind = parseBodyOrFrameID(from_body_or_frame_ind);
  int base_ind = parseBodyOrFrameID(to_body_or_frame_ind);
  std::vector<int> v_or_q_indices;
  auto J_geometric =
      geometricJacobian(cache, base_ind, body_ind, to_body_or_frame_ind,
                        in_terms_of_qdot, &v_or_q_indices);

  auto Jomega = J_geometric.template topRows<SPACE_DIMENSION>();
  auto Jv = J_geometric.template bottomRows<SPACE_DIMENSION>();

  Matrix<Scalar, Dynamic, Dynamic> J(
      points_base.size(), cols);  // TODO: size at compile time
  J.setZero();

  int row_start = 0;
  for (int i = 0; i < npoints; i++) {
    // translation part
    int col = 0;
    for (std::vector<int>::iterator it = v_or_q_indices.begin();
         it != v_or_q_indices.end(); ++it) {
      J.template block<SPACE_DIMENSION, 1>(row_start, *it) = Jv.col(col);
      J.template block<SPACE_DIMENSION, 1>(row_start, *it).noalias() +=
          Jomega.col(col).cross(points_base.col(i));
      col++;
    }
    row_start += SPACE_DIMENSION;
  }

  return J;
};

template <typename Scalar, typename DerivedPoints>
Matrix<Scalar, Dynamic, 1>
RigidBodyTree::transformPointsJacobianDotTimesV(
    const KinematicsCache<Scalar>& cache,
    const MatrixBase<DerivedPoints>& points, int from_body_or_frame_ind,
    int to_body_or_frame_ind) const {
  cache.checkCachedKinematicsSettings(true, true,
                                      "transformPointsJacobianDotTimesV");

  auto r = transformPoints(cache, points, from_body_or_frame_ind,
                           to_body_or_frame_ind);
  int expressed_in = to_body_or_frame_ind;
  const auto twist =
      relativeTwist(cache, to_body_or_frame_ind, from_body_or_frame_ind,
                    to_body_or_frame_ind);
  const auto J_geometric_dot_times_v = geometricJacobianDotTimesV(
      cache, to_body_or_frame_ind, from_body_or_frame_ind, expressed_in);

  auto omega_twist = twist.template topRows<SPACE_DIMENSION>();
  auto v_twist = twist.template bottomRows<SPACE_DIMENSION>();

  auto rdots = (-r.colwise().cross(omega_twist)).eval();
  rdots.colwise() += v_twist;
  auto Jposdot_times_v_mat = (-rdots.colwise().cross(omega_twist)).eval();
  Jposdot_times_v_mat -=
      (r.colwise().cross(
           J_geometric_dot_times_v.template topRows<SPACE_DIMENSION>())).eval();
  Jposdot_times_v_mat.colwise() +=
      J_geometric_dot_times_v.template bottomRows<SPACE_DIMENSION>();

  return Map<VectorX<Scalar>>(Jposdot_times_v_mat.data(), r.size(), 1);
};

template DRAKERBM_EXPORT MatrixX<LargeADSV>
RigidBodyTree::transformPointsJacobian<LargeADSV, Matrix3Xd>(
    KinematicsCache<LargeADSV> const&, MatrixBase<Matrix3Xd> const&,
    int, int, bool) const;

template DRAKERBM_EXPORT MatrixX<DynamicADSV>
RigidBodyTree::transformPointsJacobian<DynamicADSV, Matrix3Xd>(
    KinematicsCache<DynamicADSV> const&, MatrixBase<Matrix3Xd> const&,
    int, int, bool) const;

template DRAKERBM_EXPORT MatrixX<double>
RigidBodyTree::transformPointsJacobian<double, Matrix3Xd>(
    KinematicsCache<double> const&, MatrixBase<Matrix3Xd> const&,
    int, int, bool) const;

template DRAKERBM_EXPORT VectorX<double>
RigidBodyTree::transformPointsJacobianDotTimesV<double, Matrix3Xd>(
    KinematicsCache<double> const&, MatrixBase<Matrix3Xd> const&,
    int, int) const;

template DRAKERBM_EXPORT MatrixX<double>
RigidBodyTree::transformPointsJacobian<double, Vector3d>(
    KinematicsCache<double> const&, MatrixBase<Vector3d> const&,
    int, int, bool) const;

template DRAKERBM_EXPORT VectorX<double>
RigidBodyTree::transformPointsJacobianDotTimesV<double, Vector3d>(
    KinematicsCache<double> const&, MatrixBase<Vector3d> const&,
    int, int) const;

template DRAKERBM_EXPORT MatrixX<double>
RigidBodyTree::transformPointsJacobian<double, Block<Matrix3Xd, 3, 1, true>>(
    KinematicsCache<double> const&,
    MatrixBase<Block<Matrix3Xd, 3, 1, true>> const&,
    int, int, bool) const;

template DRAKERBM_EXPORT MatrixX<LargeADSV>
RigidBodyTree::transformPointsJacobian<
    LargeADSV, Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>>(
    KinematicsCache<LargeADSV> const&,
    MatrixBase<Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>> const&,
    int, int, bool) const;

template DRAKERBM_EXPORT MatrixX<DynamicADSV>
RigidBodyTree::transformPointsJacobian<
    DynamicADSV, Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>>(
    KinematicsCache<DynamicADSV> const&,
    MatrixBase<Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>> const&,
    int, int, bool) const;

template DRAKERBM_EXPORT MatrixX<double>
RigidBodyTree::transformPointsJacobian<
    double, Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>>(
    KinematicsCache<double> const&,
    MatrixBase<Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>> const&,
    int, int, bool) const;

template DRAKERBM_EXPORT VectorX<LargeADSV>
RigidBodyTree::transformPointsJacobianDotTimesV<
    LargeADSV, Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>>(
    KinematicsCache<LargeADSV> const&,
    MatrixBase<Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>> const&,
    int, int) const;

template DRAKERBM_EXPORT VectorX<DynamicADSV>
RigidBodyTree::transformPointsJacobianDotTimesV<
    DynamicADSV, Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>>(
    KinematicsCache<DynamicADSV> const&,
    MatrixBase<Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>> const&,
    int, int) const;

template DRAKERBM_EXPORT VectorX<double>
RigidBodyTree::transformPointsJacobianDotTimesV<
    double, Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>>(
    KinematicsCache<double> const&,
    MatrixBase<Map<Matrix3Xd const, 0, Eigen::Stride<0, 0>>> const&,
    int, int) const;
