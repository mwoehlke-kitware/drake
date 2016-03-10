#include "drake/util/drakeGeometryUtil.h"
#include "drake/systems/plants/RigidBodyTree.h"
#include "drake/systems/plants/EigenTypes.h"

using namespace std;
using namespace Eigen;

template <typename Scalar>
Matrix<Scalar, RPY_SIZE, Dynamic>
RigidBodyTree::relativeRollPitchYawJacobian(
    const KinematicsCache<Scalar>& cache, int from_body_or_frame_ind,
    int to_body_or_frame_ind, bool in_terms_of_qdot) const {
  int body_ind = parseBodyOrFrameID(from_body_or_frame_ind);
  int base_ind = parseBodyOrFrameID(to_body_or_frame_ind);
  KinematicPath kinematic_path = findKinematicPath(base_ind, body_ind);
  auto J_geometric = geometricJacobian(cache, base_ind, body_ind,
                                       to_body_or_frame_ind, in_terms_of_qdot);
  auto Jomega = J_geometric.template topRows<SPACE_DIMENSION>();
  auto rpy =
      relativeRollPitchYaw(cache, from_body_or_frame_ind, to_body_or_frame_ind);
  Matrix<Scalar, RPY_SIZE, SPACE_DIMENSION> Phi;
  angularvel2rpydotMatrix(
      rpy, Phi,
      static_cast<typename Gradient<decltype(Phi), Dynamic>::type*>(nullptr),
      static_cast<typename Gradient<decltype(Phi), Dynamic, 2>::type*>(
          nullptr));
  return compactToFull((Phi * Jomega).eval(), kinematic_path.joint_path,
                       in_terms_of_qdot);
};

template <typename Scalar>
Matrix<Scalar, Dynamic, 1>
RigidBodyTree::relativeRollPitchYawJacobianDotTimesV(
    const KinematicsCache<Scalar>& cache, int from_body_or_frame_ind,
    int to_body_or_frame_ind) const {
  cache.checkCachedKinematicsSettings(true, true,
                                      "relativeRollPitchYawJacobianDotTimesV");

  auto rpy =
      relativeRollPitchYaw(cache, from_body_or_frame_ind, to_body_or_frame_ind);
  Matrix<Scalar, RPY_SIZE, SPACE_DIMENSION> Phi;
  angularvel2rpydotMatrix(
      rpy, Phi,
      static_cast<typename Gradient<decltype(Phi), Dynamic>::type*>(nullptr),
      static_cast<typename Gradient<decltype(Phi), Dynamic, 2>::type*>(
          nullptr));

  int expressed_in = to_body_or_frame_ind;
  const auto twist = relativeTwist(cache, to_body_or_frame_ind,
                                   from_body_or_frame_ind, expressed_in);
  auto omega_twist = twist.template topRows<SPACE_DIMENSION>();
  auto rpydot = (Phi * omega_twist).eval();

  using ADScalar = AutoDiffScalar<Matrix<Scalar, Dynamic,
                                         1>>;  // would prefer to use 1 instead
                                               // of Dynamic, but this causes
                                               // issues related to
  // http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1006 on MSVC
  // 32 bit
  auto rpy_autodiff = rpy.template cast<ADScalar>().eval();
  gradientMatrixToAutoDiff(rpydot, rpy_autodiff);
  Matrix<ADScalar, RPY_SIZE, SPACE_DIMENSION> Phi_autodiff;
  angularvel2rpydotMatrix(
      rpy_autodiff, Phi_autodiff,
      static_cast<typename Gradient<decltype(Phi_autodiff), Dynamic>::type*>(
          nullptr),
      static_cast<typename Gradient<decltype(Phi_autodiff), Dynamic, 2>::type*>(
          nullptr));
  auto Phidot_vector = autoDiffToGradientMatrix(Phi_autodiff);
  Map<Matrix<Scalar, RPY_SIZE, SPACE_DIMENSION>> Phid(Phidot_vector.data());

  const auto J_geometric_dot_times_v = geometricJacobianDotTimesV(
      cache, to_body_or_frame_ind, from_body_or_frame_ind, expressed_in);
  auto ret = (Phid * omega_twist).eval();
  ret.noalias() +=
      Phi * J_geometric_dot_times_v.template topRows<SPACE_DIMENSION>();
  return ret;
};

template DRAKERBM_EXPORT Matrix3X<double>
RigidBodyTree::relativeRollPitchYawJacobian<double>(
    KinematicsCache<double> const&, int, int, bool) const;

template DRAKERBM_EXPORT Matrix3X<LargeADSV>
RigidBodyTree::relativeRollPitchYawJacobian<LargeADSV>(
    KinematicsCache<LargeADSV> const&, int, int, bool) const;

template DRAKERBM_EXPORT Matrix3X<DynamicADSV>
RigidBodyTree::relativeRollPitchYawJacobian<DynamicADSV>(
    KinematicsCache<DynamicADSV> const&, int, int, bool) const;

template DRAKERBM_EXPORT VectorX<double>
RigidBodyTree::relativeRollPitchYawJacobianDotTimesV<double>(
    KinematicsCache<double> const&, int, int) const;

template DRAKERBM_EXPORT VectorX<LargeADSV>
RigidBodyTree::relativeRollPitchYawJacobianDotTimesV<LargeADSV>(
    KinematicsCache<LargeADSV> const&, int, int) const;

template DRAKERBM_EXPORT VectorX<DynamicADSV>
RigidBodyTree::relativeRollPitchYawJacobianDotTimesV<DynamicADSV>(
    KinematicsCache<DynamicADSV> const&, int, int) const;
