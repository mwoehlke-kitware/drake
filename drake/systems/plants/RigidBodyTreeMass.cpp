#include "drake/util/drakeGeometryUtil.h"
#include "drake/systems/plants/RigidBodyTree.h"

#include "EigenTypes.h"

using namespace std;
using namespace Eigen;

template <typename Scalar>
TwistMatrix<Scalar> RigidBodyTree::worldMomentumMatrix(
    KinematicsCache<Scalar>& cache, const std::set<int>& robotnum,
    bool in_terms_of_qdot) const {
  cache.checkCachedKinematicsSettings(false, false, "worldMomentumMatrix");
  updateCompositeRigidBodyInertias(cache);

  int nq = num_positions;
  int nv = num_velocities;
  int ncols = in_terms_of_qdot ? nq : nv;
  TwistMatrix<Scalar> ret(TWIST_SIZE, ncols);
  ret.setZero();
  int gradient_row_start = 0;
  for (int i = 0; i < bodies.size(); i++) {
    RigidBody& body = *bodies[i];

    if (body.hasParent()) {
      const auto& element = cache.getElement(body);
      const DrakeJoint& joint = body.getJoint();
      int ncols_joint =
          in_terms_of_qdot ? joint.getNumPositions() : joint.getNumVelocities();
      if (isBodyPartOfRobot(body, robotnum)) {
        int start = in_terms_of_qdot ? body.position_num_start
                                     : body.velocity_num_start;

        if (in_terms_of_qdot) {
          auto crb =
              (element.crb_in_world * element.motion_subspace_in_world).eval();
          ret.middleCols(start, ncols_joint).noalias() =
              crb * element.qdot_to_v;
        } else {
          ret.middleCols(start, ncols_joint).noalias() =
              element.crb_in_world * element.motion_subspace_in_world;
        }
      }
      gradient_row_start += TWIST_SIZE * ncols_joint;
    }
  }
  return ret;
}

template <typename Scalar>
TwistVector<Scalar> RigidBodyTree::worldMomentumMatrixDotTimesV(
    KinematicsCache<Scalar>& cache, const std::set<int>& robotnum) const {
  cache.checkCachedKinematicsSettings(true, true,
                                      "worldMomentumMatrixDotTimesV");
  updateCompositeRigidBodyInertias(cache);

  TwistVector<Scalar> ret;
  ret.setZero();
  for (int i = 0; i < bodies.size(); i++) {
    RigidBody& body = *bodies[i];
    if (body.hasParent()) {
      if (isBodyPartOfRobot(body, robotnum)) {
        const auto& element = cache.getElement(body);
        ret.noalias() += element.inertia_in_world *
                         element.motion_subspace_in_world_dot_times_v;
        auto inertia_times_twist =
            (element.inertia_in_world * element.twist_in_world).eval();
        ret.noalias() +=
            crossSpatialForce(element.twist_in_world, inertia_times_twist);
      }
    }
  }

  return ret;
}

template <typename Scalar>
TwistMatrix<Scalar>
RigidBodyTree::centroidalMomentumMatrix(KinematicsCache<Scalar>& cache,
                                        const std::set<int>& robotnum,
                                        bool in_terms_of_qdot) const {
  // kinematics cache checks already being done in worldMomentumMatrix.
  auto ret = worldMomentumMatrix(cache, robotnum, in_terms_of_qdot);

  // transform from world frame to COM frame
  auto com = centerOfMass(cache, robotnum);
  auto angular_momentum_matrix = ret.template topRows<SPACE_DIMENSION>();
  auto linear_momentum_matrix = ret.template bottomRows<SPACE_DIMENSION>();
  angular_momentum_matrix += linear_momentum_matrix.colwise().cross(com);

  //  Valid for more general frame transformations but slower:
  //  Eigen::Transform<Scalar, SPACE_DIMENSION, Eigen::Isometry>
  //  T(Translation<Scalar, SPACE_DIMENSION>(-com.value()));
  //  ret.value() = transformSpatialForce(T, ret.value());

  return ret;
}

template <typename Scalar>
TwistVector<Scalar> RigidBodyTree::centroidalMomentumMatrixDotTimesV(
    KinematicsCache<Scalar>& cache, const std::set<int>& robotnum) const {
  // kinematics cache checks already being done in worldMomentumMatrixDotTimesV
  auto ret = worldMomentumMatrixDotTimesV(cache, robotnum);

  // transform from world frame to COM frame:
  auto com = centerOfMass(cache, robotnum);
  auto angular_momentum_matrix_dot_times_v =
      ret.template topRows<SPACE_DIMENSION>();
  auto linear_momentum_matrix_dot_times_v =
      ret.template bottomRows<SPACE_DIMENSION>();
  angular_momentum_matrix_dot_times_v +=
      linear_momentum_matrix_dot_times_v.cross(com);

  //  Valid for more general frame transformations but slower:
  //  Eigen::Transform<Scalar, SPACE_DIMENSION, Eigen::Isometry>
  //  T(Translation<Scalar, SPACE_DIMENSION>(-com.value()));
  //  ret.value() = transformSpatialForce(T, ret.value());

  return ret;
}

template <typename Scalar>
Eigen::Matrix<Scalar, SPACE_DIMENSION, 1> RigidBodyTree::centerOfMass(
    KinematicsCache<Scalar>& cache, const std::set<int>& robotnum) const {
  cache.checkCachedKinematicsSettings(false, false, "centerOfMass");

  Eigen::Matrix<Scalar, SPACE_DIMENSION, 1> com;
  com.setZero();
  double m = 0.0;

  for (int i = 0; i < bodies.size(); i++) {
    RigidBody& body = *bodies[i];
    if (isBodyPartOfRobot(body, robotnum)) {
      if (body.mass > 0) {
        com.noalias() +=
            body.mass * transformPoints(cache, body.com.cast<Scalar>(), i, 0);
      }
      m += body.mass;
    }
  }
  if (m > 0.0) com /= m;

  return com;
}

template <typename Scalar>
Matrix<Scalar, SPACE_DIMENSION, Eigen::Dynamic>
RigidBodyTree::centerOfMassJacobian(KinematicsCache<Scalar>& cache,
                                    const std::set<int>& robotnum,
                                    bool in_terms_of_qdot) const {
  cache.checkCachedKinematicsSettings(false, false, "centerOfMassJacobian");
  auto A = worldMomentumMatrix(cache, robotnum, in_terms_of_qdot);
  double total_mass = getMass(robotnum);
  return A.template bottomRows<SPACE_DIMENSION>() / total_mass;
}

template <typename Scalar>
Matrix<Scalar, SPACE_DIMENSION, 1> RigidBodyTree::centerOfMassJacobianDotTimesV(
    KinematicsCache<Scalar>& cache, const std::set<int>& robotnum) const {
  // kinematics cache checks are already being done in
  // centroidalMomentumMatrixDotTimesV
  auto cmm_dot_times_v = centroidalMomentumMatrixDotTimesV(cache, robotnum);
  double total_mass = getMass(robotnum);
  return cmm_dot_times_v.template bottomRows<SPACE_DIMENSION>() / total_mass;
}

template <typename Scalar>
TwistVector<Scalar> RigidBodyTree::transformSpatialAcceleration(
    const KinematicsCache<Scalar>& cache,
    const TwistVector<Scalar>& spatial_acceleration, int base_ind,
    int body_ind, int old_expressed_in_body_or_frame_ind,
    int new_expressed_in_body_or_frame_ind) const {
  cache.checkCachedKinematicsSettings(true, true,
                                      "transformSpatialAcceleration");

  if (old_expressed_in_body_or_frame_ind ==
      new_expressed_in_body_or_frame_ind) {
    return spatial_acceleration;
  }

  auto twist_of_body_wrt_base = relativeTwist(
      cache, base_ind, body_ind, old_expressed_in_body_or_frame_ind);
  auto twist_of_old_wrt_new = relativeTwist(
      cache, new_expressed_in_body_or_frame_ind,
      old_expressed_in_body_or_frame_ind, old_expressed_in_body_or_frame_ind);
  auto T_old_to_new =
      relativeTransform(cache, new_expressed_in_body_or_frame_ind,
                        old_expressed_in_body_or_frame_ind);

  TwistVector<Scalar> spatial_accel_temp =
      crossSpatialMotion(twist_of_old_wrt_new, twist_of_body_wrt_base);
  spatial_accel_temp += spatial_acceleration;
  return transformSpatialMotion(T_old_to_new, spatial_accel_temp);
}

template DRAKERBM_EXPORT Vector3<LargeADSV>
RigidBodyTree::centerOfMass<LargeADSV>(
    KinematicsCache<LargeADSV>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT Vector3<DynamicADSV>
RigidBodyTree::centerOfMass<DynamicADSV>(
    KinematicsCache<DynamicADSV>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT Vector3<double>
RigidBodyTree::centerOfMass<double>(
    KinematicsCache<double>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT Matrix3X<LargeADSV>
RigidBodyTree::centerOfMassJacobian<LargeADSV>(
    KinematicsCache<LargeADSV>&,
    set<int, less<int>, allocator<int>> const&, bool) const;

template DRAKERBM_EXPORT Matrix3X<DynamicADSV>
RigidBodyTree::centerOfMassJacobian<DynamicADSV>(
    KinematicsCache<DynamicADSV>&,
    set<int, less<int>, allocator<int>> const&, bool) const;

template DRAKERBM_EXPORT Matrix3X<double>
RigidBodyTree::centerOfMassJacobian<double>(
    KinematicsCache<double>&,
    set<int, less<int>, allocator<int>> const&, bool) const;

template DRAKERBM_EXPORT TwistMatrix<LargeADSV>
RigidBodyTree::centroidalMomentumMatrix<LargeADSV>(
    KinematicsCache<LargeADSV>&,
    set<int, less<int>, allocator<int>> const&, bool) const;

template DRAKERBM_EXPORT TwistMatrix<DynamicADSV>
RigidBodyTree::centroidalMomentumMatrix<DynamicADSV>(
    KinematicsCache<DynamicADSV>&,
    set<int, less<int>, allocator<int>> const&, bool) const;

template DRAKERBM_EXPORT TwistMatrix<double>
RigidBodyTree::centroidalMomentumMatrix<double>(
    KinematicsCache<double>&,
    set<int, less<int>, allocator<int>> const&, bool) const;

template DRAKERBM_EXPORT Vector3<LargeADSV>
RigidBodyTree::centerOfMassJacobianDotTimesV<LargeADSV>(
    KinematicsCache<LargeADSV>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT Vector3<DynamicADSV>
RigidBodyTree::centerOfMassJacobianDotTimesV<DynamicADSV>(
    KinematicsCache<DynamicADSV>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT Vector3<double>
RigidBodyTree::centerOfMassJacobianDotTimesV<double>(
    KinematicsCache<double>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT TwistVector<LargeADSV>
RigidBodyTree::centroidalMomentumMatrixDotTimesV<LargeADSV>(
    KinematicsCache<LargeADSV>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT TwistVector<DynamicADSV>
RigidBodyTree::centroidalMomentumMatrixDotTimesV<DynamicADSV>(
    KinematicsCache<DynamicADSV>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT TwistVector<double>
RigidBodyTree::centroidalMomentumMatrixDotTimesV<double>(
    KinematicsCache<double>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT TwistMatrix<double>
RigidBodyTree::worldMomentumMatrix<double>(
    KinematicsCache<double>&,
    set<int, less<int>, allocator<int>> const&, bool) const;

template DRAKERBM_EXPORT TwistVector<double>
RigidBodyTree::worldMomentumMatrixDotTimesV<double>(
    KinematicsCache<double>&,
    set<int, less<int>, allocator<int>> const&) const;

template DRAKERBM_EXPORT TwistVector<LargeADSV>
RigidBodyTree::transformSpatialAcceleration<LargeADSV>(
    KinematicsCache<LargeADSV> const&, TwistVector<LargeADSV> const&,
    int, int, int, int) const;

template DRAKERBM_EXPORT TwistVector<DynamicADSV>
RigidBodyTree::transformSpatialAcceleration<DynamicADSV>(
    KinematicsCache<DynamicADSV> const&, TwistVector<DynamicADSV> const&,
    int, int, int, int) const;

template DRAKERBM_EXPORT TwistVector<double>
RigidBodyTree::transformSpatialAcceleration<double>(
    KinematicsCache<double> const&, TwistVector<double> const&,
    int, int, int, int) const;
