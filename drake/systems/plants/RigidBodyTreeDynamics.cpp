#include "drake/util/drakeGeometryUtil.h"
#include "drake/systems/plants/RigidBodyTree.h"

#include "EigenTypes.h"

using namespace std;
using namespace Eigen;

template <typename Scalar>
Matrix<Scalar, Eigen::Dynamic, 1> RigidBodyTree::dynamicsBiasTerm(
    KinematicsCache<Scalar>& cache,
    const eigen_aligned_unordered_map<RigidBody const*,
                                      Matrix<Scalar, TWIST_SIZE, 1>>& f_ext,
    bool include_velocity_terms) const {
  Matrix<Scalar, Eigen::Dynamic, 1> vd(num_velocities, 1);
  vd.setZero();
  return inverseDynamics(cache, f_ext, vd, include_velocity_terms);
};

template <typename DerivedV>
Matrix<typename DerivedV::Scalar, Dynamic, 1> RigidBodyTree::frictionTorques(
    Eigen::MatrixBase<DerivedV> const& v) const {
  typedef typename DerivedV::Scalar Scalar;
  Matrix<Scalar, Dynamic, 1> ret(num_velocities, 1);

  for (auto it = bodies.begin(); it != bodies.end(); ++it) {
    RigidBody& body = **it;
    if (body.hasParent()) {
      const DrakeJoint& joint = body.getJoint();
      int nv_joint = joint.getNumVelocities();
      int v_start_joint = body.velocity_num_start;
      auto v_body = v.middleRows(v_start_joint, nv_joint);
      ret.middleRows(v_start_joint, nv_joint) = joint.frictionTorque(v_body);
    }
  }

  return ret;
}

template <typename Scalar>
Matrix<Scalar, Eigen::Dynamic, 1> RigidBodyTree::inverseDynamics(
    KinematicsCache<Scalar>& cache,
    const eigen_aligned_unordered_map<RigidBody const*,
                                      Matrix<Scalar, TWIST_SIZE, 1>>& f_ext,
    const Matrix<Scalar, Eigen::Dynamic, 1>& vd,
    bool include_velocity_terms) const {
  cache.checkCachedKinematicsSettings(
      include_velocity_terms, include_velocity_terms, "inverseDynamics");

  updateCompositeRigidBodyInertias(cache);

  typedef typename Eigen::Matrix<Scalar, TWIST_SIZE, 1> Vector6;

  Vector6 root_accel = -a_grav.cast<Scalar>();
  Matrix<Scalar, TWIST_SIZE, Eigen::Dynamic> net_wrenches(TWIST_SIZE,
                                                          bodies.size());
  net_wrenches.col(0).setZero();

  for (int i = 0; i < bodies.size(); i++) {
    RigidBody& body = *bodies[i];
    if (body.hasParent()) {
      const auto& element = cache.getElement(body);

      Vector6 spatial_accel = root_accel;
      if (include_velocity_terms)
        spatial_accel += element.motion_subspace_in_world_dot_times_v;

      int nv_joint = body.getJoint().getNumVelocities();
      auto vdJoint = vd.middleRows(body.velocity_num_start, nv_joint);
      spatial_accel.noalias() += element.motion_subspace_in_world * vdJoint;

      net_wrenches.col(i).noalias() = element.inertia_in_world * spatial_accel;
      if (include_velocity_terms) {
        auto I_times_twist =
            (element.inertia_in_world * element.twist_in_world).eval();
        net_wrenches.col(i).noalias() +=
            crossSpatialForce(element.twist_in_world, I_times_twist);
      }

      auto f_ext_iterator = f_ext.find(bodies[i].get());
      if (f_ext_iterator != f_ext.end()) {
        const auto& f_ext_i = f_ext_iterator->second;
        net_wrenches.col(i) -=
            transformSpatialForce(element.transform_to_world, f_ext_i);
      }
    }
  }

  Matrix<Scalar, Eigen::Dynamic, 1> ret(num_velocities, 1);

  for (int i = static_cast<int>(bodies.size()) - 1; i >= 0; i--) {
    RigidBody& body = *bodies[i];
    if (body.hasParent()) {
      const auto& element = cache.getElement(body);
      const auto& net_wrenches_const = net_wrenches;  // eliminates the need for
                                                      // another explicit
                                                      // instantiation
      auto joint_wrench = net_wrenches_const.col(i);
      int nv_joint = body.getJoint().getNumVelocities();
      auto J_transpose = element.motion_subspace_in_world.transpose();
      ret.middleRows(body.velocity_num_start, nv_joint).noalias() =
          J_transpose * joint_wrench;
      auto parent_net_wrench = net_wrenches.col(body.parent->body_index);
      parent_net_wrench += joint_wrench;
    }
  }

  if (include_velocity_terms) ret += frictionTorques(cache.getV());

  return ret;
}

template DRAKERBM_EXPORT VectorX<LargeADSV>
RigidBodyTree::dynamicsBiasTerm<LargeADSV>(
    KinematicsCache<LargeADSV>&,
    eigen_aligned_unordered_map<
        RigidBody const*, TwistVector<LargeADSV>> const&,
    bool) const;

template DRAKERBM_EXPORT VectorX<DynamicADSV>
RigidBodyTree::dynamicsBiasTerm<DynamicADSV>(
    KinematicsCache<DynamicADSV>&,
    eigen_aligned_unordered_map<
        RigidBody const*, TwistVector<DynamicADSV>> const&,
    bool) const;

template DRAKERBM_EXPORT VectorX<double>
RigidBodyTree::dynamicsBiasTerm<double>(
    KinematicsCache<double>&,
    eigen_aligned_unordered_map<
        RigidBody const*, TwistVector<double>> const&,
    bool) const;

template DRAKERBM_EXPORT VectorXd
RigidBodyTree::inverseDynamics<double>(
    KinematicsCache<double>&,
    eigen_aligned_unordered_map<
        RigidBody const*, TwistVector<double>> const&,
    VectorXd const&, bool) const;
