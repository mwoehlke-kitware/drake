#include "drake/util/drakeGeometryUtil.h"
#include "drake/systems/plants/RigidBodyTree.h"

#include "EigenTypes.h"

using namespace std;
using namespace Eigen;

template <typename Scalar>
void RigidBodyTree::updateCompositeRigidBodyInertias(
    KinematicsCache<Scalar>& cache) const {
  cache.checkCachedKinematicsSettings(false, false,
                                      "updateCompositeRigidBodyInertias");

  if (!cache.areInertiasCached()) {
    for (int i = 0; i < bodies.size(); i++) {
      auto& element = cache.getElement(*bodies[i]);
      element.inertia_in_world = transformSpatialInertia(
          element.transform_to_world, bodies[i]->I.cast<Scalar>());
      element.crb_in_world = element.inertia_in_world;
    }

    for (int i = static_cast<int>(bodies.size()) - 1; i >= 0; i--) {
      if (bodies[i]->hasParent()) {
        const auto& element = cache.getElement(*bodies[i]);
        auto& parent_element = cache.getElement(*(bodies[i]->parent));
        parent_element.crb_in_world += element.crb_in_world;
      }
    }
  }
  cache.setInertiasCached();
}

template <typename Scalar>
Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> RigidBodyTree::massMatrix(
    KinematicsCache<Scalar>& cache) const {
  cache.checkCachedKinematicsSettings(false, false, "massMatrix");

  int nv = num_velocities;
  Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ret(nv, nv);
  ret.setZero();

  updateCompositeRigidBodyInertias(cache);

  for (int i = 0; i < bodies.size(); i++) {
    RigidBody& body_i = *bodies[i];
    if (body_i.hasParent()) {
      const auto& element_i = cache.getElement(body_i);
      int v_start_i = body_i.velocity_num_start;
      int nv_i = body_i.getJoint().getNumVelocities();
      auto F =
          (element_i.crb_in_world * element_i.motion_subspace_in_world).eval();

      // Hii
      ret.block(v_start_i, v_start_i, nv_i, nv_i).noalias() =
          (element_i.motion_subspace_in_world.transpose() * F).eval();

      // Hij
      shared_ptr<RigidBody> body_j(body_i.parent);
      while (body_j->hasParent()) {
        const auto& element_j = cache.getElement(*body_j);
        int v_start_j = body_j->velocity_num_start;
        int nv_j = body_j->getJoint().getNumVelocities();
        auto Hji = (element_j.motion_subspace_in_world.transpose() * F).eval();
        ret.block(v_start_j, v_start_i, nv_j, nv_i) = Hji;
        ret.block(v_start_i, v_start_j, nv_i, nv_j) = Hji.transpose();

        body_j = body_j->parent;
      }
    }
  }

  return ret;
}

template DRAKERBM_EXPORT MatrixX<LargeADSV>
RigidBodyTree::massMatrix<LargeADSV>(
    KinematicsCache<LargeADSV>&) const;

template DRAKERBM_EXPORT MatrixX<DynamicADSV>
RigidBodyTree::massMatrix<DynamicADSV>(
    KinematicsCache<DynamicADSV>&) const;

template DRAKERBM_EXPORT MatrixXd
RigidBodyTree::massMatrix<double>(KinematicsCache<double>&) const;
