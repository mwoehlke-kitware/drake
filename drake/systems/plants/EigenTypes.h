#ifndef DRAKE_EIGENTYPES_H
#define DRAKE_EIGENTYPES_H

#include "KinematicsCache.h"

#include "drake/util/drakeGeometryUtil.h"

#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/StdVector>

using LargeVector = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 73, 1>;
using LargeADSV = Eigen::AutoDiffScalar<LargeVector>;
using DynamicADSV = Eigen::AutoDiffScalar<Eigen::VectorXd>;

template <typename Scalar>
using TwistVector = Eigen::Matrix<Scalar, TWIST_SIZE, 1>;

template <typename Scalar>
using TwistMatrix = Eigen::Matrix<Scalar, TWIST_SIZE, Eigen::Dynamic>;

template <typename Scalar>
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using Matrix3X = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;

template <typename Scalar>
using Matrix4X = Eigen::Matrix<Scalar, 4, Eigen::Dynamic>;

template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

#endif
