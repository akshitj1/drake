#pragma once

#include "limits"

#include "drake/common/trajectories/piecewise_polynomial.h"

namespace drake {
namespace examples {
namespace tailsitter {
typedef trajectories::PiecewisePolynomial<double> PPoly;
static const double kEps = std::numeric_limits<double>::epsilon();
static const double kInf = std::numeric_limits<double>::infinity() / 2;

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake