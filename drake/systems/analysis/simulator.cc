#include "drake/systems/analysis/simulator.h"

#include "drake/common/eigen_autodiff_types.h"
#include "drake/math/autodiff_overloads.h"

namespace drake {
namespace systems {

template class Simulator<double>;
template class Simulator<AutoDiffXd>;

}  // namespace systems
}  // namespace drake
