#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace multibody {
template <typename T>
class ExternalSpatialForceMultiplexer final : public systems::LeafSystem<T> {
 private:
  const std::vector<unsigned int> input_sizes_;
  int output_size() const {
    return std::accumulate(input_sizes_.begin(), input_sizes_.end(), 0,
                           std::plus<int>{});
  }

  // This is the calculator for the output port.
  void ConcatenateForceInputs(
      const systems::Context<T>& context,
      std::vector<ExternallyAppliedSpatialForce<T>>* spatial_forces) const {
    spatial_forces->resize(output_size());

    int spatial_forces_idx{0};
    for (int i = 0; i < this->num_input_ports(); ++i) {
      const auto& port_spatial_forces =
          this->get_input_port(i)
              .template Eval<std::vector<ExternallyAppliedSpatialForce<T>>>(
                  context);
      assert(port_spatial_forces.size() == input_sizes_[i]);
      for (auto port_spatial_force : port_spatial_forces) {
        spatial_forces->at(spatial_forces_idx) = port_spatial_force;
        spatial_forces_idx++;
      }
    }
  }

  // Declare friendship to enable scalar conversion.
  template <typename U>
  friend class ExternalSpatialForceMultiplexer;

 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExternalSpatialForceMultiplexer)

  explicit ExternalSpatialForceMultiplexer(
      std::vector<unsigned int> input_sizes)
      : systems::LeafSystem<T>(systems::SystemTypeTag<ExternalSpatialForceMultiplexer>{}),
       input_sizes_(input_sizes) {
    for (unsigned long port = 0; port < input_sizes_.size(); port++) {
      // const int input_size = input_sizes_[port];
      this->DeclareAbstractInputPort(
          fmt::format("external_spatial_forces_{}", port),
          Value<std::vector<ExternallyAppliedSpatialForce<T>>>());
    }

    this->DeclareAbstractOutputPort(
        "all_external_spatial_forces",
        std::vector<ExternallyAppliedSpatialForce<T>>(output_size()),
        &ExternalSpatialForceMultiplexer<T>::ConcatenateForceInputs);
  }

  template <typename U>
  explicit ExternalSpatialForceMultiplexer(
      const ExternalSpatialForceMultiplexer<U>& other)
      : ExternalSpatialForceMultiplexer<T>(std::vector<unsigned int>(
            other.input_sizes_.begin(), other.input_sizes_.end())) {}

  const systems::OutputPort<T>& get_spatial_forces_output_port() const {
    return this->get_output_port(0);
  }
};
}  // namespace multibody
}  // namespace drake
