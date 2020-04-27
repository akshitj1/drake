#include "drake/examples/tailsitter/common.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"

namespace drake {

namespace examples {
namespace tailsitter {
template <typename T>
class TailsitterController : public systems::LeafSystem<T> {
  const PPoly& u_opt;

 public:
  TailsitterController(const PPoly& u_opt) : u_opt(u_opt) {
    this->DeclareVectorInputPort("tailsitter_state", TailsitterState<T>());
    this->DeclareVectorOutputPort("control_inputs", TailsitterInput<T>(),
                                  &TailsitterController::CalcElevonDeflection);
  }
  void CalcElevonDeflection(const systems::Context<T>& context,
                            TailsitterInput<T>* control) const {
    const double& t = context.get_time();
    control->SetFromVector(u_opt.value(t));
  }
};

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake