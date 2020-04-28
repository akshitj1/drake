#pragma once

#include "drake/examples/tailsitter/common.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"
#include "drake/systems/controllers/finite_horizon_linear_quadratic_regulator.h"

namespace drake {

namespace examples {
namespace tailsitter {
class TailsitterController : public systems::LeafSystem<double> {
  const PPoly &input_nominal, state_nominal;
  const systems::controllers::FiniteHorizonLinearQuadraticRegulatorResult& lqr;

 public:
  TailsitterController(
      const PPoly& state_nominal_, const PPoly& input_nominal_,
      const systems::controllers::FiniteHorizonLinearQuadraticRegulatorResult&
          lqr_)
      : state_nominal(state_nominal_),
        input_nominal(input_nominal_),
        lqr(lqr_) {
    this->DeclareVectorInputPort("tailsitter_state", TailsitterState<double>());
    this->DeclareVectorOutputPort("control_inputs", TailsitterInput<double>(),
                                  &TailsitterController::CalcCorrectiveInputs,
                                  {this->all_state_ticket()});
  }
  void CalcCorrectiveInputs(const systems::Context<double>& context,
                            TailsitterInput<double>* control) const {
    const double& t = context.get_time();
    const Vector<double, kNumStates> state_des = state_nominal.value(t);
    const Vector<double, kNumStates> state_est =
        this->get_input_port(0).Eval(context);

    const Vector<double, kNumStates> state_err = state_est - state_des;
    Vector<double, kNumInputs> corrective_input = -lqr.K.value(t) * state_err;
    Vector<double, kNumInputs> input_des = input_nominal.value(t);
    control->SetFromVector(input_des + corrective_input);
  }
};

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake