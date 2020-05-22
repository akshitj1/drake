#include <iomanip>

#include "gflags/gflags.h"

#include "drake/common/is_approx_equal_abstol.h"
#include "drake/examples/tailsitter/common.h"
#include "drake/examples/tailsitter/fixed_state_roa.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
namespace examples {
namespace tailsitter {
namespace {
using namespace tailsitter;
using systems::Context;
using systems::DiagramBuilder;
using systems::Simulator;
using systems::analysis::FixedStateROA;
using systems::controllers::LinearQuadraticRegulator;
using systems::controllers::LinearQuadraticRegulatorResult;

void get_roa(const Tailsitter<double>& tailsitter,
             const TailsitterState<double>& x0,
             const TailsitterInput<double>& u0, const MatrixX<double>& Q,
             const MatrixX<double>& R) {
  auto hover_context = tailsitter.CreateDefaultContext();
  hover_context->SetContinuousState(x0.CopyToVector());
  hover_context->FixInputPort(0, u0);

  auto f_lin = FirstOrderTaylorApproximation(
      tailsitter, *hover_context,
      systems::InputPortSelection::kUseFirstInputIfItExists,
      systems::OutputPortSelection::kNoOutput);

  LinearQuadraticRegulatorResult lqr_res =
      LinearQuadraticRegulator(f_lin->A(), f_lin->B(), Q, R);
  FixedStateROA(tailsitter, x0.CopyToVector(), u0.CopyToVector(), lqr_res);
}

void simulate_hover() {
  DiagramBuilder<double> builder;
  auto tailsitter = builder.AddSystem<Tailsitter<double>>();
  tailsitter->set_name("tailsitter");

  TailsitterState<double> x0;
  x0.set_theta(M_PI / 2);
  TailsitterInput<double> u0;
  u0.set_prop_throttle((tailsitter->kMass * tailsitter->kG) /
                       tailsitter->kThrustMax);

  const MatrixX<double> Q{
      (VectorX<double>(kNumStates) << 10, 10, 10, 1, 1, 1, 1)
          .finished()
          .asDiagonal()};
  const MatrixX<double> R{
      (VectorX<double>(kNumInputs) << 1, 1).finished().asDiagonal()};

  // get_roa(*tailsitter, x0, u0, Q, R);
  auto hover_context = tailsitter->CreateDefaultContext();

  //   tailsitter->get_input_port(0).FixValue(hover_context.get(),
  //                                          u0.CopyToVector());
  hover_context->SetContinuousState(x0.CopyToVector());
  hover_context->FixInputPort(0, u0);

  auto controller = builder.AddSystem(
      LinearQuadraticRegulator(*tailsitter, *hover_context, Q, R));
  controller->set_name("controller");
  builder.Connect(tailsitter->get_output_port(0), controller->get_input_port());
  builder.Connect(controller->get_output_port(), tailsitter->get_input_port(0));

  auto diagram = builder.Build();
  Simulator<double> simulator(*diagram);

  TailsitterState<double> x_initial = x0;
  x_initial.set_theta(x0.theta() + 0.1);

  simulator.get_mutable_context()
      .get_mutable_continuous_state_vector()
      .SetFromVector(x_initial.CopyToVector());

  simulator.Initialize();
  simulator.set_monitor([x0](const systems::Context<double>& root_context) {
    if (is_approx_equal_abstol(
            root_context.get_continuous_state().CopyToVector(),
            x0.CopyToVector(), 1e-4)) {
      return systems::EventStatus::ReachedTermination(nullptr,
                                                      "Goal achieved.");
    }
    return systems::EventStatus::Succeeded();
  });
  simulator.set_target_realtime_rate(0.0);

  // The following accuracy is necessary for the example to satisfy its
  // ending state tolerances.
  simulator.get_mutable_integrator().set_target_accuracy(5e-5);
  auto status = simulator.AdvanceTo(10.0);
  if (status.kReachedTerminationCondition) {
    log()->info(fmt::format("achieved desired state in {:.2f} s",
                            status.return_time()));
  } else {
    // Goal state verification.
    const VectorX<double>& x_final =
        simulator.get_context().get_continuous_state().CopyToVector();
    std::cout << "deviation: " << std::fixed << std::setprecision(4)
              << x0.CopyToVector() - x_final << std::endl;
    throw std::runtime_error("Target state is not achieved.");
  }
}

}  // namespace
}  // namespace tailsitter
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Hover control of Tailsitter.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::set_log_level("info");
  drake::examples::tailsitter::simulate_hover();
  return 0;
}