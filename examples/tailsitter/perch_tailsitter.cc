#include "gflags/gflags.h"

#include "drake/examples/tailsitter/common.h"
#include "drake/examples/tailsitter/tailsitter_controller.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"
#include "drake/examples/tailsitter/trajectory_funnel.h"
#include "drake/examples/tailsitter/trajectory_generator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/finite_horizon_linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace tailsitter {
namespace {

/*
 * tutorial on unique_ptr:
 * https://thispointer.com/c11-unique_ptr-tutorial-and-examples/
 */
void simulate_trajectory(const tailsitter::Trajectory& trajectory) {
  systems::DiagramBuilder<double> builder;
  auto tailsitter = builder.AddSystem<Tailsitter>();
  PPoly state_opt, input_opt;
  tailsitter::TrajectoryGenerator trajectory_generator(*tailsitter);
  trajectory_generator.get_trajectory(trajectory, state_opt, input_opt);

  const VectorX<double> kXf_err_max{
      (VectorX<double>(kNumStates) << 0.05, 0.05, 3, 3, 1, 1, 3).finished()};
  const MatrixX<double> Qf{
      kXf_err_max.array().square().inverse().matrix().asDiagonal()};
  const MatrixX<double> Q{
      (VectorX<double>(kNumStates) << 10, 10, 10, 1, 1, 1, 1)
          .finished()
          .asDiagonal()};
  const MatrixX<double> R{
      (VectorX<double>(kNumInputs) << 0.1, 0.1).finished().asDiagonal()};

  auto lqr_context = tailsitter->CreateDefaultContext();

  systems::controllers::FiniteHorizonLinearQuadraticRegulatorOptions
      lqr_options;
  lqr_options.Qf = Qf;
  lqr_options.u0 = &input_opt;
  lqr_options.x0 = &state_opt;

  auto lqr_res = systems::controllers::FiniteHorizonLinearQuadraticRegulator(
      *tailsitter, *lqr_context, state_opt.start_time(), state_opt.end_time(),
      Q, R, lqr_options);

  log()->info("lqr S, K matrices computed.");

  log()->info("computing trajectory funnel");
  systems::analysis::TrajectoryFunnel(*tailsitter, state_opt, input_opt,
                                      lqr_res);
  log()->info("Funnel computed");

  auto ts_controller =
      builder.AddSystem<TailsitterController>(state_opt, input_opt, lqr_res);
  builder.Connect(tailsitter->get_output_port(0),
                  ts_controller->get_input_port(0));
  builder.Connect(ts_controller->get_output_port(0),
                  tailsitter->get_input_port(0));

  auto diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);

  TailsitterState<double> init_state_err;
  init_state_err.set_x(0.1);

  simulator.get_mutable_context()
      .get_mutable_continuous_state_vector()
      .SetFromVector(state_opt.value(0) + init_state_err.CopyToVector());

  simulator.set_target_realtime_rate(0);
  simulator.AdvanceTo(state_opt.end_time());

  TailsitterState<double> final_state_est;
  final_state_est.SetFromVector(
      simulator.get_context().get_continuous_state_vector().CopyToVector());

  drake::log()->info(fmt::format(
      "final state:\n x: {:.3f} m\ty: {:.3f} m\tangle: {:.0f} degree",
      final_state_est.x(), final_state_est.z(),
      180 * final_state_est.theta() / M_PI));
}
}  // namespace

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Trajectory optimization for perching tailsitter.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::set_log_level("info");
  drake::examples::tailsitter::simulate_trajectory(
      drake::examples::tailsitter::Climb);
  return 0;
}