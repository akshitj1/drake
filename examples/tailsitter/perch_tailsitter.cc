#include "gflags/gflags.h"

#include "drake/examples/tailsitter/common.h"
#include "drake/examples/tailsitter/tailsitter_controller.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"
#include "drake/examples/tailsitter/trajectory_generator.h"
#include "drake/systems/analysis/simulator.h"
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

  auto ts_controller =
      builder.AddSystem<TailsitterController<double>>(input_opt);
  builder.Connect(tailsitter->get_output_port(0),
                  ts_controller->get_input_port(0));
  builder.Connect(ts_controller->get_output_port(0),
                  tailsitter->get_input_port(0));

  auto diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);

  simulator.set_target_realtime_rate(0);
  simulator.AdvanceTo(state_opt.end_time());
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
      drake::examples::tailsitter::Transistion);
  return 0;
}