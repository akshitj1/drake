#include "gflags/gflags.h"

#include "drake/examples/tailsitter/make_tailsitter_plant.h"

namespace drake {
namespace examples {
namespace tailsitter {

void build_symbolic_tailsitter() {
  systems::DiagramBuilder<double> builder;

  geometry::SceneGraph<double>& scene_graph = *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  TailsitterPlantBuilder ts_builder;
  const MultibodyPlant<double>& tailsitter =
      *ts_builder.build(builder, &scene_graph);
  auto ts_sym = MultibodyPlant<double>::ToSymbolic(tailsitter);
  return;
}
}  // namespace tailsitter
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Symbolic tailsitter dynamics");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::set_log_level("info");
  drake::examples::tailsitter::build_symbolic_tailsitter();
  return 0;
}