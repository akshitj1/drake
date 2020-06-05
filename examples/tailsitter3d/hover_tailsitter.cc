#include "gflags/gflags.h"

#include "drake/examples/tailsitter3d/make_tailsitter_plant.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"

namespace drake {
namespace examples {
namespace tailsitter3d {
using geometry::SceneGraph;
using multibody::RigidBody;

void simulate_hover() {
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  TailsitterPlantBuilder ts_builder;
  const MultibodyPlant<double>& tailsitter =
      *ts_builder.build(builder, &scene_graph);

  builder.Connect(
      tailsitter.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(tailsitter.get_source_id().value()));

  geometry::ConnectDrakeVisualizer(&builder, scene_graph);
  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& tailsitter_context =
      diagram->GetMutableSubsystemContext(tailsitter, diagram_context.get());

  const RigidBody<double>& wing = tailsitter.GetRigidBodyByName("wing");
  const Vector3<double> pos_wing_initial(0, 0, 10);
  const math::RigidTransform<double> pose_wing_initial(pos_wing_initial);
  tailsitter.SetFreeBodyPoseInWorldFrame(&tailsitter_context, wing,
                                         pose_wing_initial);

  diagram_context->FixInputPort(0, Vector2<double>(1.0, 1.0));

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(0.0);
  simulator.Initialize();
  simulator.AdvanceTo(10.0);
}
}  // namespace tailsitter3d
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Hover control of Tailsitter.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::set_log_level("info");
  drake::examples::tailsitter3d::simulate_hover();
  return 0;
}