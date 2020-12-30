#include "gflags/gflags.h"

#include "drake/examples/tailsitter/make_tailsitter_plant.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/multibody/tree/joint_actuator.h"
// #include "drake/multibody/tree/multibody_tree_indexes.h"

namespace drake {
namespace examples {
namespace tailsitter {
using geometry::SceneGraph;
using multibody::RigidBody;
using multibody::SpatialVelocity;

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

  geometry::DrakeVisualizer::AddToBuilder(&builder, scene_graph);
  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& tailsitter_context =
      diagram->GetMutableSubsystemContext(tailsitter, diagram_context.get());


  tailsitter.get_actuation_input_port().FixValue(&tailsitter_context, Vector2<double>{10.*M_PI/6,10.*M_PI/6});


  // const auto& left_hinge = tailsitter.GetJointByName<RevoluteJoint>("elevon_left_joint");
  // left_hinge.set_angle(&tailsitter_context, M_PI_4);

  const RigidBody<double>& wing = tailsitter.GetRigidBodyByName("wing");
  
  const Vector3<double> pos_wing_initial(0, 0, 1);
  const auto rot_initial =
      math::RotationMatrix<double>::MakeXRotation(0);  //-M_PI_4);
  const math::RigidTransform<double> pose_wing_initial(rot_initial,
                                                       pos_wing_initial);
  tailsitter.SetFreeBodyPoseInWorldFrame(&tailsitter_context, wing,
                                         pose_wing_initial);

  //   tailsitter.SetFreeBodySpatialVelocity(
  //       &tailsitter_context, wing,
  //       SpatialVelocity<double>(Vector3<double>::Zero(),
  //                               5 * Vector3<double>::UnitY()));

  TailsitterParameters params;
  const double nominal_thrust_command =
      (params.tailsitter_mass() * params.gravity() / 2) /
      params.propeller_thrust_ratio();
  diagram->get_input_port(0).FixValue<Vector2<double>>(
      diagram_context.get(), nominal_thrust_command * Vector2<double>::Ones());

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  simulator.set_target_realtime_rate(0.0);

  simulator.AdvanceTo(10.0);
}
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