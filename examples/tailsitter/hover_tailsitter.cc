#include "gflags/gflags.h"

#include "drake/examples/tailsitter/tailsitter_plant.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/affine_system.h"
#include "drake/multibody/tree/fixed_offset_frame.h"


namespace drake {
namespace examples {
namespace tailsitter {
using geometry::SceneGraph;
using multibody::RigidBody;
using multibody::SpatialVelocity;
using systems::AffineSystem;

unique_ptr<Context<double>> TailsitterHoverContext(
    const TailsitterPlant<double>* tailsitter) {
  TailsitterParameters tailsitter_params;
  auto tailsitter_hover_context = tailsitter->CreateDefaultContext();
  auto tailsitter_plant = tailsitter->get_multibody_plant();

  // vetically hanging hover pose, we set the pose of wing and every other
  // connected body will follow
  const auto& wing = tailsitter_plant->GetRigidBodyByName("wing");

  const Vector3<double> pos_wing_initial(0, 0, 1);
  const auto rot_initial = math::RotationMatrix<double>::MakeXRotation(0);
  const math::RigidTransform<double> pose_wing_initial(rot_initial,
                                                       pos_wing_initial);
  auto tailsitter_multibody_context = &tailsitter->GetMutableSubsystemContext(
      *tailsitter_plant, tailsitter_hover_context.get());
  tailsitter_plant->SetFreeBodyPoseInWorldFrame(tailsitter_multibody_context,
                                                wing, pose_wing_initial);
  // actuator inputs at hover
  const double nominal_elevon_deflection = 0.0;
  const double nominal_thrust_command =
      (tailsitter_params.mass() * tailsitter_params.world().gravity() / 2) /
      tailsitter_params.propeller().thrust_ratio();
  const Vector4<double> nominal_actuators_command(
      nominal_thrust_command, nominal_thrust_command, nominal_elevon_deflection,
      nominal_elevon_deflection);
  tailsitter->get_input_port().FixValue<Vector4<double>>(
      tailsitter_hover_context.get(), nominal_actuators_command);
  return tailsitter_hover_context;
}

std::unique_ptr<AffineSystem<double>> HoverLQRController(
    const TailsitterPlant<double>* tailsitter) {
  auto tailsitter_hover_context = TailsitterHoverContext(tailsitter);
  // 12 states: (x,y,z),(roll, pitch, yaw), (x_dot,y_dot,z_dot),(roll_dot,
  // pitch_dot, yaw_dot)
  const int kNumStates = 12;
  const MatrixX<double> Q{
      (VectorX<double>(kNumStates) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
          .finished()
          .asDiagonal()};

  // inputs: prop_(l/r), elevon_(l/r)
  const int kNumInputs = 4;
  const MatrixX<double> R{
      (VectorX<double>(kNumInputs) << 1, 1, 1, 1).finished().asDiagonal()};

  return systems::controllers::LinearQuadraticRegulator(
      *tailsitter, *tailsitter_hover_context, Q, R);
}

void simulate_hover() {
  systems::DiagramBuilder<double> builder;

  const auto tailsitter = builder.AddSystem<TailsitterPlant<double>>();
  const auto controller = HoverLQRController(tailsitter);

  builder.Connect(tailsitter->get_output_port(),
                  controller->get_input_port());
  builder.Connect(controller->get_output_port(),
                  tailsitter->get_input_port());

  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& tailsitter_context =
      diagram->GetMutableSubsystemContext(*tailsitter, diagram_context.get());

  const auto& wing = tailsitter->get_multibody_plant()->GetRigidBodyByName("wing");
  const math::RigidTransform<double> pose_wing_initial(
      Vector3<double>(0, 0, 1));
  tailsitter->get_multibody_plant()->SetFreeBodyPoseInWorldFrame(&tailsitter_context, wing,
                                          pose_wing_initial);

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  simulator.set_target_realtime_rate(1.0);
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