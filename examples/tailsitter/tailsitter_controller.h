
#include <Eigen/Core>

#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/affine_system.h"

namespace drake {
namespace examples {
namespace tailsitter {

std::unique_ptr<systems::AffineSystem<double>> StabilizingLQRController(
    const systems::LeafSystem<double>* tailsitter_plant,
    Eigen::Vector3d nominal_position) {
  auto tailsitter_context_goal = tailsitter_plant->CreateDefaultContext();

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(12);
  x0.topRows(3) = nominal_position;
  tailsitter_context_goal->SetContinuousState(x0);

  // Nominal input corresponds to a hover.
  TailsitterParameters params;

  const double nominal_thrust_command =
      (params.mass() * params.world().gravity() / 2) /
      params.propeller().thrust_ratio();
  const double nominal_elevon_deflection = 0.;

  const Vector4<double> u0(nominal_thrust_command, nominal_thrust_command,
                           nominal_elevon_deflection,
                           nominal_elevon_deflection);

  tailsitter_context_goal->get_input_port(0).FixValue(tailsitter_context_goal.get(), u0);

  // Setup LQR cost matrices (penalize position error 10x more than velocity
  // error).
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(12, 12);
  Q.topLeftCorner<6, 6>() = 10 * Eigen::MatrixXd::Identity(6, 6);

  Eigen::Matrix4d R = Eigen::Matrix4d::Identity();

  return systems::controllers::LinearQuadraticRegulator(
      *tailsitter_plant, *tailsitter_context_goal, Q, R);
}

}  // namespace tailsitter

}  // namespace examples
}  // namespace drake
