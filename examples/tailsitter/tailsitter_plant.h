#pragma once
#include <cmath>
#include <memory>

#include <Eigen/Core>

#include "drake/common/default_scalars.h"
#include "drake/examples/tailsitter/gen/tailsitter_input.h"
#include "drake/examples/tailsitter/gen/tailsitter_state.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace tailsitter {

template <typename T>
class Tailsitter final : public systems::LeafSystem<T> {
 private:
  // taken from
  // https://github.com/RobotLocomotion/drake/blob/BeforeCCode/examples/Glider/GliderPlant.m
  const double kG = 9.81, kRho = 1.204;  // atm density
  const double kMass = 0.082, kInertia = 0.0015;
  const double kTailS = 0.0147, kWingS = 0.0885;
  const double kLe = 0.022, kL = 0.27;
  const double kLw = 0;

 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Tailsitter);

  Tailsitter() : systems::LeafSystem<T>(systems::SystemTypeTag<Tailsitter>{}) {
    // only one controllable input that is tail deflection
    this->DeclareVectorInputPort("tail_defl_rate", TailsitterInput<T>());
    // 3 pos -> x, y, theta ; 3 vel -> x_dot, y_dot, theta_dot ; phi
    this->DeclareContinuousState(TailsitterState<T>(), 3, 3, 1);
    this->DeclareVectorOutputPort("tailsitter_state", &Tailsitter::CopyStateOut,
                                  {this->all_state_ticket()});
  }

  template <typename U>
  explicit Tailsitter(const Tailsitter<U>&) : Tailsitter<T>() {}

  void CopyStateOut(const drake::systems::Context<T>& context,
                    TailsitterState<T>* output) const {
    *output = get_state(context);
  }

  static const TailsitterState<T>& get_state(const systems::Context<T>& ctx) {
    return dynamic_cast<const TailsitterState<T>&>(
        ctx.get_continuous_state().get_vector());
  }

  static TailsitterState<T>& get_mutable_state(
      systems::ContinuousState<T>* derivatives) {
    return dynamic_cast<TailsitterState<T>&>(derivatives->get_mutable_vector());
  }

  void DoCalcTimeDerivatives(
      const systems::Context<T>& context,
      systems::ContinuousState<T>* derivatives) const override {
    const TailsitterState<T>& q = get_state(context);

    // elevon defl. rate
    auto phi_dot = this->get_input_port(0).Eval(context)(0);

    auto xw_dot = q.x_dot() - kLw * q.theta_dot() * sin(q.theta());
    auto zw_dot = q.z_dot() + kLw * q.theta_dot() * cos(q.theta());
    auto alpha_w = q.theta() - atan2(zw_dot, xw_dot);
    auto F_w = kRho * kWingS * sin(alpha_w) * (pow(zw_dot, 2) + pow(xw_dot, 2));

    auto xe_dot = q.x_dot() + kL * q.theta_dot() * sin(q.theta()) +
                  kLe * (q.theta_dot() + phi_dot) * sin(q.theta() + q.phi());
    auto ze_dot = q.z_dot() - kL * q.theta_dot() * cos(q.theta()) -
                  kLe * (q.theta_dot() + phi_dot) * cos(q.theta() + q.phi());
    auto alpha_e = q.theta() + q.phi() - atan2(ze_dot, xe_dot);
    auto F_e = kRho * kTailS * sin(alpha_e) * (pow(ze_dot, 2) + pow(xe_dot, 2));

    auto x_ddot =
        -(F_w * sin(q.theta()) + F_e * sin(q.theta() + q.phi())) / kMass;
    auto z_ddot =
        (F_w * cos(q.theta()) + F_e * cos(q.theta() + q.phi())) / kMass - kG;
    auto theta_ddot = (F_w * kLw - F_e * (kL * cos(q.phi()) + kLe)) / kInertia;

    TailsitterState<T>& q_dot = get_mutable_state(derivatives);
    q_dot.set_x(q.x_dot());
    q_dot.set_z(q.z_dot());
    q_dot.set_theta(q.theta_dot());
    q_dot.set_phi(phi_dot);
    q_dot.set_x_dot(x_ddot);
    q_dot.set_z_dot(z_ddot);
    q_dot.set_theta_dot(theta_ddot);
  }

  T _liftCoeff(T alpha) const { return sin(2 * alpha); }
  T _dragCoeff(T alpha) const { return 2 * pow(sin(alpha), 2); }

  Vector2<T> _calcSurfaceForce(double surface_area, Vector2<T> wind_vel,
                               T inclination_intertial) const {
    Vector2<T> surface_dir(-sin(inclination_intertial),
                           cos(inclination_intertial));

    T alpha = inclination_intertial - atan2(wind_vel[1], wind_vel[0]);
    Vector2<T> aero_force =
        (1 / 2.0 * kRho * wind_vel.squaredNorm() * surface_area *
         (_liftCoeff(alpha) + _dragCoeff(alpha))) *
        surface_dir;
    return aero_force;
  }

  Vector2<T> _calcSurfaceVel(Vector2<T> com_vel, double com_dist, T theta,
                             T thetadot) const {
    return Vector2<T>(com_vel[0] + com_dist * thetadot * sin(theta),
                      com_vel[1] - com_dist * thetadot * cos(theta));
  }
};

}  // namespace tailsitter
}  // namespace examples

// The following code was added to prevent scalar conversion to symbolic scalar
// types. The QuadrotorPlant makes use of classes that are not compatible with
// the symbolic scalar. This NonSymbolicTraits is explained in
// drake/systems/framework/system_scalar_converter.h.
namespace systems {
namespace scalar_conversion {
template <>
struct Traits<examples::tailsitter::Tailsitter> : public NonSymbolicTraits {};
}  // namespace scalar_conversion
}  // namespace systems
}  // namespace drake