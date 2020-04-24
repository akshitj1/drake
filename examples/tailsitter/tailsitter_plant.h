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
  static constexpr double kG = 9.81, kRho = 1.204;  // atm density
  static constexpr double kMass = 0.082, kInertia = 0.0015;
  static constexpr double kTailS = 0.0147, kWingS = 0.0885;
  static constexpr double kLe = 0.022, kL = 0.27, kLw = 0.0;
  static constexpr double kPropDiameter = kL / 2,
                          kThrustMax = 1.6 * (kMass * kG);

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
    TailsitterInput<T> input;
    input.SetFromVector(this->get_input_port(0).Eval(context));

    Vector2<T> F_w, F_e, F_p;
    const Vector2<T> F_g(0, -kMass * kG);
    T T_w, T_e;

    Vector2<T> prop_downwash;
    get_propeller_force_and_downwash(input.prop_throttle(), q.theta(), F_p,
                                     prop_downwash);

    get_plate_forces_and_torques(
        Vector2<T>(q.x_dot(), q.z_dot()) - prop_downwash, q.theta(),
        q.theta_dot(), Vector2<T>(-kLw, 0), kWingS, 0, 0, 0, F_w, T_w);
    get_plate_forces_and_torques(
        Vector2<T>(q.x_dot(), q.z_dot()) - prop_downwash, q.theta(),
        q.theta_dot(), Vector2<T>(-kL, 0), kTailS, kLe, q.phi(),
        input.phi_dot(), F_e, T_e);

    Vector2<T> pos_ddot = (F_p, F_w + F_e + F_g) / kMass;
    T theta_ddot = (T_w + T_e) / kInertia;

    TailsitterState<T>& q_dot = get_mutable_state(derivatives);
    q_dot.set_x(q.x_dot());
    q_dot.set_z(q.z_dot());
    q_dot.set_theta(q.theta_dot());
    q_dot.set_phi(input.phi_dot());
    q_dot.set_x_dot(pos_ddot(0));
    q_dot.set_z_dot(pos_ddot(1));
    q_dot.set_theta_dot(theta_ddot);
  }

  static void get_propeller_force_and_downwash(const T& throttle,
                                               const T& parent_theta,
                                               Vector2<T>& force_I,
                                               Vector2<T>& downwash_vel_I) {
    Vector2<T> force_B = Vector2<T>(throttle * kThrustMax, 0);
    downwash_vel_I =
        rotate(Vector2<T>(-sqrt((2 * force_B(0)) / (kRho * kPropDiameter)), 0),
               parent_theta);
    force_I = rotate(force_B, parent_theta);
  }

  static void get_plate_forces_and_torques(
      const Vector2<T>& parent_pos_dot, const T& parent_theta,
      const T& parent_theta_dot,
      // w.r.t parent
      const Vector2<T>& joint_pos, const T& plate_surface_area,
      // w.r.t joint polar coordinates
      const T& plate_com_dist, const T& plate_theta, const T& plate_theta_dot,
      // to be computed by this fn.
      Vector2<T>& force_I, T& torque_B) {
    Vector2<T> plate_pos_dot =
        parent_pos_dot +
        rotate(cross(joint_pos, parent_theta_dot), parent_theta) +
        rotate(cross(Vector2<T>(-plate_com_dist, 0),
                     plate_theta_dot + parent_theta_dot),
               plate_theta + parent_theta);

    T attack_angle = parent_theta + plate_theta - get_angle(plate_pos_dot);
    Vector2<T> force_W =
        Vector2<T>(0, kRho * sin(attack_angle) * plate_surface_area *
                          get_norm_squared(plate_pos_dot));

    force_I = rotate(force_W, parent_theta + plate_theta);
    torque_B =
        cross(joint_pos + rotate(Vector2<T>(plate_com_dist, 0), plate_theta),
              force_W);
  }

  // y is in k cap
  static Vector2<T> cross(const Vector2<T>& a, const T& b) {
    return Vector2<T>(-a(1) * b, a(0) * b);
  }

  static T cross(const Vector2<T>& a, const Vector2<T>& b) {
    // returns y cap component
    return -a(0) * b(1) + a(1) * b(0);
  }

  static Vector2<T> rotate(const Vector2<T>& x, const T& theta) {
    // R(theta).x
    return Vector2<T>(x(0) * cos(theta) - x(1) * sin(theta),
                      x(0) * sin(theta) + x(1) * cos(theta));
  }

  static T get_angle(const Vector2<T>& x) { return atan2(x(1), x(0)); }

  static T get_norm_squared(const Vector2<T>& x) {
    return pow(x(0), 2) + pow(x(1), 2);
  }
};

/*
template <typename T>
class Body {
  BodyState state;
  virtual void get_dyanmics(T& x_ddot, T& z_ddot, T& theta_ddot);
};

template <typename T>
class BodyState {
 public:
  T x, z, theta, x_dot, z_dot, theta_dot;
};

template <typename T>
class TailsitterBody: Body {
  Link wing;
  Link elevon;
  const T kTailJointL;
  T kMass;
  T kIntertia;
  const double kG = 9.81;
  void get_dymamics(T& x_ddot, T& z_ddot, T& theta_ddot) {
    // newtonian
    Vector2<T> F_w_I = rotate(wing.get_force(), theta);
    Vector2<T> F_e_I = rotate(elevon.get_force(), theta + phi);
    Vector2<T> F_g_I = Vector2<T>(0, -kMass * kG);
    Vector2<T> pos_ddot = (F_w_I + F_e_I + F_g_I) / kMass;
    x_ddot = pos_ddot(0);
    z_ddot = pos_ddot(1);

    // euler
    T T_w = Vector2(wing.radial_distance, 0).cross(F_w_I);
    T T_e =
        Vector2(-kTailJointL, 0) +
        rotate(Vector2<T>(-tail.radial_distance, 0), tail.theta).cross(F_e_I);
    theta_ddot = (T_w + T_e) / kIntertia;
  }
};

template <typename T>
class Joint {
 private:
  const Flatplate<T> plate;
  const Body<T> parent;

 public:
  // w.r.t to parent body
  T radial_distance;
  Vector2<T> theta, theta_dot;

  Vector2<T> get_force(const Vector2<T>& parent_vel,
                       const T parent_angular_vel) {
    Vector2<T> plate_vel =  // self vel only has angular component
        parent_vel + rotate(radial_distance * parent_angular_vel +
                                radial_distance * theta_dot,
                            M_PI / 2 + theta);
    return plate.get_aero_force(plate_vel);
  }
};

template <typename T>
class FlatPlate: Body {
 private:
  const double kAtmDensity = 1.204, surface_area;

 public:
  FlatPlate<T>(const double& surface_area_) : surface_area(surface_area_) {}

 public:
  Vector2<T> get_aero_force(const Vector2<T>& wind_vel) {
    T attack_angle = atan2(wind_vel(1), wind_vel(0));
    // matlab impelementation ignores (sin + cos) term??
    T aero_normal_force = 1.0 / 2 * dynamics_pressure() * surface_area *
                          (lift_coeff(attack_angle) + drag_coeff(attack_angle));
    // no skin friction considered
    return Vector2<T>(0, aero_normal_force);
  }

 private:
  T dynamics_pressure(const Vector2<T>& wind_vel) {
    T wind_speed_2 = pow(wind_vel(0), 2) + pow(wind_vel(1), 2);
    return 1 / 2.0 * kAtmDensity * wind_speed_2;
  }

  // flat plate theory
  T lift_coeff(T angle) const { return 2 * sin(alpha) * cos(alpha); }
  T drag_coeff(T angle) const { return 2 * pow(sin(angle), 2); }
};
*/
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