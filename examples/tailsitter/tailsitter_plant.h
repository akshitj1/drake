#pragma once
#include <cmath>

#include "drake/common/default_scalars.h"
#include "drake/examples/tailsitter/gen/tailsitter_input.h"
#include "drake/examples/tailsitter/gen/tailsitter_state.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace tailsitter {
const int kNumStates = TailsitterState<double>::K::kNumCoordinates;
const int kNumInputs = TailsitterInput<double>::K::kNumCoordinates;

template <typename T>
class Tailsitter final : public systems::LeafSystem<T> {
 public:
  // taken from
  // https://github.com/RobotLocomotion/drake/blob/BeforeCCode/examples/Glider/GliderPlant.m
  static constexpr double kG = 9.81, kRho = 1.204;  // atm density
  static constexpr double kMass = 0.082, kInertia = 0.0015;
  static constexpr double kTailS = 0.0147, kWingS = 0.0885;
  static constexpr double kLe = 0.022, kL = 0.27, kLw = 0.0;
  static constexpr double kPropDiameter = kL / 2,
                          kPropArea =
                              (M_PI * kPropDiameter * kPropDiameter) / 4,
                          kThrustMax = 1.6 * (kMass * kG);
  static constexpr double kPhiLimitL = -M_PI / 3, kPhiLimitU = M_PI / 3,
                          kPhiDotLimit = 13;

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

    Vector2<T> pos_ddot = (F_p + F_w + F_e + F_g) / kMass;
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
        rotate(Vector2<T>(-sqrt((2 * force_B(0)) / (kRho * kPropArea)), 0),
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
        rotate(cross(parent_theta_dot, joint_pos), parent_theta) +
        rotate(cross(plate_theta_dot + parent_theta_dot,
                     Vector2<T>(-plate_com_dist, 0)),
               plate_theta + parent_theta);

    T attack_angle = parent_theta + plate_theta - get_angle(plate_pos_dot);
    // https://groups.csail.mit.edu/robotics-center/public_papers/Roberts09.pdf
    Vector2<T> force_W =
        Vector2<T>(0, kRho * sin(attack_angle) * plate_surface_area *
                          get_norm_squared(plate_pos_dot));

    // rotation inverse
    force_I = rotate(force_W, parent_theta + plate_theta);
    torque_B =
        cross(joint_pos + rotate(Vector2<T>(-plate_com_dist, 0), plate_theta),
              force_W);
  }

  // b is in k cap
  static Vector2<T> cross(const Vector2<T>& a, const T& b) {
    return Vector2<T>(a(1) * b, -a(0) * b);
  }

  // a is in k cap
  static Vector2<T> cross(const T& a, const Vector2<T>& b) {
    return Vector2<T>(-a * b(1), a * b(0));
  }

  static T cross(const Vector2<T>& a, const Vector2<T>& b) {
    // returns k cap component
    return a(0) * b(1) - a(1) * b(0);
  }

  static Vector2<T> rotate(const Vector2<T>& x, const T& theta) {
    // R(theta).x
    return Vector2<T>(x(0) * cos(theta) - x(1) * sin(theta),
                      x(0) * sin(theta) + x(1) * cos(theta));
  }

  static T get_angle(const Vector2<T>& x) { return atan2(x(1), x(0)); }

  static T get_norm_squared(const Vector2<T>& x) {
    return x(0) * x(0) + x(1) * x(1);
    // todo: why below does not works?
    // return pow(x(0), 2) + pow(x(1), 2);
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