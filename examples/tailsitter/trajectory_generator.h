#pragma once
#include <iomanip>

#include "drake/examples/tailsitter/common.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"
#include "drake/examples/tailsitter/trajectory_optimizer.h"

namespace drake {
namespace examples {
namespace tailsitter {
static enum Trajectory { Perch, Climb, Hover, Transistion };

class TrajectoryGenerator {
  TailsitterState<double> takeoff_state, land_state, land_tol, state_l, state_u;
  TailsitterInput<double> input_cost;
  TailsitterState<double> final_state_cost;
  TailsitterInput<double> input_l, input_u;
  const Tailsitter<double>& tailsitter;

 public:
  TrajectoryGenerator(const Tailsitter<double>& _tailsitter)
      : tailsitter(_tailsitter) {}

  void get_trajectory(const Trajectory& trajectory, PPoly& state_opt,
                      PPoly& input_opt) {
    switch (trajectory) {
      case Perch:
        set_perching_constraints();
        break;
      case Climb:
        set_climb_constraints();
        break;
      case Transistion:
        set_transistion_constraints();
        break;
      default:
        log()->error(fmt::format("{} trajectory not implemented", trajectory));
        throw;
    }

    input_l.set_phi_dot(-Tailsitter<double>::kPhiDotLimit);
    input_u.set_phi_dot(Tailsitter<double>::kPhiDotLimit);
    input_l.set_prop_throttle(0);
    input_u.set_prop_throttle(1);

    systems::trajectory_optimization::TrajectoryOptimizer traj_optimizer(
        tailsitter);

    traj_optimizer.set_initial_state(takeoff_state.CopyToVector());
    traj_optimizer.set_final_state(land_state.CopyToVector(),
                                   land_tol.CopyToVector());
    traj_optimizer.bound_state(state_l.CopyToVector(), state_u.CopyToVector());
    traj_optimizer.bound_input(input_l.CopyToVector(), input_u.CopyToVector());

    traj_optimizer.add_running_cost(input_cost.CopyToVector());
    traj_optimizer.add_final_cost(final_state_cost.CopyToVector());

    traj_optimizer.get_optimum_trajectories(state_opt, input_opt);

    log_trajectory(state_opt, input_opt);
  }

 private:
  /*
   * ref:
   * https://github.com/RobotLocomotion/drake/blob/last_sha_with_original_matlab/drake/examples/Glider/runDircolPerching.m
   */
  void set_perching_constraints() {
    takeoff_state.set_x(-3.5);
    takeoff_state.set_z(0.1);
    takeoff_state.set_x_dot(1);

    land_state.set_x(0);
    land_state.set_z(0);
    land_state.set_theta(M_PI / 4);
    land_state.set_z_dot(-0.5);
    land_state.set_theta_dot(-0.5);

    land_tol.set_x(kEps);
    land_tol.set_z(kEps);
    land_tol.set_theta(M_PI / 4);
    land_tol.set_phi(kInf);
    land_tol.set_x_dot(2);
    land_tol.set_z_dot(2);
    land_tol.set_theta_dot(kInf);

    state_l.SetFromVector(VectorX<double>::Constant(kNumStates, -kInf));
    state_l.set_x(-4);
    state_l.set_z(-1);
    state_l.set_theta(-M_PI / 2);
    state_l.set_phi(Tailsitter<double>::kPhiLimitL);

    state_u.SetFromVector(VectorX<double>::Constant(kNumStates, kInf));
    state_u.set_x(1);
    state_u.set_z(1);
    state_u.set_theta(M_PI / 2);
    state_u.set_phi(Tailsitter<double>::kPhiLimitU);

    input_cost.set_phi_dot(100);
    input_cost.set_prop_throttle(100);

    final_state_cost.set_x(10);
    final_state_cost.set_z(10);
    final_state_cost.set_theta(1);
    final_state_cost.set_phi(10);
    final_state_cost.set_x_dot(1);
    final_state_cost.set_z_dot(1);
    final_state_cost.set_theta_dot(1);
  }

  void set_climb_constraints() {
    takeoff_state.set_z(-1.0);
    land_state.set_z(1.0);
    takeoff_state.set_theta(M_PI / 2);
    land_state.set_theta(M_PI / 2);

    land_tol.set_x(0.05);
    land_tol.set_z(0.05);
    land_tol.set_theta(0.1);
    land_tol.set_phi(kInf);
    land_tol.set_x_dot(0.05);
    land_tol.set_z_dot(0.05);
    land_tol.set_theta_dot(0.05);

    state_l.SetFromVector(VectorX<double>::Constant(kNumStates, -kInf));
    state_u.SetFromVector(VectorX<double>::Constant(kNumStates, kInf));
    state_l.set_z(-2);
    state_u.set_z(2);
    state_l.set_x(-2);
    state_u.set_x(2);
    state_l.set_theta(M_PI / 4);
    state_u.set_theta(3 * M_PI / 4);
    state_l.set_phi(Tailsitter<double>::kPhiLimitL);
    state_u.set_phi(Tailsitter<double>::kPhiLimitU);

    input_cost.set_phi_dot(100);
    input_cost.set_prop_throttle(100);

    final_state_cost.set_x(10);
    final_state_cost.set_z(10);
    final_state_cost.set_theta(1);
    final_state_cost.set_phi(10);
    final_state_cost.set_x_dot(1);
    final_state_cost.set_z_dot(1);
    final_state_cost.set_theta_dot(1);
  }

  void set_transistion_constraints() {
    takeoff_state.set_theta(M_PI / 2);

    land_state.set_x(5);
    land_tol.set_x(5);
    land_state.set_z(5);
    land_tol.set_z(5);
    land_state.set_z_dot(0);
    land_tol.set_z_dot(0.001);
    land_state.set_x_dot(6);
    land_tol.set_x_dot(0.05);
    land_state.set_theta(M_PI / 8);
    land_tol.set_theta(M_PI / 8);
    land_state.set_theta_dot(0);
    land_tol.set_theta_dot(0.001);
    land_tol.set_phi(kInf);

    state_l.SetFromVector(VectorX<double>::Constant(kNumStates, -kInf));
    state_u.SetFromVector(VectorX<double>::Constant(kNumStates, kInf));
    state_l.set_z(-1);
    state_u.set_z(10);
    state_l.set_x(-1);
    state_u.set_x(10);
    state_l.set_theta(-M_PI / 4);
    state_u.set_theta(3 * M_PI / 4);
    state_l.set_phi(Tailsitter<double>::kPhiLimitL);
    state_u.set_phi(Tailsitter<double>::kPhiLimitU);

    input_cost.set_phi_dot(100);
    input_cost.set_prop_throttle(100);

    final_state_cost.set_x(1);
    final_state_cost.set_z(1);
    final_state_cost.set_theta(10);
    final_state_cost.set_phi(10);
    final_state_cost.set_x_dot(10);
    final_state_cost.set_z_dot(10);
    final_state_cost.set_theta_dot(10);
  }
  static void log_trajectory(const PPoly& state_opt, const PPoly& input_opt) {
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision);

    for (auto t : state_opt.get_segment_times()) {
      std::cout << std::fixed << std::setprecision(2) << t << "\t\t"
                << state_opt.value(t).transpose().format(CommaInitFmt) << "\t\t"
                << input_opt.value(t).transpose().format(CommaInitFmt)
                << std::endl;
    }

    TailsitterState<double> final_state;
    final_state.SetFromVector(state_opt.value(state_opt.end_time()));

    drake::log()->info(fmt::format(
        "final state:\n x: {:.1f} m\ty: {:.1f} m\tangle: {:.0f} degree",
        final_state.x(), final_state.z(), 180 * final_state.theta() / M_PI));
  }
};

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake