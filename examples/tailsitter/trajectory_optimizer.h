#pragma once

#include "limits"
#include <iomanip>

#include <fmt/format.h>

#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"

namespace drake {
namespace systems {
namespace trajectory_optimization {
typedef trajectories::PiecewisePolynomial<double> PPoly;
static const double kEps = std::numeric_limits<double>::epsilon();
static const double kInf = std::numeric_limits<double>::infinity() / 2;

/**
 * given initial, final and state constraints finds optimum trajectory for a
 * plant with given cost function
 */
class TrajectoryOptimizer {
 private:
  const int kNumTimeSamples = 41;
  const double kTimeStepMin = 0.01, kTimeStepMax = 2;

  systems::trajectory_optimization::DirectCollocation optimizer;

  const systems::LeafSystem<double>& plant;
  const int kNumStates, kNumInputs;

  VectorX<double> intitial_state_des, final_state_des;

 public:
  TrajectoryOptimizer(const systems::LeafSystem<double>& plant_)
      : plant(plant_),
        kNumStates(plant.CreateDefaultContext()->num_continuous_states()),
        kNumInputs(plant.get_input_port(0).size()),
        optimizer(systems::trajectory_optimization::DirectCollocation(
            &plant_, *plant_.CreateDefaultContext(), kNumTimeSamples,
            kTimeStepMin, kTimeStepMax)) {
    optimizer.AddEqualTimeIntervalsConstraints();
  }
  void set_initial_state(const VectorX<double>& init_state) {
    intitial_state_des = init_state;
    optimizer.AddConstraint(optimizer.initial_state() == intitial_state_des);
  }

  void set_final_state(const VectorX<double>& fin_state,
                       const VectorX<double>& tol) {
    final_state_des = fin_state;
    optimizer.AddBoundingBoxConstraint(
        final_state_des - tol, final_state_des + tol, optimizer.final_state());
  }

  void bound_state(const VectorX<double>& state_l,
                   const VectorX<double>& state_u) {
    optimizer.AddConstraintToAllKnotPoints(optimizer.state() >= state_l);
    optimizer.AddConstraintToAllKnotPoints(optimizer.state() <= state_u);
  }

  void bound_input(const VectorX<double>& input_l,
                   const VectorX<double>& input_u) {
    optimizer.AddConstraintToAllKnotPoints(optimizer.input() >= input_l);
    optimizer.AddConstraintToAllKnotPoints(optimizer.input() <= input_u);
  }

  void add_running_cost(const VectorX<double>& input_cost) {
    MatrixX<double> R(input_cost.asDiagonal());
    optimizer.AddRunningCost(optimizer.input().transpose() * R *
                             optimizer.input());
  }

  void add_final_cost(const VectorX<double>& penalty_weights) {
    MatrixX<double> Qf(penalty_weights.size(), penalty_weights.size());
    Qf = penalty_weights.asDiagonal();
    auto final_state_err = optimizer.final_state() - final_state_des;
    optimizer.AddFinalCost(
        (Qf * final_state_err).cwiseProduct(final_state_err).sum());
  }

 private:
  void set_trajectory_guess(const double traj_duration = 1) {
    const VectorX<double> eps = VectorX<double>::Constant(kNumStates, kEps);

    auto state_traj = PPoly::FirstOrderHold(
        {0, traj_duration}, {intitial_state_des + eps, final_state_des});
    auto input_traj = PPoly();
    // PPoly::ZeroOrderHold(
    //   {0, traj_duration},
    // {Vector2<double>(0, 0.5), Vector2<double>(0, 0.5)});

    optimizer.SetInitialTrajectory(input_traj, state_traj);
  }

 public:
  void get_optimum_trajectories(PPoly& state_opt, PPoly& input_opt,
                                const double max_traj_time = 10) {
    set_trajectory_guess();
    optimizer.AddDurationBounds(0, max_traj_time);

    optimizer.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 3);

    log()->info("computing optimal perching trajectory...");
    const auto result = solvers::Solve(optimizer);

    if (!result.is_success()) {
      log()->error(
          "{} Failed to solve optimization while finding optimum trajectory",
          result.get_solver_id().name());
      throw;
    }
    log()->info("found optimal perching trajectory...");

    state_opt = optimizer.ReconstructStateTrajectory(result);
    input_opt = optimizer.ReconstructInputTrajectory(result);
  }
};
}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake