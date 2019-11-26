/**
 * steps:
 * 1. find nominal input(u_0) and state(x_0), S(t), and K(t)
 * 
 */

#include <iostream>
#include <memory>
#include <stdio.h>

#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/solvers/solve.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"

using drake::solvers::SolutionResult;

namespace drake {
namespace examples {
namespace pendulum {

using trajectories::PiecewisePolynomial;

namespace {
typedef trajectories::PiecewisePolynomial<double> PPoly;
typedef std::pair<PPoly, PPoly> TrajPair;

TrajPair optimize_trajectory_dircol() {
  auto pendulum = std::make_unique<PendulumPlant<double>>();
  pendulum->set_name("pendulum");

  auto context = pendulum->CreateDefaultContext();

  const int kNumTimeSamples = 21;
  const double kMinimumTimeStep = 0.2;
  const double kMaximumTimeStep = 0.5;
  systems::trajectory_optimization::DirectCollocation dircol(
      pendulum.get(), *context, kNumTimeSamples, kMinimumTimeStep,
      kMaximumTimeStep);

  dircol.AddEqualTimeIntervalsConstraints();

  // TODO(russt): Add this constraint to PendulumPlant and get it
  // automatically through DirectCollocation.
  const double kTorqueLimit = 3.0;  // N*m.
  const solvers::VectorXDecisionVariable& u = dircol.input();
  dircol.AddConstraintToAllKnotPoints(-kTorqueLimit <= u(0));
  dircol.AddConstraintToAllKnotPoints(u(0) <= kTorqueLimit);

  PendulumState<double> initial_state, final_state;
  initial_state.set_theta(0.0);
  initial_state.set_thetadot(0.0);
  final_state.set_theta(M_PI);
  final_state.set_thetadot(0.0);

  dircol.AddLinearConstraint(dircol.initial_state() ==
                             initial_state.get_value());
  dircol.AddLinearConstraint(dircol.final_state() == final_state.get_value());

  const double R = 10;  // Cost on input "effort".
  dircol.AddRunningCost((R * u) * u);

  const double timespan_init = 4;
  auto traj_init_x = PiecewisePolynomial<double>::FirstOrderHold(
      {0, timespan_init}, {initial_state.get_value(), final_state.get_value()});
  dircol.SetInitialTrajectory(PiecewisePolynomial<double>(), traj_init_x);
  const auto result = solvers::Solve(dircol);
  if (!result.is_success()) {
    std::cerr << "Failed to solve optimization for the swing-up trajectory"
              << std::endl;
    throw "Failed to solve optimization for the swing-up trajectory";
  }
  std::cout<<"Optimum trajectory found with directo collocation. Solver used: "<<result.get_solver_id()<<std::endl; 

  const PPoly u_opt = dircol.ReconstructInputTrajectory(result);
  const PPoly x_opt  = dircol.ReconstructStateTrajectory(result);

  for (double t : x_opt.get_segment_times()) {
    std::cout<<"t: "<<t<<"\tx0: "<<x_opt.value(t)<< std::endl;
  }

  return TrajPair(x_opt, u_opt);
}

}  // namespace
}  // namespace pendulum
}  // namespace examples
}  // namespace drake
