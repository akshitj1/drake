#include "limits"
#include <fmt/format.h>

#include "gflags/gflags.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"

#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/solvers/solve.h"
//#include "drake/systems/analysis/simulator.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"

namespace drake {
using systems::Context;
using systems::ContinuousState;
// using systems::Simulator;
using systems::VectorBase;

namespace examples {
namespace tailsitter {
typedef trajectories::PiecewisePolynomial<double> PPoly;
typedef std::pair<PPoly, PPoly> TrajPair;
namespace {

/*
 * ref:
 * https://github.com/RobotLocomotion/drake/blob/last_sha_with_original_matlab/drake/examples/Glider/runDircolPerching.m
 */
TrajPair run_dircol(const Tailsitter<double>& tailsitter) {
  const int kNumTimeSamples = 41;
  const double kMinimumTimeStep = 0.01, kMaximumTimeStep = 2;

  const double kPhiLimitL = -M_PI / 3, kPhiLimitU = M_PI / 6, kPhiDotLimit = 13;
  const double kInf = std::numeric_limits<double>::infinity();
  const double kEps = std::numeric_limits<double>::epsilon();

  VectorX<double> takeoff_state(7), traj_state_l(7), traj_state_u(7),
      land_state(7), land_state_l(7), land_state_u(7), eps(7);

  // set initial state
  takeoff_state << -3.5, 0.1, 0, 0, 7, 0, 0;

  traj_state_l << -4, -1, -M_PI / 2, kPhiLimitL, -kInf, -kInf, -kInf;
  traj_state_u << 1, 1, M_PI / 2, kPhiLimitU, kInf, kInf, kInf;

  land_state << 0, 0, M_PI / 4, 0, 0, -0.5, -0.5;
  land_state_l << -kEps, -kEps, M_PI / 8, -kInf, -2, -2, -kInf;
  land_state_u << kEps, kEps, M_PI / 2, kInf, 2, 2, kInf;

  const double fly_time_init = 1;  // abs(takeoff_state(0)) / takeoff_state(4);

  const double R = 100;

  auto context = tailsitter.CreateDefaultContext();
  systems::trajectory_optimization::DirectCollocation dircol(
      &tailsitter, *context, kNumTimeSamples, kMinimumTimeStep, kMaximumTimeStep);

  dircol.AddEqualTimeIntervalsConstraints();

  // input constraints
  auto u = dircol.input();
  dircol.AddConstraintToAllKnotPoints(u(0) >= -kPhiDotLimit);
  dircol.AddConstraintToAllKnotPoints(u(0) <= kPhiDotLimit);

  // state constraints
  auto state = dircol.state();
  auto i_state = dircol.initial_state();
  auto f_state = dircol.final_state();
  dircol.AddConstraintToAllKnotPoints(traj_state_l <= state);
  dircol.AddConstraintToAllKnotPoints(state <= traj_state_u);
  // dircol.AddBoundingBoxConstraint(traj_state_l, traj_state_u, state);

  dircol.AddLinearConstraint(i_state == takeoff_state);

  // set final box target
  dircol.AddBoundingBoxConstraint(land_state_l, land_state_u, f_state);

  // bound running time
  dircol.AddDurationBounds(0, 10);

  dircol.AddRunningCost(u.transpose() * R * u);

  VectorX<double> q(7);
  q << 10, 10, 1, 10, 1, 1, 1;
  MatrixX<double> Q(7, 7);
  Q = q.asDiagonal();
  auto x_err = f_state - land_state;
  dircol.AddFinalCost((Q * x_err).cwiseProduct(x_err).sum());

  eps << kEps, kEps, kEps, 0, 0, 0, 0;

  auto traj_init = PPoly::FirstOrderHold({0, fly_time_init},
                                         {takeoff_state + eps, land_state});

  dircol.SetInitialTrajectory(PPoly(), traj_init);
  const auto result = solvers::Solve(dircol);

  if (!result.is_success()) {
    throw result.get_solver_id().name() +
        " Failed to solve optimization for the perching trajectory";
  }

  PPoly x_des = dircol.ReconstructStateTrajectory(result);
  PPoly u_des = dircol.ReconstructInputTrajectory(result);

  const VectorX<double> x_final = x_des.value(x_des.end_time());
  drake::log()->info(
      fmt::format("final state:\n x: {:.1f} m\ty: {:.1f} m\tangle: {:.0f} degree",
              x_final[0], x_final[1], 180 * x_final[2] / M_PI));

  return TrajPair(x_des, u_des);
}

/*
 * tutorial on unique_ptr:
 * https://thispointer.com/c11-unique_ptr-tutorial-and-examples/
 */
void do_main() {
  Tailsitter<double> tailsitter;
  ;
  tailsitter.set_name("tailsitter");

  // get optimal state and input trajectories to reach goal
  auto tr_des = run_dircol(tailsitter);
}
}  // namespace

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Trajectory optimization for perching tailsitter.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::tailsitter::do_main();
  return 0;
}