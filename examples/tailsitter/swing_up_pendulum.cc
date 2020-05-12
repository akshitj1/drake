#include "limits"

#include "gflags/gflags.h"

#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/pendulum/gen/pendulum_input.h"
#include "drake/examples/pendulum/gen/pendulum_state.h"
#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/examples/tailsitter/trajectory_funnel.h"
#include "drake/examples/tailsitter/trajectory_optimizer.h"
#include "drake/systems/controllers/finite_horizon_linear_quadratic_regulator.h"

namespace drake {
namespace examples {
namespace pendulum {
namespace {
using systems::analysis::TrajectoryFunnel;
using systems::controllers::FiniteHorizonLinearQuadraticRegulator;
using systems::controllers::FiniteHorizonLinearQuadraticRegulatorOptions;
using systems::controllers::FiniteHorizonLinearQuadraticRegulatorResult;

using trajectories::PiecewisePolynomial;
void log_trajectory(const PiecewisePolynomial<double>& x0,
                    const PiecewisePolynomial<double>& u0) {
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision);

  for (auto t : x0.get_segment_times()) {
    std::cout << std::fixed << std::setprecision(2) << t << "\t\t"
              << x0.value(t).transpose().format(CommaInitFmt) << "\t\t"
              << u0.value(t).transpose().format(CommaInitFmt) << std::endl;
  }
}

void swing_up_trajectory(const PendulumPlant<double>& pendulum,
                         PiecewisePolynomial<double>& x0,
                         PiecewisePolynomial<double>& u0) {
  static const double kEps = std::numeric_limits<double>::epsilon();
  static const double kInf = std::numeric_limits<double>::infinity() / 2;

  PendulumState<double> xi, xf, xf_tol;
  xf.set_theta(M_PI);
  xf_tol.set_theta(0.01);
  xf_tol.set_thetadot(1);

  PendulumState<double> xl, xu;
  xl.set_theta(-M_PI);
  xu.set_theta(M_PI);
  xl.set_thetadot(-kInf);
  xu.set_thetadot(kInf);

  PendulumInput<double> ul, uu;
  const double kTorqueLimit = 3.0;
  ul.set_tau(-kTorqueLimit);
  uu.set_tau(kTorqueLimit);

  PendulumInput<double> u_cost;
  u_cost.set_tau(10);

  systems::trajectory_optimization::TrajectoryOptimizer traj_optimizer(
      pendulum);

  traj_optimizer.set_initial_state(xi.CopyToVector());
  traj_optimizer.set_final_state(xf.CopyToVector(), xf_tol.CopyToVector());
  traj_optimizer.bound_state(xl.CopyToVector(), xu.CopyToVector());
  traj_optimizer.bound_input(ul.CopyToVector(), uu.CopyToVector());
  traj_optimizer.add_running_cost(u_cost.CopyToVector());

  traj_optimizer.get_optimum_trajectories(x0, u0);
  log_trajectory(x0, u0);
}

FiniteHorizonLinearQuadraticRegulatorResult stabilize_lqr(
    const PendulumPlant<double>& pendulum,
    const PiecewisePolynomial<double>& x0,
    const PiecewisePolynomial<double>& u0) {
  PendulumState<double> xf_tol;
  xf_tol.set_theta(0.1);
  xf_tol.set_thetadot(1);
  const MatrixX<double> Qf{
      xf_tol.CopyToVector().array().square().inverse().matrix().asDiagonal()};

  PendulumState<double> x_cost;
  x_cost.set_theta(0.1);
  x_cost.set_thetadot(0.1);
  const MatrixX<double> Q{x_cost.CopyToVector().asDiagonal()};

  PendulumInput<double> u_cost;
  u_cost.set_tau(0.1);
  const MatrixX<double> R{u_cost.CopyToVector().asDiagonal()};

  auto lqr_context = pendulum.CreateDefaultContext();

  FiniteHorizonLinearQuadraticRegulatorOptions lqr_options;
  lqr_options.Qf = Qf;
  lqr_options.u0 = &u0;
  lqr_options.x0 = &x0;

  FiniteHorizonLinearQuadraticRegulatorResult lqr_res =
      FiniteHorizonLinearQuadraticRegulator(pendulum, *lqr_context,
                                            x0.start_time(), x0.end_time(), Q,
                                            R, lqr_options);
  log()->info("lqr S, K matrices computed.");
  return lqr_res;
}

void swing_up() {
  PendulumPlant<double> pendulum;
  PiecewisePolynomial<double> x0, u0;
  swing_up_trajectory(pendulum, x0, u0);
  FiniteHorizonLinearQuadraticRegulatorResult lqr_res =
      stabilize_lqr(pendulum, x0, u0);
  log()->info("computing trajectory funnel");
  TrajectoryFunnel(pendulum, x0, u0, lqr_res);
  log()->info("Funnel computed");
}
}  // namespace
}  // namespace pendulum
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Trajectory optimization for pendulum swing up.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::set_log_level("info");
  drake::examples::pendulum::swing_up();
  return 0;
}