#include "limits"
#include <iomanip>

#include <fmt/format.h>

#include "gflags/gflags.h"

#include "drake/common/default_scalars.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/examples/tailsitter/gen/tailsitter_input.h"
#include "drake/examples/tailsitter/gen/tailsitter_state.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"

namespace drake {
using systems::Context;
using systems::ContinuousState;
using systems::VectorBase;

namespace examples {
namespace tailsitter {
typedef trajectories::PiecewisePolynomial<double> PPoly;

namespace {
static const double kEps = std::numeric_limits<double>::epsilon();
static const double kInf = std::numeric_limits<double>::infinity() / 2;
const int kNumStates = TailsitterState<double>::K::kNumCoordinates;
const int kNumInputs = TailsitterInput<double>::K::kNumCoordinates;

template <typename T>
class TailsitterController : public systems::LeafSystem<T> {
  const PPoly& u_opt;

 public:
  TailsitterController(const PPoly& u_opt) : u_opt(u_opt) {
    this->DeclareVectorInputPort("tailsitter_state", TailsitterState<T>());
    this->DeclareVectorOutputPort("control_inputs", TailsitterInput<T>(),
                                  &TailsitterController::CalcElevonDeflection);
  }
  void CalcElevonDeflection(const systems::Context<T>& context,
                            TailsitterInput<T>* control) const {
    const double& t = context.get_time();
    // auto u = u_opt.value(t);
    // control->SetFromVector(u);
    control->SetFromVector(u_opt.value(t));
  }
};

class TrajectoryOptimizer {
 private:
  const int kNumTimeSamples = 41;
  const double kTimeStepMin = 0.01, kTimeStepMax = 2;

  systems::trajectory_optimization::DirectCollocation optimizer;

  const systems::LeafSystem<double>& plant;
  const int kNumStates;

  VectorX<double> intitial_state_des, final_state_des;

 public:
  TrajectoryOptimizer(const systems::LeafSystem<double>& plant_)
      : plant(plant_),
        kNumStates(plant.CreateDefaultContext()->num_continuous_states()),
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
    auto input_traj = PPoly::ZeroOrderHold(
        {0, traj_duration},
        {Vector2<double>(0, 0.5), Vector2<double>(0, 0.5)});  // PPoly();

    optimizer.SetInitialTrajectory(input_traj, state_traj);
  }

 public:
  void get_optimum_trajectories(PPoly& state_opt, PPoly& input_opt,
                                const double max_traj_time = 10) {
    set_trajectory_guess();
    optimizer.AddDurationBounds(0, max_traj_time);

    optimizer.SetSolverOption(solvers::IpoptSolver::id(), "print_level", 5);

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

static void get_perching_constraints(TailsitterState<double>& takeoff_state,
                                     TailsitterState<double>& land_state,
                                     TailsitterState<double>& land_tol,
                                     TailsitterState<double>& state_l,
                                     TailsitterState<double>& state_u,
                                     TailsitterInput<double>& input_cost,
                                     TailsitterState<double>& final_cost) {
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

  final_cost.set_x(10);
  final_cost.set_z(10);
  final_cost.set_theta(1);
  final_cost.set_phi(10);
  final_cost.set_x_dot(1);
  final_cost.set_z_dot(1);
  final_cost.set_theta_dot(1);
}

static void get_climb_constraints(TailsitterState<double>& takeoff_state,
                                  TailsitterState<double>& land_state,
                                  TailsitterState<double>& land_tol,
                                  TailsitterState<double>& state_l,
                                  TailsitterState<double>& state_u,
                                  TailsitterInput<double>& input_cost,
                                  TailsitterState<double>& final_cost) {
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

  final_cost.set_x(10);
  final_cost.set_z(10);
  final_cost.set_theta(1);
  final_cost.set_phi(10);
  final_cost.set_x_dot(1);
  final_cost.set_z_dot(1);
  final_cost.set_theta_dot(1);
}

static void get_transistion_constraints(TailsitterState<double>& takeoff_state,
                                        TailsitterState<double>& land_state,
                                        TailsitterState<double>& land_tol,
                                        TailsitterState<double>& state_l,
                                        TailsitterState<double>& state_u,
                                        TailsitterInput<double>& input_cost,
                                        TailsitterState<double>& final_cost) {
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

  final_cost.set_x(1);
  final_cost.set_z(1);
  final_cost.set_theta(10);
  final_cost.set_phi(10);
  final_cost.set_x_dot(10);
  final_cost.set_z_dot(10);
  final_cost.set_theta_dot(10);
}

enum Trajectory { Perch, Climb, Hover, Transistion };

/*
 * ref:
 * https://github.com/RobotLocomotion/drake/blob/last_sha_with_original_matlab/drake/examples/Glider/runDircolPerching.m
 */
static void get_trajectory(const Trajectory& trajectory,
                           const Tailsitter<double>& tailsitter,
                           PPoly& state_opt, PPoly& input_opt) {
  TailsitterState<double> takeoff_state, land_state, land_tol, state_l, state_u;
  TailsitterInput<double> input_cost;
  TailsitterState<double> final_state_cost;

  switch (trajectory) {
    case Perch:
      get_perching_constraints(takeoff_state, land_state, land_tol, state_l,
                               state_u, input_cost, final_state_cost);
      break;
    case Climb:
      get_climb_constraints(takeoff_state, land_state, land_tol, state_l,
                            state_u, input_cost, final_state_cost);
      break;
    case Transistion:
      get_transistion_constraints(takeoff_state, land_state, land_tol, state_l,
                                  state_u, input_cost, final_state_cost);
      break;
    default:
      log()->error(fmt::format("{} trajectory not implemented", trajectory));
      throw;
  }

  TailsitterInput<double> input_l, input_u;
  input_l.set_phi_dot(-Tailsitter<double>::kPhiDotLimit);
  input_u.set_phi_dot(Tailsitter<double>::kPhiDotLimit);
  input_l.set_prop_throttle(0);
  input_u.set_prop_throttle(1);

  TrajectoryOptimizer traj_optimizer(tailsitter);

  traj_optimizer.set_initial_state(takeoff_state.CopyToVector());
  traj_optimizer.set_final_state(land_state.CopyToVector(),
                                 land_tol.CopyToVector());
  traj_optimizer.bound_state(state_l.CopyToVector(), state_u.CopyToVector());
  traj_optimizer.bound_input(input_l.CopyToVector(), input_u.CopyToVector());

  traj_optimizer.add_running_cost(input_cost.CopyToVector());
  traj_optimizer.add_final_cost(final_state_cost.CopyToVector());

  traj_optimizer.get_optimum_trajectories(state_opt, input_opt);

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
};

/*
 * tutorial on unique_ptr:
 * https://thispointer.com/c11-unique_ptr-tutorial-and-examples/
 */
void simulate_trajectory(const Trajectory& trajectory) {
  systems::DiagramBuilder<double> builder;
  auto tailsitter = builder.AddSystem<Tailsitter>();
  PPoly state_opt, input_opt;
  get_trajectory(trajectory, *tailsitter, state_opt, input_opt);

  auto ts_controller =
      builder.AddSystem<TailsitterController<double>>(input_opt);
  builder.Connect(tailsitter->get_output_port(0),
                  ts_controller->get_input_port(0));
  builder.Connect(ts_controller->get_output_port(0),
                  tailsitter->get_input_port(0));

  auto diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);

  simulator.set_target_realtime_rate(0);
  simulator.AdvanceTo(state_opt.end_time());
}
}  // namespace

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Trajectory optimization for perching tailsitter.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::set_log_level("info");
  drake::examples::tailsitter::simulate_trajectory(
      drake::examples::tailsitter::Transistion);
  return 0;
}