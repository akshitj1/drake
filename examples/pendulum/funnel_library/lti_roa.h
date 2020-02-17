// #include <cmath>
#include <vector>

#include "gflags/gflags.h"

#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/solvers/decision_variable.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/vector_system.h"

namespace drake {
namespace systems {
namespace controllers {

LinearQuadraticRegulatorResult LinearQuadraticRegulatorGains(
    const System<double>& system, const Context<double>& context,
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const int input_port_index = 0) {
  // const int num_inputs = system.get_input_port(input_port_index).size();
  const int num_states = context.num_total_states();
  DRAKE_DEMAND(num_states > 0);
  auto linear_system =
      Linearize(system, context, InputPortIndex{input_port_index},
                OutputPortSelection::kNoOutput);

  LinearQuadraticRegulatorResult lqr_result =
      LinearQuadraticRegulator(linear_system->A(), linear_system->B(), Q, R);
  return lqr_result;
}
}  // namespace controllers
}  // namespace systems
namespace examples {
namespace pendulum {
namespace analysis {

using std::cout;
using std::endl;

using symbolic::Environment;
using symbolic::Expression;
using symbolic::Polynomial;
using symbolic::Variable;
using symbolic::Variables;

using namespace drake::examples::pendulum;
using namespace drake::systems::controllers;

LinearQuadraticRegulatorResult getLqrGains(
    const PendulumState<double>& goal_state) {
  Eigen::MatrixXd Q(2, 2);
  Q << 10, 0, 0, 1;
  Eigen::MatrixXd R(1, 1);
  R << 15;

  PendulumPlant<double> pendulum_d;
  auto lqr_context = pendulum_d.CreateDefaultContext();
  lqr_context->SetContinuousState(goal_state.CopyToVector());
  lqr_context->FixInputPort(0, PendulumInput<double>{}.with_tau(0.0));

  auto lqr_res = systems::controllers::LinearQuadraticRegulatorGains(
      pendulum_d, *lqr_context, Q, R);
  return lqr_res;
}

double LtiRegionOfAttraction() {
  // Create the simple system.
  PendulumPlant<Expression> pendulum;
  auto context = pendulum.CreateDefaultContext();
  auto derivatives = pendulum.AllocateTimeDerivatives();

  PendulumState<double> goal_state;
  goal_state.set_theta(M_PI);
  goal_state.set_thetadot(0);

  auto lqr_res = getLqrGains(goal_state);

  // Setup the optimization problem.
  solvers::MathematicalProgram prog;
  const VectorX<Variable> xvar{prog.NewIndeterminates<2>(
      std::array<std::string, 2>{"theta", "thetadot"})};
  const VectorX<Expression> x = xvar.cast<Expression>();

  // Extract the polynomial dynamics.
  context->get_mutable_continuous_state_vector().SetFromVector(
      x + goal_state.CopyToVector());
  // tau_goal is zero, so additive term ignored
  context->FixInputPort(0, -lqr_res.K * x);
  // pendulum params default are same as req. values
  pendulum.CalcTimeDerivatives(*context, derivatives.get());

  // Define the Lyapunov function.
  const Expression V = x.transpose() * lqr_res.S * x;
  const Environment poly_approx_env{{xvar(0), 0},
                                    {xvar(1), 0}};
  const Expression theta_ddot_poly_approx = symbolic::TaylorExpand(
      derivatives->CopyToVector()[1], poly_approx_env, 3);
  VectorX<Expression> f_poly_approx(2);
  f_poly_approx << derivatives->CopyToVector()[0], theta_ddot_poly_approx;

  const Expression Vdot = 2 * x.transpose() * lqr_res.S * f_poly_approx;

  const Expression lambda{
      prog.NewSosPolynomial(Variables(xvar), 2).first.ToExpression()};

  const double kPrec = 0.1;
  double lb = 0.0, ub = 20.0, rho = (lb + ub) / 2;
  for (rho = (lb + ub) / 2; ub - lb >= kPrec; rho = (lb + ub) / 2) {
    std::cout << "rho: " << rho << endl;
    auto _prog = prog.Clone();
    _prog->AddSosConstraint(-(Vdot + lambda * (rho - V)));
    auto res = Solve(*_prog);
    const double is_feasible = res.is_success();
    if (is_feasible)
      lb = rho;
    else
      ub = rho;
  }

  const double rho_max = rho;

  cout << "Verified that " << V << " < " << rho_max
       << " is in the region of attraction." << endl;

  // Check that 10.0 < ρ < 11.0
  DRAKE_DEMAND(rho_max > 10.0 && rho_max < 11.0);
  return rho_max;
}
}  // namespace analysis
}  // namespace pendulum
}  // namespace examples
}  // namespace drake