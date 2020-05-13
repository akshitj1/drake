#pragma once

#include <vector>

#include "drake/common/polynomial.h"
#include "drake/common/symbolic.h"
#include "drake/common/symbolic_variables.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/math/quadratic_form.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/region_of_attraction.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/finite_horizon_linear_quadratic_regulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/system.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
namespace systems {
namespace analysis {
namespace {
using namespace trajectories;
using controllers::FiniteHorizonLinearQuadraticRegulatorResult;
using solvers::MathematicalProgram;
using solvers::Solve;
using std::vector;
using symbolic::Environment;
using symbolic::Expression;
using symbolic::Polynomial;
using symbolic::Substitution;
using symbolic::TaylorExpand;
using symbolic::Variable;

typedef MatrixX<Polynomial> PolynomialFrame;
typedef std::vector<Polynomial> PolynomialTrajectory;

void rho_guess(const vector<double>& t_breaks, vector<double>& rho,
               vector<double>& rho_dot) {
  const double c = 15.5;
  for (int i = 0; i < t_breaks.size(); i++) {
    rho.push_back(exp(-c * (t_breaks.back() - t_breaks[i]) /
                      (t_breaks.back() - t_breaks.front())));
    if (i)
      rho_dot.push_back((rho[i] - rho[i - 1]) /
                        (t_breaks[i] - t_breaks[i - 1]));
  }
}

VectorX<Polynomial> f_approx_polynomial(const System<double>& system,
                                        const VectorX<Variable>& x_bar,
                                        const MatrixX<double>& K0,
                                        const VectorX<double>& x0,
                                        const VectorX<double>& u0) {
  const auto symbolic_system = system.ToSymbolic();
  const auto symbolic_context = symbolic_system->CreateDefaultContext();
  // our dynamics are time invariant. Do we need this?
  symbolic_context->SetTime(0.0);
  symbolic_context->SetContinuousState(x0 + x_bar);
  symbolic_context->FixInputPort(0, u0 - K0 * x_bar);

  // todo: move this outside
  // for taylor approximating system
  Environment f_approx_env;
  for (int i = 0; i < x_bar.size(); i++) {
    f_approx_env.insert(x_bar(i), 0.0);
  }

  const VectorX<Expression> f =
      symbolic_system->EvalTimeDerivatives(*symbolic_context)
          .get_vector()
          .CopyToVector();

  const VectorX<double> f0 =
      f.unaryExpr([f_approx_env](const Expression& xi_dot) {
        return xi_dot.Evaluate(f_approx_env);
      });

  const VectorX<Polynomial> f_poly =
      f.unaryExpr([f_approx_env](const Expression& xi_dot) {
        return Polynomial(TaylorExpand(xi_dot, f_approx_env, 2));
      });
  return f_poly - f0;
}

void balance_V_with_Vdot(const VectorX<Variable>& x, Polynomial& V,
                         Polynomial& Vdot) {
  Environment env;
  for (int i = 0; i < x.size(); i++) {
    env.insert(x(i), 0.0);
  }
  const Eigen::MatrixXd S =
      symbolic::Evaluate(symbolic::Jacobian(V.Jacobian(x), x), env);
  const MatrixX<double> P =
      symbolic::Evaluate(symbolic::Jacobian(Vdot.Jacobian(x), x), env);

  // check if negative definite
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(P);
  DRAKE_THROW_UNLESS(eigensolver.info() == Eigen::Success);

  // A positive max eigenvalue indicates the system is locally unstable.
  const double max_eigenvalue = eigensolver.eigenvalues().maxCoeff();
  // According to the Lapack manual, the absolute accuracy of eigenvalues is
  // eps*max(|eigenvalues|), so I will write my thresholds in those units.
  // Anderson et al., Lapack User's Guide, 3rd ed. section 4.7, 1999.
  const double tolerance = 1e-8;
  const double max_abs_eigenvalue =
      eigensolver.eigenvalues().cwiseAbs().maxCoeff();
  // DRAKE_THROW_UNLESS(max_eigenvalue <=
  //                    tolerance * std::max(1., max_abs_eigenvalue));

  bool Vdot_is_locally_negative_definite =
      (max_eigenvalue <= -tolerance * std::max(1., max_abs_eigenvalue));

  if (!Vdot_is_locally_negative_definite) return;

  const Eigen::MatrixXd T = math::BalanceQuadraticForms(S, -P);
  const VectorX<Expression> Tx = T * x;
  symbolic::Substitution subs;
  for (int i = 0; i < static_cast<int>(x.size()); i++) {
    subs.emplace(x(i), Tx(i));
  }
  V = Polynomial(V.ToExpression().Substitute(subs));
  Vdot = Polynomial(Vdot.ToExpression().Substitute(subs));
}

void lyapunov_guess(const System<double>& system,
                    const const VectorX<Variable>& x_bar,
                    const vector<double>& t_breaks,
                    const FiniteHorizonLinearQuadraticRegulatorResult& lqr_res,
                    const PiecewisePolynomial<double>& state_nominal,
                    const PiecewisePolynomial<double>& input_nominal,
                    vector<Polynomial>& V, vector<Polynomial>& V_dot) {
  for (int i = 0; i < t_breaks.size(); i++) {
    const double t = t_breaks[i];
    const MatrixX<double> S = lqr_res.S.value(t);
    V.push_back(Polynomial(x_bar.dot(S * x_bar)));

    if (i) {  // v_dot = jac(V, x_bar)*x_bar_dot + del_V/del_t
      const Polynomial V_dot_prev =
          V[i - 1].Jacobian(x_bar) *
              f_approx_polynomial(system, x_bar, lqr_res.K.value(t),
                                  state_nominal.value(t_breaks[i - 1]),
                                  input_nominal.value(t_breaks[i - 1])) +
          (V[i] - V[i - 1]) / (t_breaks[i] - t_breaks[i - 1]);
      V_dot.push_back(V_dot_prev);
      // balance V and V_dot
      // todo: V is for break while V_dot is for segment. is this right?
      balance_V_with_Vdot(x_bar, V[i - 1], V_dot[i - 1]);
    }
  }
}

Polynomial optimize_lagrange_multipliers(const VectorX<Variable>& x,
                                         const Polynomial& V,
                                         const Polynomial& V_dot,
                                         const double& rho,
                                         const double& rho_dot) {
  MathematicalProgram prog;
  prog.AddIndeterminates(x);

  // creates decision variable
  const Variable gamma = prog.NewContinuousVariables<1>("gamma")[0];
  const int mu_degree = 2;
  const Polynomial mu =
      prog.NewFreePolynomial(symbolic::Variables(x), mu_degree);
  prog.AddSosConstraint(gamma - (V_dot - rho_dot + mu * (V - rho)));
  prog.AddCost(gamma);

  const auto result = solvers::Solve(prog);
  assert(result.is_success());
  assert(result.GetSolution(gamma) < 0);
  return Polynomial(result.GetSolution(mu.ToExpression()));
}

void optimize_lagrange_multipliers(const VectorX<Variable>& x_bar,
                                   const vector<Polynomial>& V,
                                   const vector<Polynomial>& V_dot,
                                   const vector<double>& rho,
                                   const vector<double>& rho_dot,
                                   vector<Polynomial>& mu) {
  // start bilinear optimization of rho and lagrange multipliers
  for (int i = 0; i < V_dot.size(); i++) {
    drake::log()->info(
        fmt::format("finding lagrange multipliers for segment {}", i));
    mu.push_back(optimize_lagrange_multipliers(x_bar, V[i], V_dot[i], rho[i],
                                               rho_dot[i]));
  }
}

void optimize_rho(const VectorX<Variable>& x, const vector<double>& t_breaks,
                  const PolynomialTrajectory& V,
                  const PolynomialTrajectory& V_dot,
                  const PolynomialTrajectory& mu, vector<double>& rho_opt,
                  vector<double>& rho_opt_dot, double& rho_integral) {
  assert(mu.size() == V_dot.size());
  MathematicalProgram prog;
  prog.AddIndeterminates(x);
  const solvers::VectorXDecisionVariable rho =
      prog.NewContinuousVariables(V.size(), "rho");
  prog.AddConstraint(rho[rho.size() - 1] == 1.0);

  Polynomial volume_obj;

  for (int i = 0; i < t_breaks.size() - 1; i++) {
    // for speed, bound variables to avoid free variables
    prog.AddConstraint(rho[i] >= 0);

    const double dt = t_breaks[i + 1] - t_breaks[i];
    const Polynomial rho_dot((rho[i + 1] - rho[i]) / dt);
    volume_obj += Polynomial(rho[i] * dt) + (dt * rho_dot * dt) / 2;
    prog.AddSosConstraint(
        -(V_dot[i] - rho_dot + mu[i] * (V[i] - Polynomial(rho[i]))));
  }

  prog.AddCost(-volume_obj.ToExpression());

  const auto result = solvers::Solve(prog);
  assert(result.is_success());

  for (int i = 0; i < rho.size(); i++) {
    rho_opt[i] = result.GetSolution(rho[i]);
    assert(rho_opt[i] > 0);
    if (i)
      rho_opt_dot[i - 1] =
          (rho_opt[i] - rho_opt[i - 1]) / (t_breaks[i] - t_breaks[i - 1]);
  }
  rho_integral = -result.get_optimal_cost();
}

void plot_funnel(const VectorX<Variable>& x_bar, const Polynomial& V,
                 const double& rho) {
  // fix theta_dot
  Polynomial V_theta = V.EvaluatePartial(x_bar[1], 0.0);
  MathematicalProgram prog;
  prog.AddDecisionVariables(Vector1<Variable>(x_bar[0]));
  prog.AddConstraint(V_theta.ToExpression() == rho);
  prog.AddCost(-x_bar[0]);
  const auto res = Solve(prog);
  assert(res.is_success());
  log()->info(fmt::format("theta max dev.: {:.3f}", res.GetSolution(x_bar[0])));
}

void maximize_funnel_for_fixed_controller(
    const VectorX<Variable>& x_bar, const vector<double>& t_breaks,
    const vector<Polynomial>& V, const vector<Polynomial>& V_dot,
    vector<double>& rho, vector<double>& rho_dot, const double max_iter = 10,
    const double convergence_tol = 0.01) {
  assert(V.size() == t_breaks.size() && rho.size() == V.size() &&
         V_dot.size() == rho_dot.size() && V_dot.size() == V.size() - 1);

  double prev_rho_integral = 0, rho_integral = 2 * convergence_tol;
  for (int iter = 0; iter < max_iter;
       iter++, prev_rho_integral = rho_integral) {
    PolynomialTrajectory mu;
    optimize_lagrange_multipliers(x_bar, V, V_dot, rho, rho_dot, mu);
    log()->info("lagrange multipliers optimized for trajectory");

    optimize_rho(x_bar, t_breaks, V, V_dot, mu, rho, rho_dot, rho_integral);
    log()->info(
        fmt::format("rhos optimized for trajectory.\niter: {}\nvolume: "
                    "{:.3f}\ngain: {:.3f}",
                    iter, rho_integral, rho_integral - prev_rho_integral));
    plot_funnel(x_bar, V[0], rho[0]);

    if (rho_integral - prev_rho_integral < convergence_tol) {
      log()->info("funnel optimization converged.");
      return;
    }
  }
  log()->warn("solution failed to converge. Reached max iteration!!!");
}

/**
 * Compute a funnel in which Time varying LQR can track given trajectory of
 * system and reach into goal region. End goal is to use this for constructing
 * LQR trees with finite number of such funnels. refs:
 *
 * [1] LQR-Trees: Feedback Motion Planning via Sums-of-Squares Verification:
 * https://groups.csail.mit.edu/robotics-center/public_papers/Tedrake10.pdf
 * (more understndable but involves greedy non-convex optimization)
 *
 * Accompanying video, setting motivation for approach:
 * https://www.youtube.com/watch?v=uh13FoZLnPo
 *
 * [2] Invariant Funnels around Trajectories using Sum-of-Squares Programming:
 * https://arxiv.org/pdf/1010.3013v1.pdf (intermediate proofs along with convex
 * otimization)
 *
 * Last matlab implementation:
 * https://github.com/RobotLocomotion/drake/blob/last_sha_with_original_matlab/drake/matlab/systems/%40PolynomialSystem/sampledFiniteTimeVerification.m
 *
 * Usage for glider:
 * https://github.com/RobotLocomotion/drake/blob/last_sha_with_original_matlab/drake/examples/Glider/perchingFunnel.m
 *
 * Zippped matlab implementation(old):
 * https://groups.csail.mit.edu/locomotion/software.html
 *
 * [3] Control Design along Trajectories with Sums of Squares Programming - Ani
 * Majumdar https://arxiv.org/pdf/1210.0888.pdf
 *
 * Unlike [2] which tries to maximize funnel for fixed feedback controller, [3]
 * aims to find feedback controller which maximizes size of funnel. This is
 * desirable as we want to minimize number of funnels for our LQR-trees
 *
 * matlab implementation:
 * https://github.com/RobotLocomotion/drake/blob/last_sha_with_original_matlab/drake/matlab/systems/%40PolynomialSystem/maxROAFeedback.m
 */

void TrajectoryFunnel(
    const System<double>& system,
    const PiecewisePolynomial<double>& state_nominal,
    const PiecewisePolynomial<double>& input_nominal,
    const FiniteHorizonLinearQuadraticRegulatorResult& lqr_res) {
  const int num_states = system.CreateDefaultContext()->num_total_states();
  const int num_inputs = system.get_input_port(0).size();
  solvers::MathematicalProgram prog;

  // Define the relative coordinates: x_bar = x - x0
  const VectorX<Variable> x_bar = prog.NewIndeterminates(num_states, "x");
  vector<double> t_breaks = lqr_res.S.get_segment_times();

  // rho guess
  // rho[i] defines rho for [t[i], t[i+1]), rho[N] for t[N]
  vector<double> rho, rho_dot;
  rho_guess(t_breaks, rho, rho_dot);

  // V guess
  vector<Polynomial> V, V_dot;
  lyapunov_guess(system, x_bar, t_breaks, lqr_res, state_nominal, input_nominal,
                 V, V_dot);
  // todo: assert V_f=Q_f?

  maximize_funnel_for_fixed_controller(x_bar, t_breaks, V, V_dot, rho, rho_dot);
}

}  // namespace
}  // namespace analysis
}  // namespace systems
}  // namespace drake
