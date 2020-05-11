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
using namespace trajectories;
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

/**
 * Compute a funnel in which Time varying LQR can stabilize system and reach
 * into goal state. Goal is to use this for constructing LQR trees with finite
 * number of such funnels.
 * refs:
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
 */

/**
 * Blueprint
 * Vtraj=x'Sx from tvlqr
 * replace sys with polysys <- taylor approx of closed loop system f(x_bar,
 * -k(x_bar)) of order 3 c=15.5, lagrange degree=2, niters=2 G = x'Qfx(G is
 * unused) assert G.eval(tf)==V.eval(tf) at each break:
 * - substitute t=t_break into V, Vdot=jac(V,x)*f_poly + jac(V, t)
 * - balance V, Vdot
 * - Vmin = -slack s.t slack+V=sos
 * rho = exp increasing from 1/e^c to 1.0 + max(Vmin)
 * approx dt and rho_dot with delta rho/ delta t
 * sample check asserting Vdot<=rhodot for all samples
 * for optimzation iteration
 * - find multipliers, if gamma>1e-4 debug v-rho, vdot-rhodot
 */

class TrajectoryFunnel {
  const System<double>& system;
  const PiecewisePolynomial<double>& state_nominal;
  const PiecewisePolynomial<double>& input_nominal;
  const controllers::FiniteHorizonLinearQuadraticRegulatorResult& lqr_res;
  const double start_time, end_time;
  const int num_states, num_inputs;

 public:
  TrajectoryFunnel(
      const System<double>& system_,
      const PiecewisePolynomial<double>& state_nominal_,
      const PiecewisePolynomial<double>& input_nominal_,
      const controllers::FiniteHorizonLinearQuadraticRegulatorResult& lqr_res_)
      : system(system_),
        state_nominal(state_nominal_),
        input_nominal(input_nominal_),
        lqr_res(lqr_res_),
        start_time(state_nominal.start_time()),
        end_time(state_nominal.end_time()),
        num_states(system_.CreateDefaultContext()->num_total_states()),
        num_inputs(system_.get_input_port(0).size()) {
    solvers::MathematicalProgram prog;
    // Define the relative coordinates: x_bar = x - x0
    const VectorX<Variable> x_bar = prog.NewIndeterminates(num_states, "x");
    vector<double> t_breaks = lqr_res.S.get_segment_times();

    // rho guess
    vector<double> rho, rho_dot;
    // rho[i] defines rho for [t[i], t[i+1]), rho[N] for t[N]
    const double c = 15.5;
    for (int i = 0; i < t_breaks.size(); i++) {
      rho.push_back(exp(-c * (t_breaks.back() - t_breaks[i]) /
                        (t_breaks.back() - t_breaks.front())));
      if (i)
        rho_dot.push_back((rho[i] - rho[i - 1]) /
                          (t_breaks[i] - t_breaks[i - 1]));
    }

    // V guess
    vector<Polynomial> V, V_dot;
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
    // todo: assert V_f=Q_f?

    assert(V.size() == t_breaks.size() && rho.size() == V.size() &&
           V_dot.size() == rho_dot.size() && V_dot.size() == V.size() - 1);

    const int max_iter = 10;
    const double convergence_tol = 0.01;
    double prev_rho_integral = 0, rho_integral;
    for (int iter = 0; iter < max_iter; iter++) {
      PolynomialTrajectory mu;

      // start bilinear optimization of rho and lagrange multipliers
      for (int i = 0; i < V_dot.size(); i++) {
        drake::log()->info(
            fmt::format("finding lagrange multipliers for segment {}", i));
        mu.push_back(optimize_lagrange_multipliers(x_bar, V[i], V_dot[i],
                                                   rho[i], rho_dot[i]));
      }
      log()->info("lagrange multipliers optimized for trajectory");
      optimize_rho(x_bar, t_breaks, V, V_dot, mu, rho, rho_dot, rho_integral);
      log()->info(
          fmt::format("rhos optimized for trajectory.\niter: {}\nvolume: "
                      "{:.3f}\ngain: {:.3f}",
                      iter, rho_integral, rho_integral - prev_rho_integral));
      if (rho_integral - prev_rho_integral < convergence_tol) {
        log()->info("funnel optimization converged.");
        break;
      }
      prev_rho_integral = rho_integral;
    }
  }

  static VectorX<Polynomial> f_approx_polynomial(const System<double>& system,
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

  static void balance_V_with_Vdot(const VectorX<Variable>& x, Polynomial& V,
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
};  // namespace analysis

}  // namespace analysis
}  // namespace systems
}  // namespace drake
