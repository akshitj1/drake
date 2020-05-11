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
using std::vector;
using symbolic::Environment;
using symbolic::Expression;
using symbolic::Substitution;
using symbolic::Variable;

typedef MatrixX<symbolic::Polynomial> PolynomialFrame;
typedef std::vector<symbolic::Polynomial> PolynomialTrajectory;

static symbolic::Polynomial toSymbolicPolynomial(
    const drake::Polynomial<double>& p, const symbolic::Variable& t_sy) {
  symbolic::Polynomial p_sy;
  // polynomial is sum of its monomials and stores monomials list in vector
  // monomials are coeff, and terms(vector)
  // terms is var id(int) and int power(int)
  // var id is generated from var name string
  // for our use case we only have to deal with time variable t(name "t").
  const std::vector<drake::Polynomial<double>::Monomial> monomials =
      p.GetMonomials();
  const drake::Polynomial<double>::VarType t =
      drake::Polynomial<double>::VariableNameToId("t");
  for (drake::Polynomial<double>::Monomial m : monomials) {
    assert(m.terms.size() <= 1);
    if (m.terms.size()) {
      const drake::Polynomial<double>::Term term = m.terms[0];
      assert(term.var == t);
      symbolic::Monomial m_sy(t_sy, term.power);
      p_sy += m.coefficient * m_sy;
    } else {
      p_sy += m.coefficient;
    }
  }
  return p_sy;
}
static MatrixX<symbolic::Polynomial> toSymbolicPolynomial(
    const MatrixX<drake::Polynomial<double>>& p_mat,
    const symbolic::Variable& t_sy) {
  MatrixX<symbolic::Polynomial> p_mat_sy =
      p_mat.unaryExpr([t_sy](drake::Polynomial<double> p) {
        return toSymbolicPolynomial(p, t_sy);
      });
  return p_mat_sy;
}

static vector<MatrixX<symbolic::Polynomial>> toSymbolicPolynomial(
    const PiecewisePolynomial<double>& p_traj, const symbolic::Variable& t_sy) {
  vector<MatrixX<symbolic::Polynomial>> p_traj_sy;
  for (int i = 0; i < p_traj.get_number_of_segments(); i++) {
    p_traj_sy.push_back(
        toSymbolicPolynomial(p_traj.getPolynomialMatrix(i), t_sy));
  }
  return p_traj_sy;
}

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
    const Variable t("t");
    vector<double> time_samples;
    PolynomialTrajectory V;
    get_initial_lyapunov_candidate(x_bar, t, V, time_samples);
    drake::log()->info(fmt::format(
        "created {} time breaks out of span {:.1f} secs", time_samples.size(),
        time_samples.back() - time_samples.front()));
    PolynomialTrajectory rho = get_initial_rho(time_samples, t);

    // find V_dot
    PolynomialTrajectory V_dot, rho_dot;

    // for taylor approximating system
    Environment f_approx_env;
    for (int i = 0; i < num_states; i++) {
      f_approx_env.insert(x_bar(i), 0.0);
    }

    for (int i = 0; i < V.size(); i++) {
      const double t_i = time_samples[i];
      const auto symbolic_system = system.ToSymbolic();
      const auto symbolic_context = symbolic_system->CreateDefaultContext();
      // our dynamics are time invariant. Do we need this?
      symbolic_context->SetTime(0.0);
      symbolic_context->SetContinuousState(state_nominal.value(t_i) + x_bar);
      symbolic_context->FixInputPort(
          0, input_nominal.value(t_i) - lqr_res.K.value(t_i) * x_bar);

      const VectorX<Expression> f =
          symbolic_system->EvalTimeDerivatives(*symbolic_context)
              .get_vector()
              .CopyToVector();

      const VectorX<symbolic::Polynomial> f_bar_poly =
          f.unaryExpr([f_approx_env](const Expression& xi_dot) {
            const double x0i_dot = xi_dot.Evaluate(f_approx_env);
            return symbolic::Polynomial(
                symbolic::TaylorExpand(xi_dot - x0i_dot, f_approx_env, 2));
          });

      symbolic::Polynomial V_dot_i = (V[i].Jacobian(x_bar) * f_bar_poly +
                                      V[i].Jacobian(Vector1<Variable>(t)))[0];

      V[i] = V[i].EvaluatePartial(t, t_i);
      V_dot.push_back(V_dot_i.EvaluatePartial(t, t_i));
      // balance V and Vdot if Vdot is negative definite. todo: why do this?
      balance_V_with_Vdot(x_bar, V[i], V_dot[i]);

      // these are simply doubles todo: change type
      rho_dot.push_back(rho[i].Differentiate(t).EvaluatePartial(t, t_i));
      rho[i] = rho[i].EvaluatePartial(t, t_i);

      drake::log()->info(
          fmt::format("finding lagrange multipliers for segment {}", i));
      optimize_lagrange_multipliers(x_bar, V[i], V_dot[i], rho[i], rho_dot[i]);
    }
    const int max_iterations = 2;
    for (int iter = 0; iter < max_iterations; iter++) {
      PolynomialTrajectory mu =
          optimize_lagrange_multipliers(x_bar, V, V_dot, rho, rho_dot);
      optimize_rho(prog, x_bar, t, time_samples, V, V_dot, mu, rho, rho_dot);
    }
  }

  static void balance_V_with_Vdot(const VectorX<Variable>& x,
                                  symbolic::Polynomial& V,
                                  symbolic::Polynomial& Vdot) {
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
    V = symbolic::Polynomial(V.ToExpression().Substitute(subs));
    Vdot = symbolic::Polynomial(Vdot.ToExpression().Substitute(subs));
  }

  // sec. 3.1 [2]
  PolynomialTrajectory optimize_lagrange_multipliers(
      const solvers::VectorXIndeterminate& x, const PolynomialTrajectory& V,
      const PolynomialTrajectory& V_dot, const PolynomialTrajectory& rho,
      const PolynomialTrajectory& rho_dot) {
    // todo: assert all time segments consistent
    PolynomialTrajectory mu;
    for (int i = 0; i < V.size(); i++) {
      // for (int x_idx = 0; x_idx < x.size(); x_idx++) {
      //   const int x_i_deg = V_dot[i].Degree(x[x_idx]);
      //   if (x_i_deg > mu_degree) mu_degree = x_i_deg;
      // }
      // drake::log()->info(
      //     fmt::format("lagrange polynomial degree: {}", mu_degree));

      mu.push_back(
          optimize_lagrange_multipliers(x, V[i], V_dot[i], rho[i], rho_dot[i]));
    }
    return mu;
  }

  symbolic::Polynomial optimize_lagrange_multipliers(
      const solvers::VectorXIndeterminate& x, const symbolic::Polynomial& V,
      const symbolic::Polynomial& V_dot, const symbolic::Polynomial& rho,
      const symbolic::Polynomial& rho_dot) {
    solvers::MathematicalProgram prog;
    prog.AddIndeterminates(x);

    // creates decision variable
    const Variable gamma = prog.NewContinuousVariables<1>("gamma")[0];
    const int mu_degree = 2;
    const symbolic::Polynomial mu =
        prog.NewFreePolynomial(symbolic::Variables(x), mu_degree);
    prog.AddSosConstraint(gamma - (V_dot - rho_dot + mu * (V - rho)));
    prog.AddCost(gamma);

    const auto result = solvers::Solve(prog);
    assert(result.is_success());
    assert(result.GetSolution(gamma) < 0);
    return symbolic::Polynomial(result.GetSolution(mu.ToExpression()));
  }

  void optimize_rho(
      const MathematicalProgram& _prog, const solvers::VectorXIndeterminate& x,
      const symbolic::Variable& t, const vector<double>& time_samples,
      const PolynomialTrajectory& V, const PolynomialTrajectory& V_dot,
      const PolynomialTrajectory& mu, PolynomialTrajectory& rho_opt,
      PolynomialTrajectory& rho_opt_dot) {
    std::unique_ptr<MathematicalProgram> prog = _prog.Clone();
    PolynomialTrajectory rho, rho_dot;
    for (int i = 0; i < V.size(); i++) {
      rho.push_back(prog->NewFreePolynomial(symbolic::Variables({t}), 1));
      rho_dot.push_back(rho.back().Differentiate(t));
    }
    const int num_time_samples = time_samples.size();
    prog->AddConstraint(
        rho.back().EvaluatePartial(t, time_samples.back()).ToExpression() ==
        1.0);
    for (int i = 0; i < rho.size() - 1; i++) {
      prog->AddConstraint(rho[i].EvaluatePartial(t, time_samples[i + 1]) ==
                          rho[i + 1].EvaluatePartial(t, time_samples[i + 1]));
    }
    symbolic::Polynomial volume_objective;

    for (int i = 0; i < rho.size(); i++) {
      volume_objective += rho[i].EvaluatePartial(t, time_samples[i]) *
                          (time_samples[i + 1] - time_samples[i]);
      prog->AddSosConstraint(-(V_dot[i] - rho_dot[i] + mu[i] * (rho[i] - V[i]))
                                  .EvaluatePartial(t, time_samples[i]));
    }
    prog->AddCost(-volume_objective.ToExpression());

    auto result = solvers::Solve(*prog);
    assert(result.is_success());
    for (int i = 0; i < rho.size(); i++) {
      rho_opt[i] =
          symbolic::Polynomial(result.GetSolution(rho[i].ToExpression()));
      rho_opt_dot[i] = rho_opt[i].Differentiate(t);
    }
  }
  static VectorX<symbolic::Polynomial> polynomial_cast(
      const VectorX<Variable>& x) {
    return x.unaryExpr(
        [](const Variable& el) { return symbolic::Polynomial(el); });
  }

  vector<symbolic::Polynomial> get_initial_rho(
      const std::vector<double>& time_samples, const Variable& t) {
    const double c = 15.5;

    vector<MatrixX<double>> rho_knots;
    std::transform(time_samples.begin(), time_samples.end(),
                   std::back_inserter(rho_knots),
                   [time_samples, c](const double& t_break) {
                     return Vector1<double>(
                         exp(-c * (time_samples.back() - t_break) /
                             (time_samples.back() - time_samples.front())));
                   });
    const PiecewisePolynomial<double> rho =
        PiecewisePolynomial<double>::FirstOrderHold(time_samples, rho_knots);
    vector<symbolic::Polynomial> rho_sy;
    for (int i = 0; i < rho.get_number_of_segments(); i++) {
      rho_sy.push_back(toSymbolicPolynomial(rho.getPolynomial(i), t));
    }
    return rho_sy;
  }

  void get_initial_lyapunov_candidate(const VectorX<Variable>& x_bar,
                                      const Variable& t,
                                      vector<symbolic::Polynomial>& V,
                                      vector<double>& time_samples) {
    const PiecewisePolynomial<double> S = DensePPolyToSpare(lqr_res.S);

    time_samples = S.get_segment_times();
    const VectorX<symbolic::Polynomial> x_bar_poly = polynomial_cast(x_bar);

    for (int i = 0; i < S.get_number_of_segments(); i++) {
      const symbolic::Polynomial V_i = x_bar_poly.dot(
          toSymbolicPolynomial(S.getPolynomialMatrix(i), t) * x_bar_poly);
      V.push_back(V_i);
    }
  }
  static PiecewisePolynomial<double> DensePPolyToSpare(
      const PiecewisePolynomial<double>& p, const int num_final_segments = 40) {
    VectorX<double> time_breaks_vectorized = VectorX<double>::LinSpaced(
        num_final_segments, p.start_time(), p.end_time());
    vector<double> time_breaks(
        time_breaks_vectorized.data(),
        time_breaks_vectorized.data() + time_breaks_vectorized.size());
    vector<MatrixX<double>> p_knots;
    std::transform(time_breaks.begin(), time_breaks.end(),
                   std::back_inserter(p_knots),
                   [p](const double& t_break) { return p.value(t_break); });
    return PiecewisePolynomial<double>::FirstOrderHold(time_breaks, p_knots);
  }
};

}  // namespace analysis
}  // namespace systems
}  // namespace drake
