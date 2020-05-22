#pragma once

// #include <algorithm>
#include <chrono>
#include <vector>

#include "drake/common/polynomial.h"
#include "drake/common/symbolic.h"
#include "drake/common/symbolic_variables.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/math/quadratic_form.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
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
using symbolic::Monomial;
using symbolic::Polynomial;
using symbolic::Substitution;
using symbolic::TaylorExpand;
using symbolic::Variable;
using symbolic::Variables;
using namespace std::chrono;

typedef MatrixX<Polynomial> PolynomialFrame;
typedef std::vector<Polynomial> PolynomialTrajectory;

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
 * (for LTI fixed point tracking)
 * https://github.com/RobotLocomotion/drake/pull/1176/files
 * (Funnel library for quadrotor)
 */
class FunnelOptimizer {
 private:
  const System<double>& system;
  const PiecewisePolynomial<double>& state_nominal;
  const PiecewisePolynomial<double>& input_nominal;
  // Define the relative coordinates: x_bar = x - x0
  VectorX<Variable> x_bar;
  Variables x_bar_vars;
  const vector<double> t_breaks;
  const int num_points, num_segments;
  const int num_states, num_inputs;

  vector<VectorX<Polynomial>> u_bar;
  vector<Polynomial> rho;
  vector<Polynomial> V;
  // for normalization constratint in V
  vector<double> trace_V0;
  vector<Polynomial> lambda;

  const int u_deg = 1, l_deg = u_deg + 1, f_deg = 1, V_deg = 2;
  const int max_iter = 10;
  const double convergence_tolerance = 0.1;

 public:
  FunnelOptimizer(const System<double>& _system,
                  const PiecewisePolynomial<double>& _state_nominal,
                  const PiecewisePolynomial<double>& _input_nominal,
                  const FiniteHorizonLinearQuadraticRegulatorResult& _lqr_res,
                  // to change u(x) in order to maximize funnel?
                  const bool optimize_controller = false)
      : system(_system),
        state_nominal(_state_nominal),
        input_nominal(_input_nominal),
        t_breaks(resample(_lqr_res.S.get_segment_times())),
        num_points(t_breaks.size()),
        num_segments(num_points - 1),
        num_states(system.CreateDefaultContext()->num_total_states()),
        num_inputs(system.get_input_port(0).size()) {
    x_bar = symbolic::MakeVectorContinuousVariable(num_states, "x_bar");
    x_bar_vars = Variables(x_bar);
    rho_guess();
    u_guess(_lqr_res.K);
    lyapunov_guess(_lqr_res.S);
    assert(V.size() == num_points && rho.size() == num_points &&
           trace_V0.size() == num_points);

    maximize_funnel(false);
  }

 private:
  static vector<double> resample(const vector<double>& x,
                                 const int num_samples = 105) {
    vector<double> x_resampled(num_samples);
    const double span = x.back() - x.front();
    const double interval = span / (num_samples - 1);
    for (int i = 0; i < num_samples; i++)
      x_resampled[i] = std::min(x.front() + i * interval, x.back());
    return x_resampled;
  }
  Polynomial to_poly(const Expression& e) { return Polynomial(e, x_bar_vars); }
  VectorX<Polynomial> to_poly(const VectorX<Expression>& e) {
    return e.unaryExpr(
        [this](const Expression& _e) { return this->to_poly(_e); });
  }

  void rho_guess() {
    const double c = 15.5;
    for (int i = 0; i < num_points; i++) {
      rho.push_back(Polynomial(exp(-c * (t(-1) - t(i)) / (t(-1) - t(0)))));
    }
  }
  vector<Polynomial> get_rho_dot() {
    vector<Polynomial> rho_dot;
    for (int i = 0; i < num_segments; i++) rho_dot.push_back(get_rho_dot(i));
    return rho_dot;
  }
  Polynomial get_rho_dot(const int& point) {
    assert(point < num_segments);
    return (rho[point] - rho[point + 1]) / (t(point) - t(point + 1));
  }

  void lyapunov_guess(const PiecewisePolynomial<double>& S) {
    for (int i = 0; i < t_breaks.size(); i++) {
      V.push_back(to_poly(x_bar.dot(S.value(t(i)) * x_bar))
                      .RemoveTermsWithSmallCoefficients(1e-4));
      trace_V0.push_back(S.value(t(i)).trace());
    }
  }

  void u_guess(const PiecewisePolynomial<double>& K) {
    for (int i = 0; i < num_segments; i++) {
      u_bar.push_back(to_poly(-K.value(t(i)) * x_bar));
    }
  }

  double t(const int& point) {
    int in_range_point = point % num_points < 0
                             ? point % num_points + num_points
                             : point % num_points;
    return t_breaks[in_range_point];
  }

  // approx system to a polynomial at break in time
  VectorX<Polynomial> f_approx_polynomial(const int& point) {
    if (f_deg == 1) return f_approx_linear(point);
    const auto symbolic_system = system.ToSymbolic();
    const auto symbolic_context = symbolic_system->CreateDefaultContext();
    // our dynamics are time invariant. Do we need this?
    symbolic_context->SetTime(0.0);
    symbolic_context->SetContinuousState(state_nominal.value(t(point)) + x_bar);
    symbolic_context->FixInputPort(
        0, input_nominal.value(t(point)) +
               u_bar[point].unaryExpr(
                   [](const Polynomial& p) { return p.ToExpression(); }));

    // todo: move this outside
    // for taylor approximating system
    Environment f_approx_env;
    f_approx_env.insert(x_bar, VectorX<double>::Zero(num_states));

    const VectorX<Expression> f =
        symbolic_system->EvalTimeDerivatives(*symbolic_context)
            .get_vector()
            .CopyToVector();

    // we can expand x_bar terms to n degree but input terms to only single
    // degree as u coeffecients need to be linear for SDP.
    // todo: fix for arbitrary f_deg>1
    const VectorX<Polynomial> f_poly =
        f.unaryExpr([f_approx_env, this](const Expression& xi_dot) {
          return to_poly(TaylorExpand(xi_dot, f_approx_env, this->f_deg));
        });

    const VectorX<Polynomial> f0 =
        f.unaryExpr([this, f_approx_env](const Expression& xi_dot) {
          return this->to_poly(xi_dot.EvaluatePartial(f_approx_env));
        });

    return f_poly - f0;
  }
  VectorX<Polynomial> f_approx_linear(const int& point) {
    auto lin_context = system.CreateDefaultContext();
    lin_context->SetContinuousState(state_nominal.value(t(point)));
    lin_context->FixInputPort(0, input_nominal.value(t(point)));
    auto f_lin = FirstOrderTaylorApproximation(
        system, *lin_context, InputPortSelection::kUseFirstInputIfItExists,
        OutputPortSelection::kNoOutput);

    const VectorX<Expression> fbar_cl_lin =
        f_lin->A() * x_bar +
        f_lin->B() * u_bar[point].unaryExpr(
                         [](const Polynomial& p) { return p.ToExpression(); });
    return to_poly(fbar_cl_lin).unaryExpr([](const Polynomial& x) {
      return x.RemoveTermsWithSmallCoefficients(1e-4);
    });
  }

  void balance_V_with_Vdot(Polynomial& V, Polynomial& Vdot) {
    Environment env;
    for (int i = 0; i < num_states; i++) {
      env.insert(x_bar(i), 0.0);
    }
    const Eigen::MatrixXd S =
        symbolic::Evaluate(symbolic::Jacobian(V.Jacobian(x_bar), x_bar), env);
    const MatrixX<double> P = symbolic::Evaluate(
        symbolic::Jacobian(Vdot.Jacobian(x_bar), x_bar), env);

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
    const VectorX<Expression> Tx = T * x_bar;
    symbolic::Substitution subs;
    for (int i = 0; i < num_states; i++) {
      subs.emplace(x_bar(i), Tx(i));
    }
    V = to_poly(V.ToExpression().Substitute(subs));
    Vdot = to_poly(Vdot.ToExpression().Substitute(subs));
  }

  Polynomial get_V_dot(const int& point) {
    // v_dot = jac(V, x_bar)*x_bar_dot + del_V/del_t
    Polynomial V_dot{V[point].Jacobian(x_bar) * f_approx_polynomial(point) +
                     (V[point] - V[point + 1]) / (t(point) - t(point + 1))};

    //  balance V and V_dot
    // todo: V is for break while V_dot is for segment. is this right?
    // balance_V_with_Vdot(x_bar, V[i - 1], V_dot[i - 1]);

    return V_dot.RemoveTermsWithSmallCoefficients(1e-4);
  }
  void extract_solution(const solvers::MathematicalProgramResult& res,
                        Polynomial& x) {
    x = to_poly(res.GetSolution(x.ToExpression()));
  }

  void extract_solution(const solvers::MathematicalProgramResult& res,
                        VectorX<Polynomial>& x) {
    for (int i = 0; i < x.size(); i++) extract_solution(res, x[i]);
  }

  static Polynomial get_linear_polynomial(MathematicalProgram& prog,
                                          const VectorX<Variable>& x) {
    const solvers::VectorXDecisionVariable coeffs{
        prog.NewContinuousVariables(x.size(), "a")};
    symbolic::Polynomial p;
    for (int i = 0; i < x.size(); ++i) {
      p.AddProduct(coeffs(i),
                   Monomial(x(i)));  // p += coeffs(i) * m(i);
    }
    return p;
  }

  VectorX<Polynomial> get_parametrized_u(MathematicalProgram& prog) {
    VectorX<Polynomial> u_bar(num_inputs);
    for (int i = 0; i < num_inputs; i++)
      u_bar[i] = get_linear_polynomial(prog, x_bar);
    return u_bar;
  }

  // Step 1
  double find_l_u() {
    if (lambda.size() == 0) lambda = vector<Polynomial>(num_segments);
    for (int i = 0; i < num_segments; i++) {
      drake::log()->info(
          fmt::format("finding lagrange multipliers for segment {}/{}", i + 1,
                      num_segments));
      MathematicalProgram prog;
      prog.AddIndeterminates(x_bar);
      u_bar[i] = get_parametrized_u(prog);
      // todo: clean V and V_dot terms with small coeffs.
      lambda[i] = prog.NewFreePolynomial(x_bar_vars, l_deg);

      // const Variable gamma = prog.NewContinuousVariables<1>("gamma")[0];
      // prog.AddConstraint(gamma <= 0);
      // prog.AddCost(gamma);

      prog.AddSosConstraint(-get_V_dot(i) + get_rho_dot(i) +
                            lambda[i] * (rho[i] - V[i]));
      // we do not add any cost as we are solving for only SOS feasibility

      const auto res = Solve(prog);
      assert(res.is_success());
      extract_solution(res, u_bar[i]);
      extract_solution(res, lambda[i]);
    }
    Expression rho_integral;
    for (int i = 0; i < num_points; i++) rho_integral += rho[i].ToExpression();
    return rho_integral.Evaluate();
  }

  // Step 2
  double find_u_rho() {
    MathematicalProgram prog;
    prog.AddIndeterminates(x_bar);
    Polynomial rho_integral;
    for (int i = 0; i < num_points; i++) {
      // prog.newnonnegativePoly requires deg>0
      rho[i] = Polynomial(prog.NewContinuousVariables<1>("rho")[0]);
      prog.AddConstraint(rho[i].ToExpression() >= 0);
      rho_integral += rho[i];
      // todo: rhof=1?
      if (i < num_segments) u_bar[i] = get_parametrized_u(prog);

      // to get dot at x_i_dot, x_i+1 should also be avail.
      if (i)
        prog.AddSosConstraint(-get_V_dot(i - 1) + get_rho_dot(i - 1) +
                              lambda[i - 1] * (rho[i - 1] - V[i - 1]));
    }
    prog.AddConstraint(rho.back().ToExpression() == 1.0);
    prog.AddCost(-rho_integral.ToExpression());

    const auto res = Solve(prog);
    assert(res.is_success());

    for (int i = 0; i < num_points; i++) {
      extract_solution(res, rho[i]);
      if (i < num_segments) extract_solution(res, u_bar[i]);
    }
    // rho integral
    extract_solution(res, rho_integral);
    return rho_integral.Evaluate(Environment());
  }

  static Expression trace(const MatrixX<Variable>& X) {
    // built-in eigen cannot convert variable sum to expression
    assert(X.rows() == X.cols());
    Expression trace_X;
    for (int i = 0; i < X.rows(); i++) trace_X += X(i, i);
    return trace_X;
  }

  // Step 3
  double find_V_rho() {
    MathematicalProgram prog;
    prog.AddIndeterminates(x_bar);
    Polynomial rho_integral;
    for (int i = 0; i < num_points; i++) {
      // prog.newnonnegativePoly requires deg>0
      rho[i] = Polynomial(prog.NewContinuousVariables<1>("rho")[0]);
      prog.AddConstraint(rho[i].ToExpression() >= 0);
      rho_integral += rho[i];

      auto V_S = prog.NewSosPolynomial(x_bar.template cast<Monomial>());
      assert(V_S.second.rows() == num_states &&
             V_S.second.cols() == num_states);
      V[i] = V_S.first;
      prog.AddConstraint(trace(V_S.second) == trace_V0[i]);

      // to get dot at x_i_dot, x_i+1 should also be avail.
      if (i)
        prog.AddSosConstraint(-get_V_dot(i - 1) + get_rho_dot(i - 1) +
                              lambda[i - 1] * (rho[i - 1] - V[i - 1]));
    }
    prog.AddConstraint(rho.back().ToExpression() == 1.0);
    prog.AddCost(-rho_integral.ToExpression());

    const auto res = Solve(prog);
    assert(res.is_success());

    for (int i = 0; i < num_points; i++) {
      extract_solution(res, rho[i]);
      extract_solution(res, V[i]);
      V[i] = V[i].RemoveTermsWithSmallCoefficients(1e-4);
    }
    // rho integral
    extract_solution(res, rho_integral);
    return rho_integral.Evaluate(Environment());
  };
  void plot_funnel() {
    // fix theta_dot
    Polynomial V_theta = V[0].EvaluatePartial(x_bar[1], 0.0);
    MathematicalProgram prog;
    prog.AddDecisionVariables(Vector1<Variable>(x_bar[0]));
    prog.AddConstraint(V_theta.ToExpression() == rho[0].ToExpression());
    prog.AddCost(-x_bar[0]);
    const auto res = Solve(prog);
    assert(res.is_success());
    log()->info(
        fmt::format("theta max dev.: {:.3f}", res.GetSolution(x_bar[0])));
  }

  void find_l() {
    if (lambda.size() == 0) lambda = vector<Polynomial>(num_segments);
    for (int i = 0; i < num_segments; i++) {
      drake::log()->info(
          fmt::format("finding lagrange multipliers for segment {}/{}", i + 1,
                      num_segments));

      MathematicalProgram prog;
      prog.AddIndeterminates(x_bar);
      // todo: clean V and V_dot terms with small coeffs.
      lambda[i] = prog.NewFreePolynomial(x_bar_vars, l_deg);

      const Variable gamma = prog.NewContinuousVariables<1>("gamma")[0];
      prog.AddConstraint(gamma <= 0);
      prog.AddCost(gamma);

      const Polynomial sos_condn =
          gamma - get_V_dot(i) + get_rho_dot(i) + lambda[i] * (rho[i] - V[i]);
      // this asserts that polynomials were formed from prog indeterminates
      // itself and no reparse is needed, as this step is time consuming
      assert(sos_condn.indeterminates().IsSubsetOf(
          Variables(prog.indeterminates())));
      auto start = high_resolution_clock::now();
      prog.AddSosConstraint(sos_condn);
      log()->info("time taken: {} ms", duration_cast<milliseconds>(
                                           high_resolution_clock::now() - start)
                                           .count());

      const auto res = Solve(prog);

      assert(res.is_success());
      extract_solution(res, lambda[i]);
    }
  }

  double find_rho() {
    MathematicalProgram prog;
    prog.AddIndeterminates(x_bar);
    Polynomial funnel_volume;
    for (int i = 0; i < num_points; i++) {
      // prog.newnonnegativePoly requires deg>0
      rho[i] = to_poly(prog.NewContinuousVariables<1>("rho")[0]);
      prog.AddConstraint(rho[i].ToExpression() >= 0);

      // to get dot at x_i_dot, x_i+1 should also be avail.
      if (i) {
        prog.AddSosConstraint(-get_V_dot(i - 1) + get_rho_dot(i - 1) +
                              lambda[i - 1] * (rho[i - 1] - V[i - 1]));

        double dt = t(i) - t(i - 1);
        funnel_volume += rho[i - 1] * dt + (dt * get_rho_dot(i - 1) * dt) / 2;
      }
    }
    prog.AddConstraint(rho.back().ToExpression() == 1.0);
    prog.AddCost(-funnel_volume.ToExpression());

    const auto res = Solve(prog);
    assert(res.is_success());

    for (int i = 0; i < num_points; i++) extract_solution(res, rho[i]);

    extract_solution(res, funnel_volume);
    return funnel_volume.Evaluate(Environment());
  }

  void maximize_funnel(const bool& optimize_controller = true) {
    double prev_rho_integral = 0.1,
           rho_integral = 2 * (1 + convergence_tolerance) * prev_rho_integral;
    for (int iter = 0; iter < max_iter;
         iter++, prev_rho_integral = rho_integral) {
      if (optimize_controller) {
        log()->info("Step 1: Find L and u with V and rho fixed.");
        rho_integral = find_l_u();
        log()->info(fmt::format("iter: {} 1/3\trho integral: {:.3f}", iter,
                                rho_integral));

        log()->info("Step 2: Optimize u and rho with V and l fixed.");
        rho_integral = find_u_rho();
        log()->info(fmt::format("iter: {} 2/3\trho integral: {:.3f}", iter,
                                rho_integral));

        log()->info("Step 3: Optimize V and rho with u and l fixed.");
        rho_integral = find_V_rho();
        log()->info(fmt::format("iter: {} 3/3\trho integral: {:.3f}", iter,
                                rho_integral));

        log()->info(
            fmt::format("rhos optimized for trajectory.\niter: {}\nvolume: "
                        "{:.3f}\ngain: {:.3f}",
                        iter, rho_integral, rho_integral - prev_rho_integral));
      } else {
        log()->info("Step 1: Find l with rho fixed");
        find_l();

        log()->info("Step 2: Optimize rho with l fixed.");
        rho_integral = find_rho();

        log()->info(
            fmt::format("rhos optimized for trajectory.\niter: {}\nvolume: "
                        "{:.3f}\ngain: {:.3f}",
                        iter, rho_integral, rho_integral - prev_rho_integral));
      }
      plot_funnel();

      if ((rho_integral - prev_rho_integral) / prev_rho_integral <
          convergence_tolerance) {
        log()->info("funnel optimization converged.");
        return;
      }
    }
    log()->warn("solution failed to converge. Reached max iteration!!!");
  }
};
}  // namespace
}  // namespace analysis
}  // namespace systems
}  // namespace drake
