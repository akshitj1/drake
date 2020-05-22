#pragma once

#include <chrono>
#include <vector>

#include "drake/common/polynomial.h"
#include "drake/common/symbolic.h"
#include "drake/common/symbolic_variables.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/system.h"

namespace drake {
namespace systems {
namespace analysis {
namespace {

using solvers::MathematicalProgram;
using solvers::Solve;
using std::vector;
using symbolic::Environment;
using symbolic::Expression;
using symbolic::Monomial;
using symbolic::Polynomial;
using symbolic::TaylorExpand;
using symbolic::Variable;
using symbolic::Variables;
using namespace std::chrono;

class FixedStateROA {
 private:
  const System<double>& system;
  const VectorX<double>& state_nominal;
  const VectorX<double>& input_nominal;
  // Define the relative coordinates: x_bar = x - x0
  VectorX<Variable> x_bar;
  Variables x_bar_vars;
  const int num_states, num_inputs;

  VectorX<Polynomial> u_bar;
  Polynomial V;
  Polynomial V_dot;
  Polynomial lambda;

  const int u_deg = 1, l_deg = u_deg + 1, f_deg = 2, V_deg = 2;
  const int max_iter = 10;
  const double convergence_tolerance = 0.1;

 public:
  FixedStateROA(const System<double>& _system,
                const VectorX<double>& _state_nominal,
                const VectorX<double>& _input_nominal,
                const controllers::LinearQuadraticRegulatorResult& _lqr_res)
      : system(_system),
        state_nominal(_state_nominal),
        input_nominal(_input_nominal),
        num_states(system.CreateDefaultContext()->num_total_states()),
        num_inputs(system.get_input_port(0).size()) {
    x_bar = symbolic::MakeVectorContinuousVariable(num_states, "x_bar");
    x_bar_vars = Variables(x_bar);
    u_bar = to_poly(-_lqr_res.K * x_bar);
    // lyapunov guess
    V = to_poly(x_bar.dot(_lqr_res.S * x_bar));
    V_dot = Polynomial(V.Jacobian(x_bar) * f_approx_poly());

    const double rho = find_rho();
    log()->info(fmt::format("rho max: {:3f}", rho));
    // theta plot
    plot_funnel(rho, 2);
  }
  double find_rho() {
    // performs line search on rho, for each rho attempts to check sos
    // feasibility for lambda
    const double kPrec = 0.01;
    double lb = 0.0, ub = 100.0;
    double rho;
    for (rho = (lb + ub) / 2; ub - lb >= kPrec; rho = (lb + ub) / 2) {
      MathematicalProgram prog;
      prog.AddIndeterminates(x_bar);
      lambda = prog.NewSosPolynomial(x_bar_vars, l_deg).first;
      prog.AddSosConstraint(-V_dot + lambda * (V - rho));
      auto res = Solve(prog);
      if (res.is_success())
        lb = rho;
      else
        ub = rho;
      log()->info(fmt::format("rho: {:.3f}", rho));
    }
    rho = lb;
    return rho;
  }

 private:
  Polynomial to_poly(const Expression& e) { return Polynomial(e, x_bar_vars); }
  VectorX<Polynomial> to_poly(const VectorX<Expression>& e) {
    return e.unaryExpr(
        [this](const Expression& _e) { return this->to_poly(_e); });
  }

  VectorX<Expression> to_expression(const VectorX<Polynomial>& p) {
    return p.unaryExpr([](const Polynomial& _p) { return _p.ToExpression(); });
  }

  void extract_solution(const solvers::MathematicalProgramResult& res,
                        Polynomial& x) {
    x = to_poly(res.GetSolution(x.ToExpression()));
  }
  Polynomial clean(const Polynomial& p, const double neglect_threshold = 1e-4) {
    return p.RemoveTermsWithSmallCoefficients(neglect_threshold);
  }

  VectorX<Polynomial> f_approx_poly() {
    const auto symbolic_system = system.ToSymbolic();
    const auto symbolic_context = symbolic_system->CreateDefaultContext();
    symbolic_context->SetContinuousState(state_nominal + x_bar);
    symbolic_context->FixInputPort(0, to_expression(input_nominal + u_bar));

    // for taylor approximating system
    Environment f_approx_env;
    f_approx_env.insert(x_bar, VectorX<double>::Zero(num_states));

    const VectorX<Expression> f =
        symbolic_system->EvalTimeDerivatives(*symbolic_context)
            .get_vector()
            .CopyToVector();

    const VectorX<Polynomial> f_poly =
        f.unaryExpr([f_approx_env, this](const Expression& xi_dot) {
          return clean(
              to_poly(TaylorExpand(xi_dot, f_approx_env, this->f_deg)));
        });

    // our f0 should be zero. todo: assert zero?
    // const VectorX<Polynomial> f0 =
    //     f.unaryExpr([this, f_approx_env](const Expression& xi_dot) {
    //       return this->to_poly(xi_dot.EvaluatePartial(f_approx_env));
    //     });

    return f_poly;
  }
  void plot_funnel(const double& rho, const int& plot_x_idx = 0) {
    Environment plot_env;
    for (int i = 0; i < num_states; i++)
      if (i != plot_x_idx) plot_env.insert(x_bar[i], state_nominal[i]);

    // fix theta_dot
    Polynomial V_x_plot = V.EvaluatePartial(plot_env);
    MathematicalProgram prog;
    prog.AddDecisionVariables(Vector1<Variable>(x_bar[plot_x_idx]));
    prog.AddConstraint(V_x_plot.ToExpression() <= rho);
    prog.AddCost(-x_bar[plot_x_idx]);
    const auto res = Solve(prog);
    assert(res.is_success());
    log()->info(fmt::format("x_bar({}) max dev.: {:.3f}", plot_x_idx,
                            res.GetSolution(x_bar[plot_x_idx])));
  }
};

}  // namespace
}  // namespace analysis
}  // namespace systems
}  // namespace drake