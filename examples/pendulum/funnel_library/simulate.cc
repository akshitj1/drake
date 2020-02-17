#include <map>
#include <gflags/gflags.h>
#include "drake/common/drake_assert.h"
#include "drake/examples/pendulum/funnel_library/dircol_optimize.h"
#include "drake/examples/pendulum/funnel_library/lti_roa.h"
#include "drake/examples/pendulum/funnel_library/tvlqr/tvlqr.h"
#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/sos_basis_generator.h"

namespace drake {
namespace examples {
namespace pendulum {
namespace analysis {
std::unique_ptr<systems::controllers::TimeVaryingLQR> StabilizingLQRController(
    const PendulumPlant<double>* pendulum, const PPoly& x_des,
    const PPoly& u_des) {
  const VectorX<double> kXf_err_max{(VectorX<double>(2) << 10, 1).finished()};
  const MatrixX<double> Qf{
      kXf_err_max.array().square().inverse().matrix().asDiagonal()};
  const MatrixX<double> Q{
      (VectorX<double>(2) << 10, 1).finished().asDiagonal()};
  const MatrixX<double> R{(MatrixX<double>(1, 1) << 15).finished()};

  return std::make_unique<systems::controllers::TimeVaryingLQR>(
      *pendulum, x_des, u_des, Q, R, Qf);
}

using symbolic::Environment;
using symbolic::Expression;
using symbolic::Polynomial;
using symbolic::Variable;
using symbolic::Variables;

typedef std::unordered_map<Variable, int> Exponents;

Exponents PolynomialMaxExponents(const drake::symbolic::Polynomial& p,
                                 const Variables& indeterminates) {
  // we take indeterminates as parameters and not the member of p as we want all
  // indeterminates
  Exponents var_exps;
  for (auto var : indeterminates) {
    var_exps.insert(std::pair<Variable, int>(var, 0));
  }
  for (const auto& m : p.monomial_to_coefficient_map()) {
    for (const auto& var : indeterminates) {
      var_exps.at(var) = std::max(var_exps.at(var), m.first.degree(var));
    }
  }
  return var_exps;
}

class sosFeasible {
  const solvers::MathematicalProgram& prog;
  const Polynomial &t, J, J_dot;
  const double &t_k, t_kplus1;

 public:
  sosFeasible(const solvers::MathematicalProgram& prog, const Polynomial& t,
              const Polynomial& J, const Polynomial& J_dot, const double& t_k,
              const double& t_kplus1)
      : prog(prog), t(t), J(J), J_dot(J_dot), t_k(t_k), t_kplus1(t_kplus1) {}

  bool operator()(const double& rho) {
    log()->info("checking feasibility at rho: {}", rho);
    auto _prog = prog.Clone();
    const double rho_dot = 0;
    Variables vars(_prog->indeterminates());

    // max_exps are dictated by J_dot and J
    Exponents max_exps = PolynomialMaxExponents(J_dot + J, vars);
    Exponents J_exps = PolynomialMaxExponents(J, vars);
    auto map_diff = [vars](Exponents a, Exponents b) {
      Exponents exp_diff;
      for (auto var : vars)
        exp_diff.insert(std::pair<Variable, int>(var, a.at(var) - b.at(var)));
      return exp_diff;
    };
    auto h1_exps = map_diff(max_exps, J_exps);
    int h1_deg = 0;
    for (auto exp : h1_exps) {
      h1_deg += exp.second;
    }

    const int h2_deg = 2, h3_deg = h2_deg;  //(J + J_dot).TotalDegree() - 1,
                                            // h3_deg=h2_deg;  // excluding t^1

    const Polynomial h1 = _prog->NewFreePolynomial(vars, h1_deg);
    const Polynomial h2 = _prog->NewSosPolynomial(vars, h2_deg).first;
    const Polynomial h3 = _prog->NewSosPolynomial(vars, h3_deg).first;

    _prog->AddSosConstraint(-((J_dot - rho_dot) + h1 * (rho - J) +
                              h2 * (t - t_k) + h3 * (t_kplus1 - t)));
    return solvers::Solve(*_prog).is_success();
  }
};

double line_search(const std::function<bool(const double&)>& isFeasible,
                   double lb, double ub, const double kPrec = 0.1) {
  double rho = (lb + ub) / 2;
  if (isFeasible(rho)) {
    lb = rho;
    if (ub - lb <= kPrec) return lb;
  } else
    ub = rho;

  if (ub - lb <= kPrec) throw "no feasible solution exists in given range";
  return line_search(isFeasible, lb, ub, kPrec);
}

symbolic::Polynomial toSymbolicPoly(const Polynomiald& p, const Variable& t) {
  symbolic::Polynomial p_exp;
  for (auto m : p.GetMonomials()) {
    // we are dealing only with t1
    const unsigned int t1 = Polynomiald::VariableNameToId("t");
    DRAKE_ASSERT(m.terms.size() < 2);
    symbolic::Monomial m_exp;
    if (m.terms.size() == 0)
      m_exp = symbolic::Monomial();
    else {
      assert(m.terms[0].var == t1);
      auto term = m.terms[0];
      m_exp = symbolic::Monomial(t, term.power);
    }
    p_exp.AddProduct(m.coefficient, m_exp);
  }
  return p_exp;
}
MatrixX<symbolic::Polynomial> toSymbolicPoly(const MatrixX<Polynomiald>& P,
                                             const Variable& t) {
  auto toPoly = [t](const Polynomiald& x) { return toSymbolicPoly(x, t); };
  return P.unaryExpr(toPoly);
}

std::vector<double> ltv_roa(PendulumPlant<double>& pendulum, const PPoly& x_opt,
                            const PPoly& u_opt) {
  auto lqr = StabilizingLQRController(&pendulum, x_opt, u_opt);
  PPoly S_t, K_t;
  lqr->getSKTrajectory(S_t, K_t);
  std::vector<double> breaks = x_opt.get_segment_times();
  const int N_breaks = breaks.size();

  // #breaks + rho_final
  std::vector<double> alpha_t(N_breaks, 0);
  alpha_t[N_breaks - 1] = 10.2734;  // LtiRegionOfAttraction();

  // start from second last rho is already set to rho_lti
  for (int k = N_breaks - 2; k >= 0; k--) {
    solvers::MathematicalProgram prog;
    const VectorX<Variable> xvar{prog.NewIndeterminates<2>(
        std::array<std::string, 2>{"theta", "thetadot"})};

    const Variable tvar{
        prog.NewIndeterminates<1>(std::array<std::string, 1>{"t"})(0)};
    const Variables indeterminates(prog.indeterminates());

    double t_k = breaks[k], t_kplus1 = breaks[k + 1];
    VectorX<Polynomial> x0 = toSymbolicPoly(x_opt.getPolynomialMatrix(k), tvar);
    VectorX<Polynomial> u0 = toSymbolicPoly(u_opt.getPolynomialMatrix(k), tvar);
    auto S = toSymbolicPoly(S_t.getPolynomialMatrix(k), tvar);
    auto K = toSymbolicPoly(K_t.getPolynomialMatrix(k), tvar);
    // SOS
    const Polynomial t(tvar, indeterminates);
    const VectorX<Polynomial> x = xvar.unaryExpr(
        [indeterminates](Variable x) { return Polynomial(x, indeterminates); });

    PendulumPlant<Expression> _pendulum;
    auto context = _pendulum.CreateDefaultContext();
    context->get_mutable_continuous_state_vector().SetFromVector(
        (x0 + x).cast<Expression>());
    context->FixInputPort(0, (u0 - K * x).cast<Expression>());
    auto x_dot = _pendulum.AllocateTimeDerivatives();
    _pendulum.CalcTimeDerivatives(*context, x_dot.get());

    const Environment poly_approx_env{{xvar(0), 0}, {xvar(1), 0}, {tvar, t_k}};

    const Polynomial theta_ddot_approx(
        symbolic::TaylorExpand(x_dot->CopyToVector()[1], poly_approx_env, 3),
        indeterminates);

    VectorX<Polynomial> x_dot_approx(x.size());
    x_dot_approx << Polynomial(x_dot->CopyToVector()[0], indeterminates),
        theta_ddot_approx;

    const MatrixX<Polynomial> S_dot =
        S.unaryExpr([tvar](Polynomial x) { return x.Differentiate(tvar); });

    Polynomial J = (x.transpose() * S * x)(0);
    Polynomial J_dot = (x.transpose() * S_dot * x +
                        Polynomial(2) * x.transpose() * S * x_dot_approx)(0);

    // Variable rhovar("rho");
    // Polynomial rho(rhovar, indeterminates);
    // prog.AddDecisionVariables(
    //     (solvers::VectorXDecisionVariable(1) << rhovar).finished());
    // const int rho_dot = 0;
    // todo: fix this
    // auto isFeasible = [_prog=prog.Clones, t, J, J_dot, h1, h2, h3, t_k,
    //                    t_kplus1](const double& rho) {
    //   return solvers::Solve(*_prog).is_success();
    // };
    alpha_t[k] = line_search(sosFeasible(prog, t, J, J_dot, t_k, t_kplus1), 0.0,
                             alpha_t[k + 1]);
  }
  return alpha_t;
}

void simulate() {
  PendulumPlant<double> pendulum;
  TrajPair trajs_opt = optimize_trajectory_dircol();
  PPoly x_opt = trajs_opt.first;
  PPoly u_opt = trajs_opt.second;

  auto rho_t = ltv_roa(pendulum, x_opt, u_opt);

  std::cout << "rho: ";
  for (auto rho : rho_t) std::cout << rho << " ";
  std::cout << std::endl;
}
}  // namespace analysis
}  // namespace pendulum
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Region of attraction for Time varying LQR controller");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::set_log_level("info");
  drake::examples::pendulum::analysis::simulate();
  return 0;
}