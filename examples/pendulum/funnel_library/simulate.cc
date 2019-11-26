#include <map>
#include <gflags/gflags.h>
#include "drake/common/drake_assert.h"
#include "drake/examples/pendulum/funnel_library/dircol_optimize.h"
#include "drake/examples/pendulum/funnel_library/lti_roa.h"
#include "drake/examples/pendulum/funnel_library/tvlqr/tvlqr.h"
#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/solvers/mathematical_program.h"

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

class sosFeasible {
  const solvers::MathematicalProgram& prog;
  const Polynomial &t, J, J_dot, h1, h2, h3;
  const double &t_k, t_kplus1;

 public:
  sosFeasible(const solvers::MathematicalProgram& prog, const Polynomial& t,
              const Polynomial& J, const Polynomial& J_dot,
              const Polynomial& h1, const Polynomial& h2, const Polynomial& h3,
              const double& t_k, const double& t_kplus1)
      : prog(prog),
        t(t),
        J(J),
        J_dot(J_dot),
        h1(h1),
        h2(h2),
        h3(h3),
        t_k(t_k),
        t_kplus1(t_kplus1) {}

  bool operator()(const double& rho) {
    auto _prog = prog.Clone();
    const double rho_dot = 0;
    _prog->AddSosConstraint(-((J_dot - rho_dot) + h1 * (rho - J) +
                              h2 * (t - t_k) + h3 * (t_kplus1 - t)));
    return solvers::Solve(*_prog).is_success();
  }
};

double line_search(const std::function<bool(const double&)>& isFeasible, double lb,
                   double ub, const double kPrec = 0.1) {
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

  // #breaks + rho_final
  std::vector<double> alpha_t(x_opt.get_number_of_segments() + 1);
  alpha_t[x_opt.get_number_of_segments()] = 10.3;  // LtiRegionOfAttraction();

  for (int k = x_opt.get_number_of_segments() - 1; k >= 0; k--) {
    solvers::MathematicalProgram prog;
    const VectorX<Variable> xvar{prog.NewIndeterminates<2>(
        std::array<std::string, 2>{"theta", "thetadot"})};
    const Variable tvar{
        prog.NewIndeterminates<1>(std::array<std::string, 1>{"t"})(0)};
    const Variables indeterminates(prog.indeterminates());

    double t_k = breaks[k];
    VectorX<Polynomial> x0 = toSymbolicPoly(x_opt.getPolynomialMatrix(k), tvar);
    VectorX<Polynomial> u0 = toSymbolicPoly(u_opt.getPolynomialMatrix(k), tvar);
    // all element polynomials have variable name "t"
    auto S = toSymbolicPoly(S_t.getPolynomialMatrix(k), tvar);

    auto K = toSymbolicPoly(K_t.getPolynomialMatrix(k), tvar);
    // SOS
    const Polynomial t(tvar, indeterminates);
    const VectorX<Polynomial> x = xvar.unaryExpr(
        [indeterminates](Variable x) { return Polynomial(x, indeterminates); });

    const MatrixX<Polynomial> S_dot =
        S.unaryExpr([tvar](Polynomial x) { return x.Differentiate(tvar); });

    PendulumPlant<Expression> _pendulum;
    auto context = _pendulum.CreateDefaultContext();
    // Extract the polynomial dynamics.
    context->get_mutable_continuous_state_vector().SetFromVector(
        (x0+x).cast<Expression>());
    context->FixInputPort(0, (u0 - K * (x)).cast<Expression>());
    auto x_dot = _pendulum.AllocateTimeDerivatives();
    // pendulum params default are same as req. values
    _pendulum.CalcTimeDerivatives(*context, x_dot.get());

    // Define the Lyapunov function.
    VectorX<double> x0_val = x_opt.value(t_k);
    // todo: do we need time?
    const Environment poly_approx_env{
        {xvar(0), x0_val(0)}, {xvar(1), x0_val(1)}, {tvar, t_k}};

    const Polynomial theta_ddot_approx(
        symbolic::TaylorExpand(x_dot->CopyToVector()[1], poly_approx_env, 3,
                               false),
        indeterminates);

    VectorX<Polynomial> x_dot_approx(x.size());
    x_dot_approx << Polynomial(x_dot->CopyToVector()[0], indeterminates),
        theta_ddot_approx;

    Polynomial J = (x.transpose() * S * x)(0);
    Polynomial J_dot = (x.transpose() * S_dot * x +
                        Polynomial(2) * x.transpose() * S * x_dot_approx)(0);

    const Polynomial h1 = prog.NewFreePolynomial(indeterminates, 2);
    const Polynomial h2 = prog.NewSosPolynomial(indeterminates, 2).first;
    const Polynomial h3 = prog.NewSosPolynomial(indeterminates, 2).first;

    // Variable rhovar("rho");
    // Polynomial rho(rhovar, indeterminates);
    // prog.AddDecisionVariables(
    //     (solvers::VectorXDecisionVariable(1) << rhovar).finished());
    // const int rho_dot = 0;

    auto t_kplus1 = breaks[k + 1];
    // todo: fix this
    // auto isFeasible = [_prog=prog.Clones, t, J, J_dot, h1, h2, h3, t_k,
    //                    t_kplus1](const double& rho) {
    //   return solvers::Solve(*_prog).is_success();
    // };
    alpha_t[k] =
        line_search(sosFeasible(prog, t, J, J_dot, h1, h2, h3, t_k, t_kplus1),
                    0.0, alpha_t[k + 1]);
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