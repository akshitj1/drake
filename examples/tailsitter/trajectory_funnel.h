#pragma once

#include <vector>

#include "drake/common/polynomial.h"
#include "drake/common/symbolic.h"
#include "drake/common/symbolic_variables.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
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

// Implements the *time-reversed* Lyapunov differential equation (eq. 5 [2]).
// When this system evaluates the contained system/cost at time t, it will
// always replace t=-t.
class LyapunovSystem : public LeafSystem<double> {
 private:
  const System<double>& system;
  const Trajectory<double>& S;
  const Trajectory<double>& K;
  const Trajectory<double>& u0;
  const Trajectory<double>& x0;
  const int num_states, num_inputs;

 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LyapunovSystem);

  LyapunovSystem(const System<double>& system_, const Trajectory<double>& S_,
                 const Trajectory<double>& K_, const Trajectory<double>& x0_,
                 const Trajectory<double>& u0_)
      : system(system_),
        S(S_),
        K(K_),
        x0(x0_),
        u0(u0_),
        num_states(system_.CreateDefaultContext()->num_total_states()),
        num_inputs(system_.get_input_port(0).size()) {
    this->DeclareContinuousState(num_states * num_states);

    // Initialize autodiff.
    // context_->SetTimeStateAndParametersFrom(context);
    // system_->FixInputPortsFrom(system, context, context_.get());
  }

  static MatrixX<double> to_block(VectorX<double>& x, const int kDim) {
    // kDim can be derived from x.size() also as square
    MatrixX<double> X(kDim, kDim);
    // kDim can be derived from x.size() also as square
    X << Eigen::Map<MatrixX<double>>(x.data(), kDim, kDim);
    return X;
  }

  static VectorX<double> to_flat(const MatrixX<double>& _X) {
    MatrixX<double> X(_X);
    return Eigen::Map<VectorX<double>>(X.data(), X.size());
  }

  static std::unique_ptr<AffineSystem<double>> Linearize(
      const System<double>& system, const VectorX<double>& x0,
      const VectorX<double>& u0) {
    auto lin_context = system.CreateDefaultContext();
    // todo: set time too?
    lin_context->SetContinuousState(x0);
    lin_context->FixInputPort(0, u0);
    auto affine_system = FirstOrderTaylorApproximation(
        system, *lin_context, InputPortSelection::kUseFirstInputIfItExists,
        OutputPortSelection::kNoOutput);
    return affine_system;
  }
  void DoCalcTimeDerivatives(
      const Context<double>& context,
      ContinuousState<double>* derivatives) const override {
    VectorX<double> p = context.get_continuous_state_vector().CopyToVector();
    const auto P = to_block(p, num_states);
    // Note: negation of time
    const double t = -context.get_time();

    auto affine_system = Linearize(system, x0.value(t), u0.value(t));

    const MatrixX<double> A =
        affine_system->A() - affine_system->B() * K.value(t);

    // todo: validate if Q is indeed S
    MatrixX<double> minus_Pdot = A.transpose() * P + P * A + S.value(t);
    // todo: do we have to negate derivative?
    derivatives->SetFromVector(to_flat(minus_Pdot));
  }
};

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
    const Variable t = prog.NewIndeterminates(1, "t")[0];
    vector<double> time_samples;
    PolynomialTrajectory V;
    get_initial_lyapunov_candidate(x_bar, t, V, time_samples);
    PolynomialTrajectory rho = get_initial_rho(time_samples, t);

    // find V_dot
    PolynomialTrajectory V_dot, rho_dot;
    for (int i = 0; i < V.size(); i++) {
      auto affine_system = LyapunovSystem::Linearize(
          system, state_nominal.value(time_samples[i]),
          input_nominal.value(time_samples[i]));
      const MatrixX<double> A =
          affine_system->A() -
          affine_system->B() * lqr_res.K.value(time_samples[i]);

      const VectorX<symbolic::Polynomial> f = A * polynomial_cast(x_bar);
      const symbolic::Polynomial V_dot_i =
          (V[i].Jacobian(x_bar) * f + V[i].Jacobian(Vector1<Variable>(t)))[0];

      V_dot.push_back(V_dot_i);
      rho_dot.push_back(rho[i].Differentiate(t));
    }
    const int max_iterations = 2;
    for (int iter = 0; iter < max_iterations; iter++) {
      PolynomialTrajectory mu = optimize_lagrange_multipliers(
          prog, x_bar, t, time_samples, V, V_dot, rho, rho_dot);
      optimize_rho(prog, x_bar, t, time_samples, V, V_dot, mu, rho, rho_dot);
    }
  }

  // sec. 3.1 [2]
  PolynomialTrajectory optimize_lagrange_multipliers(
      const MathematicalProgram& _prog, const solvers::VectorXIndeterminate& x,
      const symbolic::Variable& t, const vector<double>& time_samples,
      const PolynomialTrajectory& V, const PolynomialTrajectory& V_dot,
      const PolynomialTrajectory& rho, const PolynomialTrajectory& rho_dot) {
    // todo: assert all time segments consistent
    PolynomialTrajectory mu;
    for (int i = 0; i < V.size(); i++) {
      std::unique_ptr<MathematicalProgram> prog = _prog.Clone();
      const Variable gamma = prog->NewContinuousVariables<1>("gamma")[0];
      const int mu_degree = V_dot[i].TotalDegree();
      const symbolic::Polynomial mu_i = prog->NewFreePolynomial(
          symbolic::Variables(prog->indeterminates()), mu_degree);
      prog->AddSosConstraint(gamma -
                             (V_dot[i] - rho_dot[i] + mu_i * (rho[i] - V[i])));
      prog->AddCost(-gamma);

      const auto result = solvers::Solve(*prog);
      assert(result.is_success());
      assert(result.GetSolution(gamma) < 0);
      mu.push_back(
          symbolic::Polynomial(result.GetSolution(mu_i.ToExpression())));
    }
    return mu;
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
      prog->AddSosConstraint(
          -(V_dot[i] - rho_dot[i] + mu[i] * (rho[i] - V[i])));
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
    const double c = 3;
    const Expression rho_continuous =
        symbolic::exp(-c * (time_samples.back() - t) /
                      (time_samples.back() - time_samples.front()));

    vector<MatrixX<double>> rho_knots;
    std::transform(time_samples.begin(), time_samples.end(), rho_knots.begin(),
                   [rho_continuous, t](const double& t_break) {
                     return Vector1<double>(rho_continuous.Evaluate(
                         {symbolic::Environment{{t, t_break}}}));
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
    LyapunovSystem lyapunov(system, lqr_res.S, lqr_res.K, state_nominal,
                            input_nominal);

    // Simulator doesn't support integrating backwards in time, so simulate the
    // time-reversed Lyapunov equation from -tf to -t0, and reverse it after the
    // fact.
    Simulator<double> simulator(lyapunov);
    // todo: set this
    auto lti_context = system.CreateDefaultContext();
    lti_context->SetContinuousState(
        state_nominal.value(state_nominal.end_time()));
    lti_context->FixInputPort(0, input_nominal.value(input_nominal.end_time()));

    const MatrixX<double> P_f = RegionOfAttractionP(system, *lti_context);
    simulator.get_mutable_context().SetContinuousState(
        LyapunovSystem::to_flat(P_f));

    simulator.get_mutable_context().SetTime(-end_time);
    IntegratorBase<double>& integrator = simulator.get_mutable_integrator();
    integrator.StartDenseIntegration();

    simulator.AdvanceTo(-start_time);
    PiecewisePolynomial<double> P =
        std::move(*(integrator.StopDenseIntegration()));
    P.ReverseTime();
    P.Reshape(num_states, num_states);
    time_samples = P.get_segment_times();
    const VectorX<symbolic::Polynomial> x_bar_poly = polynomial_cast(x_bar);

    for (int i = 0; i < P.get_number_of_segments(); i++) {
      const symbolic::Polynomial V_i = x_bar_poly.dot(
          toSymbolicPolynomial(P.getPolynomialMatrix(i), t) * x_bar_poly);
      V.push_back(V_i);
    }
  }
};
// void TrajectoryFunnel(
//     const System<double>& system,
//     const trajectories::PiecewisePolynomial<double>& state_nominal,
//     const trajectories::PiecewisePolynomial<double>& input_nominal,
//     const controllers::FiniteHorizonLinearQuadraticRegulatorResult& lqr_res)
//     {
//   // can be taken as argument
//   const int num_time_samples = state_nominal.get_number_of_segments();
//   VectorX<double> time_samples(
//       num_time_samples);  // VectorX<double>::LinSpaced(num_time_samples,
//                           // state_nominal.start_time(),
//                           // state_nominal.end_time());

//   // convert system into polynomial in error coordiantes in only state
//   // variablles ie. f(x,t) = A(t)*x_bar by fixing input u=-Kx
//   trajectories::PiecewisePolynomial<double> A =
//       Linearize(system, state_nominal, input_nominal, time_samples, lqr_res);
// }

// void TimeVaryingLyaunov(const std::vector<double>& time_samples,
//                         const trajectories::PiecewisePolynomial<double>& A,
//                         const trajectories::PiecewisePolynomial<double>& Q,
//                         const MatrixX<double>& Qf){
// // returns matrix A(t) of x_dot(x_bar,t) = A(t)*x_bar
// trajectories::PiecewisePolynomial<double> Linearize(
//     const System<double>& system,
//     const trajectories::PiecewisePolynomial<double>& state_nominal,
//     const trajectories::PiecewisePolynomial<double>& input_nominal,
//     const std::vector<double>& time_samples,
//     const controllers::FiniteHorizonLinearQuadraticRegulatorResult& lqr_res)
//     {
//   std::vector<MatrixX<double>> A;
//   for (auto t_i : time_samples) {
//     auto x0 = state_nominal.value(t_i);
//     auto u0 = state_nominal.value(t_i);
//     auto lin_context = system.CreateDefaultContext();
//     // todo: set time too?
//     lin_context->SetContinuousState(x0);
//     lin_context->FixInputPort(0, u0);
//     // we can use Linearize at equilibrium points only
//     auto affine_system = systems::FirstOrderTaylorApproximation(
//         system, *lin_context, InputPortSelection::kUseFirstInputIfItExists,
//         OutputPortSelection::kNoOutput);
//     MatrixX<double> A_i =
//         affine_system->A() - affine_system->B() * lqr_res.K.value(t_i);
//     A.push_back(A_i);
//   }

//   auto A_pp = trajectories::PiecewisePolynomial<double>::FirstOrderHold(
//       time_samples, A);
//   return A_pp;
// }

}  // namespace analysis
}  // namespace systems
}  // namespace drake
