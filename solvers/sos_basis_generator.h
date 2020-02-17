#pragma once

#include <vector>

#include <Eigen/Core>

#include "drake/common/symbolic.h"

namespace drake {
namespace solvers {

/**
  * Given input polynomial p, outputs a set M of monomials with the following
  * guarantee: if p = f1*f1 + f2*f2 + ... + fn*fn for some (unknown) polynomials
  * f1, f2, ..., fn, then the span of M contains f1, f2, ..., fn,  Given M, one
  * can then find the polynomials fi using semidefinite programming; see,
  * e.g., Chapter 3 of Semidefinite Optimization and Convex Algebraic Geometry
  * by G. Blekherman, P. Parrilo, R. Thomas.
  * @param p A polynomial
  * @return A vector whose entries are the elements of M
*/
drake::VectorX<symbolic::Monomial> ConstructMonomialBasis(
    const drake::symbolic::Polynomial& p);
}  // namespace solvers
}  // namespace drake

// Returns a list of all exponents that appear in a polynomial p.
// E.g., given p = 1 + 2x₀² + 3x₀*x₁², returns [0, 0; 2, 0; 1, 2];
Eigen::Matrix<int, -1, -1, Eigen::RowMajor> GetPolynomialExponents(
    const drake::symbolic::Polynomial& p);