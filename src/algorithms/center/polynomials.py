import numba
import numpy as np
from collections import defaultdict
import itertools # For polynomial multiplication iteration


class Polynomial:
    """
    Represents a multivariate polynomial optimized for Hamiltonian mechanics.

    Uses a dictionary `coeffs` mapping exponent tuples to coefficients.
    Assumes variables are ordered [q1, p1, q2, p2, q3, p3, ...]
    """
    def __init__(self, data=None, n_vars=6):
        """
        Initializes the Polynomial.

        Args:
            data (dict, optional): A dictionary {exponent_tuple: coeff}. Defaults to None (empty polynomial).
                                   Exponent tuples must have length n_vars.
            n_vars (int, optional): Number of variables (dimensionality of phase space). Defaults to 6.
        """
        self.n_vars = n_vars
        # Use complex coefficients as intermediate steps in CMR often involve them
        self.coeffs = defaultdict(complex)
        if data:
            for exp, coeff in data.items():
                if len(exp) != self.n_vars:
                    raise ValueError(f"Exponent tuple {exp} length mismatch. Expected {self.n_vars}.")
                # Only store non-zero coefficients
                if not np.isclose(coeff, 0.0):
                    # Ensure exponents are integers
                    self.coeffs[tuple(map(int, exp))] = complex(coeff)

    def __repr__(self):
        return f"Polynomial({dict(self.coeffs)}, n_vars={self.n_vars})"

    def __str__(self):
        if not self.coeffs:
            return "0"
        terms = []
        # Sort terms for consistent output (e.g., by total degree, then lexicographically)
        sorted_exponents = sorted(self.coeffs.keys(), key=lambda exp: (sum(exp), exp))
        for exp in sorted_exponents:
            coeff = self.coeffs[exp]
            term_str = f"{coeff:.4e}" # Format coefficient
            var_parts = []
            for i, e in enumerate(exp):
                if e == 1:
                    var_parts.append(f"x{i}") # Placeholder variable name
                elif e > 1:
                    var_parts.append(f"x{i}^{e}")
            if var_parts:
                term_str += "*" + "*".join(var_parts)
            terms.append(term_str)
        return " + ".join(terms).replace(" + -", " - ")

    def __len__(self):
        """Number of non-zero terms."""
        return len(self.coeffs)

    def __add__(self, other):
        """Polynomial addition."""
        if isinstance(other, (int, float, complex)): # Adding a constant
             new_coeffs = self.coeffs.copy()
             zero_exp = tuple([0] * self.n_vars)
             new_coeffs[zero_exp] = new_coeffs.get(zero_exp, 0.0) + other
             # Remove if zero
             if np.isclose(new_coeffs[zero_exp], 0.0):
                 del new_coeffs[zero_exp]
             return Polynomial(new_coeffs, self.n_vars)
        elif isinstance(other, Polynomial):
            if self.n_vars != other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
            new_coeffs = self.coeffs.copy()
            # Numba *could* potentially accelerate this loop if coefficients
            # were extracted into arrays, but dict ops might be fast enough.
            for exp, coeff in other.coeffs.items():
                new_val = new_coeffs.get(exp, 0.0) + coeff
                if np.isclose(new_val, 0.0):
                    if exp in new_coeffs: # Remove if it became zero
                       del new_coeffs[exp]
                else:
                    new_coeffs[exp] = new_val
            return Polynomial(new_coeffs, self.n_vars)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Polynomial subtraction."""
        # Similar logic to __add__, negating other's coeffs
        if isinstance(other, (int, float, complex)):
            return self.__add__(-other) # Reuse add
        elif isinstance(other, Polynomial):
             if self.n_vars != other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
             new_coeffs = self.coeffs.copy()
             for exp, coeff in other.coeffs.items():
                new_val = new_coeffs.get(exp, 0.0) - coeff
                if np.isclose(new_val, 0.0):
                    if exp in new_coeffs:
                        del new_coeffs[exp]
                else:
                    new_coeffs[exp] = new_val
             return Polynomial(new_coeffs, self.n_vars)
        else:
            return NotImplemented

    def __rsub__(self, other):
        # other - self = -(self - other)
        neg_self = self * -1
        return neg_self.__add__(other)


    def __mul__(self, other):
        """Polynomial multiplication (Polynomial * Polynomial or Scalar * Polynomial)."""
        if isinstance(other, (int, float, complex)): # Scalar multiplication
            new_coeffs = defaultdict(complex)
            # Numba *could* accelerate this loop over values
            for exp, coeff in self.coeffs.items():
                new_coeffs[exp] = coeff * other
                # No need to check for zero, scalar * non-zero = non-zero (unless scalar is 0)
            if np.isclose(other, 0.0):
                 return Polynomial({}, self.n_vars) # Return empty polynomial
            return Polynomial(new_coeffs, self.n_vars)
        elif isinstance(other, Polynomial): # Polynomial * Polynomial
            if self.n_vars != other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
            new_coeffs = defaultdict(complex)
            # This nested loop is the core of polynomial multiplication
            # Numba is hard to apply directly here due to dict key generation (tuples)
            # but *could* accelerate coefficient math if refactored.
            for exp1, coeff1 in self.coeffs.items():
                for exp2, coeff2 in other.coeffs.items():
                    new_coeff = coeff1 * coeff2
                    if not np.isclose(new_coeff, 0.0):
                        new_exp = tuple(e1 + e2 for e1, e2 in zip(exp1, exp2))
                        current_coeff = new_coeffs.get(new_exp, 0.0)
                        new_val = current_coeff + new_coeff
                        if np.isclose(new_val, 0.0):
                            # If term becomes zero, remove if exists (it might not if current_coeff was 0)
                            if new_exp in new_coeffs:
                                del new_coeffs[new_exp]
                        else:
                            new_coeffs[new_exp] = new_val
            return Polynomial(new_coeffs, self.n_vars)
        else:
            return NotImplemented

    def __rmul__(self, other):
        # Handles Scalar * Polynomial
        return self.__mul__(other)

    def differentiate(self, var_index):
        """
        Differentiates the polynomial with respect to a given variable index.
        Assumes variables are 0-indexed (e.g., 0 for q1, 1 for p1, etc.).
        """
        if not (0 <= var_index < self.n_vars):
            raise ValueError(f"var_index must be between 0 and {self.n_vars-1}")

        new_coeffs = defaultdict(complex)
        # Numba unlikely to help much here due to tuple manipulation
        for exp, coeff in self.coeffs.items():
            original_exponent = exp[var_index]
            if original_exponent > 0:
                new_coeff = coeff * original_exponent
                # Check for zero coefficient *after* multiplication
                if not np.isclose(new_coeff, 0.0):
                    new_exp_list = list(exp)
                    new_exp_list[var_index] -= 1
                    new_coeffs[tuple(new_exp_list)] = new_coeff
        return Polynomial(new_coeffs, self.n_vars)

    def poisson_bracket(self, other):
        """
        Computes the Poisson bracket {self, other}.
        Assumes canonical variables ordered [q1, p1, q2, p2, ...].
        Requires n_vars to be even.
        """
        if not isinstance(other, Polynomial):
            raise TypeError("Poisson bracket requires another Polynomial.")
        if self.n_vars != other.n_vars:
            raise ValueError("Polynomials must have the same number of variables.")
        if self.n_vars % 2 != 0:
            raise ValueError("Number of variables must be even for Poisson bracket.")

        result = Polynomial({}, self.n_vars) # Start with zero polynomial

        # Loop over canonical pairs (q_i, p_i)
        # This loop itself is small, but the operations inside are polynomial ops
        num_dof = self.n_vars // 2
        for i in range(num_dof):
            qi_index = 2 * i
            pi_index = 2 * i + 1

            # Compute partial derivatives
            d_self_dqi = self.differentiate(qi_index)
            d_other_dpi = other.differentiate(pi_index)
            term1 = d_self_dqi * d_other_dpi # Uses __mul__

            d_self_dpi = self.differentiate(pi_index)
            d_other_dqi = other.differentiate(qi_index)
            term2 = d_self_dpi * d_other_dqi # Uses __mul__

            # Accumulate result: result += term1 - term2
            # These use the __add__ and __sub__ methods defined above
            result = result + term1
            result = result - term2

        return result

    def get_terms_of_degree(self, degree):
        """Returns a new Polynomial containing only terms of a specific total degree."""
        terms = {exp: coeff for exp, coeff in self.coeffs.items() if sum(exp) == degree}
        return Polynomial(terms, self.n_vars)

    def evaluate(self, values):
        """Evaluates the polynomial given a list/tuple of variable values."""
        if len(values) != self.n_vars:
            raise ValueError("Number of values must match n_vars.")
        total = 0.0
        for exp, coeff in self.coeffs.items():
            term_val = coeff
            for i, e in enumerate(exp):
                term_val *= values[i]**e
            total += term_val
        return total