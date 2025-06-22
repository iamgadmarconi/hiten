from typing import List

import numpy as np

from hiten.algorithms.polynomial.base import decode_multiindex


def _monomial_to_string(exps: tuple[int, ...]) -> str:
    """
    Convert a tuple of exponents to a formatted monomial string.
    
    Parameters
    ----------
    exps : tuple[int, ...]
        Tuple of exponents for each variable (q1, q2, q3, p1, p2, p3)
        
    Returns
    -------
    str
        Formatted string representation of the monomial
        
    Notes
    -----
    For each variable with non-zero exponent:
    - If exponent is 1, only the variable name is included
    - If exponent is greater than 1, the variable and exponent are included
    - Variables are separated by spaces
    - If all exponents are zero, returns "1"
    
    Example: (1, 2, 0, 0, 0, 3) becomes "q1 q2^2 p3^3"
    """
    out: list[str] = []
    names = ("q1", "q2", "q3", "p1", "p2", "p3")
    for e, name in zip(exps, names):
        if e == 0:
            continue
        if e == 1:
            out.append(name)
        else:
            out.append(f"{name}^{e}")
    return " ".join(out) if out else "1"


def _fmt_coeff(c: complex, width: int = 25) -> str:
    """
    Format a complex coefficient as a right-justified string.
    
    Parameters
    ----------
    c : complex
        Complex coefficient to format
    width : int, optional
        Width of the resulting string, default is 25
        
    Returns
    -------
    str
        Formatted string representation of the complex coefficient
        
    Notes
    -----
    Three different formats are used:
    - Real numbers (|imag| < 1e-14): " <real>"
    - Pure imaginary (|real| < 1e-14): " <imag>i"
    - Complex: " <real>+<imag>i"
    
    All numbers use scientific notation with 16 digits of precision.
    The result is right-justified to the specified width.
    """
    s: str
    if abs(c.imag) < 1e-14:  # Effectively real
        s = f"{c.real: .16e}"
    elif abs(c.real) < 1e-14:  # Effectively pure imaginary
        # Format as " <num>i", e.g., " 1.23...e+00i"
        imag_s = f"{c.imag: .16e}"
        s = f"{imag_s.strip()}i" # Use strip() to handle potential leading/trailing spaces from imag_s before adding 'i'
    else:  # Truly complex
        # Format as "<real>+<imag>i", e.g., " 1.23e+00-4.56e-01i"
        # This will likely be much longer than 'width'.
        s = f"{c.real: .16e}{c.imag:+.16e}i" # Note: space before c.real part, '+' for imag sign
    
    return s.rjust(width)


def format_cm_table(poly_cm_cn: List[np.ndarray], clmo: np.ndarray) -> str:
    """
    Create a formatted table of center manifold Hamiltonian coefficients.
    
    Parameters
    ----------
    poly_cm_cn : List[numpy.ndarray]
        List of coefficient arrays for center manifold Hamiltonian in complex normal form
    clmo : numpy.ndarray
        List of arrays containing packed multi-indices
        
    Returns
    -------
    str
        Formatted string table of Hamiltonian coefficients
        
    Notes
    -----
    The table displays coefficients of the center manifold Hamiltonian organized by:
    - Exponents of q2, p2, q3, p3 (k1, k2, k3, k4)
    - Terms with q1 or p1 are excluded
    - Terms with coefficients smaller than 1e-14 are excluded
    - Terms are sorted by degree and within each degree by a predefined order
    
    The table has a two-column layout, with headers:
    "k1  k2  k3  k4  hk    k1  k2  k3  k4  hk"
    
    Each row shows the exponents (k1, k2, k3, k4) and the corresponding coefficient (hk)
    in scientific notation.
    """
    structured_terms: list[tuple[int, tuple[int, int, int, int], complex]] = []
    
    k_col_width = 2
    hk_col_width = 25
    k_spacing = "  "

    MIN_DEG_TO_DISPLAY = 2
    MAX_DEG_TO_DISPLAY = 5 # As per original image and previous MAX_DEG usage

    for deg in range(MIN_DEG_TO_DISPLAY, MAX_DEG_TO_DISPLAY + 1):
        if deg >= len(poly_cm_cn) or not poly_cm_cn[deg].any():
            continue
        
        coeff_vec = poly_cm_cn[deg]

        for pos, c_val_complex in enumerate(coeff_vec):
            # Ensure c_val is treated as a number; np.isscalar checks for single numpy values
            c_val = np.complex128(c_val_complex) # Ensure it's a Python/Numpy complex
            if not (isinstance(c_val, (int, float, complex)) or np.isscalar(c_val)):
                continue
            if abs(c_val) <= 1e-14: # Skip zero coefficients
                continue
            
            k_exps = decode_multiindex(pos, deg, clmo) 
            
            if k_exps[0] != 0 or k_exps[3] != 0:  # Skip terms involving q1 or p1
                continue

            k1_q2 = k_exps[1]  # exponent of q2
            k2_p2 = k_exps[4]  # exponent of p2
            k3_q3 = k_exps[2]  # exponent of q3
            k4_p3 = k_exps[5]  # exponent of p3
            
            current_k_tuple = (k1_q2, k2_p2, k3_q3, k4_p3)
            structured_terms.append((deg, current_k_tuple, c_val))

    # Define the desired sort order based on the image
    desired_k_tuple_order_by_degree = {
        2: [(2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2)],
        3: [(2,1,0,0), (0,3,0,0), (0,1,2,0), (0,0,1,2)],
        4: [(4,0,0,0), (2,2,0,0), (0,4,0,0), (2,0,2,0), (0,2,2,0), 
            (0,0,4,0), (1,1,1,1), (2,0,0,2), (0,2,0,2), (0,0,2,2)],
        5: [(4,1,0,0), (2,3,0,0), (0,5,0,0), (2,1,2,0), (0,3,2,0), 
            (0,1,4,0), (3,0,1,1), (1,2,1,1), (1,0,3,1), (2,1,0,2), 
            (0,3,0,2), (0,1,2,2), (1,0,1,3), (0,1,0,4)]
    }

    def sort_key(term_data):
        term_deg = term_data[0]
        term_k_tuple = term_data[1]
        
        order_list_for_degree = desired_k_tuple_order_by_degree.get(term_deg, [])
        try:
            k_tuple_sort_order = order_list_for_degree.index(term_k_tuple)
        except ValueError:
            k_tuple_sort_order = float('inf') # Place unknown tuples at the end of their degree group
        return (term_deg, k_tuple_sort_order)

    structured_terms.sort(key=sort_key)

    data_lines: list[str] = []
    for term_deg, k_tuple, c_val_sorted in structured_terms:
        k1_q2, k2_p2, k3_q3, k4_p3 = k_tuple
        formatted_hk = _fmt_coeff(c_val_sorted, width=hk_col_width)
        line = (
            f"{k1_q2:<{k_col_width}d}{k_spacing}"
            f"{k2_p2:<{k_col_width}d}{k_spacing}"
            f"{k3_q3:<{k_col_width}d}{k_spacing}"
            f"{k4_p3:<{k_col_width}d}{k_spacing}"
            f"{formatted_hk}"
        )
        data_lines.append(line)

    # Header for one block of the table
    header_part = (
        f"{'k1':>{k_col_width}s}{k_spacing}"
        f"{'k2':>{k_col_width}s}{k_spacing}"
        f"{'k3':>{k_col_width}s}{k_spacing}"
        f"{'k4':>{k_col_width}s}{k_spacing}"
        f"{'hk':>{hk_col_width}s}"
    )
    block_separator = "    "  # Four spaces between the two table blocks
    full_header_line = header_part + block_separator + header_part

    if not data_lines:
        return full_header_line + "\n(No data to display)"

    num_total_lines = len(data_lines)
    # Ensure num_left_lines is at least 0, even if num_total_lines is 0
    num_left_lines = (num_total_lines + 1) // 2 if num_total_lines > 0 else 0
    
    output_table_lines = [full_header_line]
    len_one_data_block = len(header_part)

    for i in range(num_left_lines):
        left_data_part = data_lines[i]
        
        right_data_idx = i + num_left_lines
        if right_data_idx < num_total_lines:
            right_data_part = data_lines[right_data_idx]
        else:
            # Fill with spaces if no corresponding right-side data
            right_data_part = " " * len_one_data_block 
        
        output_table_lines.append(left_data_part + block_separator + right_data_part)
        
    return "\n".join(output_table_lines)