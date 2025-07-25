from hiten.algorithms.hamiltonian.center._lie import \
    _lie_transform as _lie_transform_partial
from hiten.algorithms.hamiltonian.normal._lie import \
    _lie_transform as _lie_transform_full
from hiten.algorithms.hamiltonian.transforms import (
    _polylocal2realmodal, _polyrealmodal2local,
    _restrict_poly_to_center_manifold, _substitute_complex, _substitute_real)
from hiten.system.hamiltonians.base import Hamiltonian, register_conversion
from hiten.system.libration.collinear import CollinearPoint
from hiten.system.libration.triangular import TriangularPoint


@register_conversion("physical", "real_modal", 
                    required_context=["point"],
                    default_params={"tol": 1e-12})
def _physical_to_real_modal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    tol = kwargs.get("tol", 1e-12)
    new_poly = _polylocal2realmodal(point, ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="real_modal")


@register_conversion("real_modal", "physical", 
                    required_context=["point"],
                    default_params={"tol": 1e-12})
def _real_modal_to_physical(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    tol = kwargs.get("tol", 1e-12)
    new_poly = _polyrealmodal2local(point, ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="physical")


@register_conversion("real_modal", "complex_modal", 
                    required_context=["point"],
                    default_params={"tol": 1e-12})
def _real_modal_to_complex_modal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-12)
    new_poly = _substitute_complex(ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="complex_modal")


@register_conversion("complex_modal", "real_modal", 
                    required_context=["point"],
                    default_params={"tol": 1e-12})
def _complex_modal_to_real_modal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-12)
    new_poly = _substitute_real(ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="real_modal")


@register_conversion("complex_modal", "complex_partial_normal", 
                    required_context=["point"],
                    default_params={"tol_lie": 1e-30})
def _complex_modal_to_complex_partial_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    tol_lie = kwargs.get("tol_lie", 1e-30)
    # This returns (poly_trans, poly_G_total, poly_elim_total)
    new_poly, _, _ = _lie_transform_partial(point, ham.poly_H, ham._psi, ham._clmo, ham.max_degree, tol=tol_lie)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="complex_partial_normal")


@register_conversion("complex_partial_normal", "real_partial_normal", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _complex_partial_normal_to_real_partial_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_real(ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol, mix_pairs=mix_pairs)
    
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="real_partial_normal")


@register_conversion("real_partial_normal", "complex_partial_normal", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _real_partial_normal_to_complex_partial_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_complex(ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="complex_partial_normal")


@register_conversion("complex_partial_normal", "center_manifold_complex", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _complex_partial_normal_to_center_manifold_complex(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    tol = kwargs.get("tol", 1e-14)
    new_poly = _restrict_poly_to_center_manifold(point, ham.poly_H, ham._clmo, tol=tol)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="center_manifold_complex")


@register_conversion("center_manifold_complex", "center_manifold_real", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _center_manifold_complex_to_center_manifold_real(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_real(ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="center_manifold_real")


@register_conversion("center_manifold_real", "center_manifold_complex", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _center_manifold_real_to_center_manifold_complex(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_complex(ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="center_manifold_complex")


@register_conversion("complex_modal", "complex_full_normal", 
                    required_context=["point"],
                    default_params={"tol_lie": 1e-30, "resonance_tol": 1e-14})
def _complex_modal_to_complex_full_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    tol_lie = kwargs.get("tol_lie", 1e-30)
    resonance_tol = kwargs.get("resonance_tol", 1e-14)
    new_poly, _, _ = _lie_transform_full(point, ham.poly_H, ham._psi, ham._clmo, ham.max_degree, tol=tol_lie, resonance_tol=resonance_tol)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="complex_full_normal")


@register_conversion("complex_full_normal", "real_full_normal", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _complex_full_normal_to_real_full_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_real(ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="real_full_normal")


@register_conversion("real_full_normal", "complex_full_normal", 
                    required_context=["point"],
                    default_params={"tol": 1e-14})
def _real_full_normal_to_complex_full_normal(ham: Hamiltonian, **kwargs) -> Hamiltonian:
    point = kwargs["point"]
    if isinstance(point, CollinearPoint):
        mix_pairs = (1, 2)
    elif isinstance(point, TriangularPoint):
        mix_pairs = (0, 1, 2)

    tol = kwargs.get("tol", 1e-14)
    new_poly = _substitute_complex(ham.poly_H, ham.max_degree, ham._psi, ham._clmo, tol=tol, mix_pairs=mix_pairs)
    return Hamiltonian(new_poly, ham.max_degree, ham._ndof, name="complex_full_normal")