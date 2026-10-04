# UDCT v7.07 — reconstructed structural-action audit

Won Shik Paik · manuscript dated 29 September 2026 (New Zealand) · reconstruction prepared 4 October 2026.

This repository provides runnable reconstructions of the five structural controls in **UDCT v7.07, Post-v7.06: structural tests of reciprocal projection, three-dimensional actions, and Solar-System limits**. This code was reconstructed from the supplied manuscript and frozen v7.06 source. It is **not the recovered historical v7.07 program**. Original numerical output and complete historical integration/planet inputs were not supplied. Newly computed reference results and discrepancies are reported below.

The five items are related first-gradient/action controls, not five validated UDCT theories. All choices were made after the historical 0/22 O8d result; these are post-hoc analytic, synthetic or conditional calculations, not preregistered observational successes.

## Files

- `UDCT_v7_07_structural_tests_reproduce.py`: runnable audit of all five controls.
- `UDCT_v7_06_Hessian_Term_Sign_Audit_reproduce.py`: unchanged frozen stellar/Galactic geometry, virial and Jeans dependency, protected by SHA-256.
- `requirements.txt`: tested numerical dependency versions.
- `reference_results/`: actual reconstructed run, JSON and Markdown tables.
- `validation.json`: comparisons with manuscript numbers; includes mismatches.
- `LICENSE.txt`: MIT license for code. The original PDF has separate rights and is not included here.

## Run

Python >=3.11 is recommended; the included run used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python UDCT_v7_07_structural_tests_reproduce.py --out results
```

For the fast analytic/orbit subset:

```bash
python UDCT_v7_07_structural_tests_reproduce.py --skip-dwarfs --out analytic_results
```

For reproducible CPU use on Linux/macOS, optionally prefix the run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. There is no random sampling, network download, parameter fitting or time evolution. Paths are relative to the working directory. `--out` creates the directory and replaces its `results.json` and `results.md`.

## Equations and scope

### 1. First-gradient QUMOND-form control

The source is div[nu(|grad U|/a0) grad U], with nu(x)=(1+x^(-2))^(1/4). The angle-averaged inward flux is integrated for density-weighted virial work; the action-Hessian tensor P is zero. Newtonian/Galactic Hessians remain in the frozen local Taylor-field geometry, but there is no constitutive double-divergence P term.

Fixed M/L=2; a0=1.082401e-10 m/s²; same three dwarfs, disk/bulge components and Plummer/tapered cusp as v7.06. Virial grids are 180×16×24 and 360×24×36; Jeans grids 450×16×24 and 900×24×36. Disk nodes are fixed at 120×64×12. The inherited radial domain is 0.002–80 r_h for Plummer and 0.002–40 r_h for cusp, with r_h and 6 r_h inserted. Boundary pressure is zero at the outer numerical radius; LOS aperture is the projected R_e cylinder. This is conditional spherical/isotropic angular-mean modelling, not a full 3-D distribution function or observational aperture fit.

### 2. Reciprocal linear massive scalar

For beta=alpha²/(1-alpha²)>=0 and lambda=sqrt(1-alpha²)/m, g/gN=1+beta(1+t)exp(-t), t=r/lambda. Its slope is -beta*t*exp(-t). The sample uses beta=1 and t=0.01,1,10. The vacuum dispersion is omega²=cB²[(1-alpha²)k²+m²]; |alpha|<1 and m²>=0 ensure positive linear omega² for nonzero modes, not nonlinear stability. Massless m=0 gives a constant G rescaling. No old 1-D feedback trajectory is rerun.

### 3. Exact planar reciprocal projection

Sigma=integral rho dz, Delta_2 B=kappa Sigma, return P†B=B(X). Thus gB=kappa M/(2pi R), vflat²=kappa M/(2pi), and the vertical return force is zero. kappa=2pi sqrt(G a0/M0), M0=10^10 solar masses. The script compares this mass exponent with v^4=GMa0 and computes exponential-disk enclosed fractions 1-(1+t)exp(-t). For a 2-D p-Laplacian, v² scales as M^(1/(p-1)) R^((p-2)/(p-1)); p=2 gives the wrong mass exponent, p=3 a rising curve. No planar PDE solver or galaxy fitting is claimed.

### 4. Polynomial 3-D scalar

F(y²)=(2/3)y³+(2/5)y⁵, mu=y+y³, y=b/a0. Spherical Gauss law gives y²+y⁴=gN/a0. The implementation evaluates y²=2q/(1+sqrt(1+4q)), q=gN/a0. Elliptic eigenvalues are y+y³ and 2y+4y³ (positive for y>0; degenerate at y=0). Deep-field scaling gives v^4=GMa0; the high-field extra scalar force gives very large retrograde perihelion shifts. This is an isolated-Sun perturbation diagnostic, not an N-body integration.

### 5. Saturated 3-D completion

F=y sqrt(1+y²)-asinh(y), mu=y/sqrt(1+y²), a*=2a0 and Gbare=Glab/2. Spherical y²/sqrt(1+y²)=q, q=gbare/a*. The numerical calculation rationalizes b-gbare to avoid subtracting nearly equal accelerations. In the strong field the residual is 4a0²/glab. The tiny radial residual addresses the polynomial completion's high-field objection; this is the pre-existing standard MOND interpolation, not established new UDCT physics or a Solar-System pass.

### Orbit integration

For inward extra acceleration b(r), Delta(varpi)=(1/(e GM_sun)) integral b(r(f)) r(f)² cos(f) df, r=a(1-e²)/(1+e cos(f)). Test 5 uses the residual b-gbare, because the inverse-square rescaling is absorbed in the measured solar GM. Composite Simpson quadrature uses 257, 1025, 4097 and 16385 nodes. Negative shifts are retrograde. Convert rad/orbit to arcsec/Julian century using 36525 days.

GM_sun=1.32712440018e20 m³/s²; AU=149597870700 m. Mercury inputs supplied by the manuscript: a=0.38709893 AU, e=0.205630, period=87.9691 days. The manuscript does not enumerate the other planets' elements. This reconstruction declares the following illustrative choices explicitly; they are not asserted to be the historical inputs:

| Planet | a (AU) | e | Period (days) |
|---|---:|---:|---:|
| Earth | 1.00000011 | 0.01671022 | 365.25636 |
| Mars | 1.52366231 | 0.09341233 | 686.97959 |
| Saturn | 9.53707032 | 0.05415060 | 10759.22 |

## Numerical comparison with the manuscript

| Quantity | Manuscript | Reconstruction | Assessment |
|---|---:|---:|---|
| Leo IV Plummer V/V_N | 13.935 | 13.979732 | mismatch |
| Pegasus III Plummer V/V_N | 19.321 | 19.348459 | mismatch |
| Hydrus I Plummer V/V_N | 3.626 | 3.880819 | mismatch |
| Leo IV cusp V/V_N | 13.861 | 13.861157 | agrees at printed precision |
| Pegasus III cusp V/V_N | 19.192 | 19.192335 | agrees at printed precision |
| Hydrus I cusp V/V_N | 3.423 | 3.422871 | agrees at printed precision |
| Mercury polynomial shift (arcsec/century) | -149.972 | -149.971613 | agrees at printed precision |
| Earth polynomial shift | -152.609 | -152.609154 | agrees at printed precision |
| Saturn polynomial shift | -153.426 | -152.349998 | mismatch; historical inputs unspecified |
| Mercury saturated shift | -3.185e-8 | -3.184806e-8 | agrees at printed precision |
| Earth saturated shift | -3.454e-7 | -3.453918e-7 | agrees at printed precision |
| Mars saturated shift | -9.877e-7 | -9.876300e-7 | small mismatch; historical inputs unspecified |

All six LOS dispersions reproduce the printed three decimals. Linear-field and planar-table values reproduce the printed precision. Orbit quadrature agrees across all four node counts to floating-point accuracy. The largest cusp coarse/fine virial difference is recorded in `validation.json`; the LOS coarse/fine difference is also recorded.

The Plummer discrepancy has not been resolved. Possible differences in historical radial domains or prescriptions cannot be established from the supplied record. The local Taylor approximation at large radius also limits the interpretation of a long Plummer tail. We retain the declared inherited domain, disclose the mismatch and do not tune a cutoff to the desired values. Original code, integration settings and planetary inputs are needed to establish exact historical reproduction. Convergence under the declared settings is not proof that these settings match the original calculation.

## Interpretation

These controls do not validate UDCT or MOND as a whole. Positive virial work is a necessary static diagnostic, not observational agreement. The archived dispersions are unmatched reference values; no likelihood or observational rejection significance is inferred. The v7.06 obstruction is not erased by these different constitutive laws. Reciprocal coupling does not define a unique physical bidirectional projection. Dynamic back-reaction, time evolution, external-field Solar-System quadrupoles, relativistic lensing, and global conservation/DF tests have not been computed. No new claim about the v7.09 outcome is derived here.

## Archival release guidance

Keep the manuscript version **v7.07** and its 29 September 2026 preparation date, but label this implementation as a **4 October 2026 reconstruction**. Describe the upload as retrospective/post-hoc structural tests; include both negative controls and the qualified saturated completion. Upload the original PDF unchanged alongside this package if its rights permit. Do not present mismatched numerical tables as exactly reproduced. The v7.09 experiment is distinct, so a separate archival record is clearer than treating this as its newer result.

Zenodo publication date should reflect first public publication; manuscript preparation date and code reconstruction date should be separate description fields. If there was no previous public publication, use the actual first publication date. Zenodo also tracks repository upload dates. See https://help.zenodo.org/docs/deposit/describe-records/publication-date/ .

## Provenance and references

1. Paik, W. S. (2026), supplied v7.07 PDF, 29 September. Original PDF SHA-256: 12c6bb57e03a41f09bd3959c8198a4840544a56b37f975fd1565f40d09358191.
2. Supplied v7.06 frozen dependency: SHA-256 a64440e1350ae8bf10eb3188ec34829b6d586abf275de5cdc841a50ad3478284. Related record: https://doi.org/10.5281/zenodo.23039754 .
3. Geometry reconstruction was adapted from the supplied v7.06/v7.09 sources; use of later code as a reconstruction aid is disclosed and does not backdate this software. v7.09: https://doi.org/10.5281/zenodo.23120660 .
4. Milgrom (2010), QUMOND: https://doi.org/10.1111/j.1365-2966.2009.16184.x .

Prepared with assistance from OpenAI ChatGPT/Codex. Author should review equations, provenance and release metadata before publication. MIT code license does not imply that the manuscript is MIT-licensed.
