#!/usr/bin/env python3
"""Reconstructed v7.07 structural audit; not the unavailable historical code.
Copyright 2026 Won Shik Paik. MIT license. See README.md for scope/provenance.
"""
import argparse, hashlib, json, platform
from pathlib import Path
import scipy
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson
import UDCT_v7_06_Hessian_Term_Sign_Audit_reproduce as V706
G, MSUN, PC, A0, ML = V706.G, V706.MSUN, V706.PC, V706.A0, V706.ML
GM = 1.32712440018e20
AU = 149597870700.0
# Mercury supplied by manuscript; other elements are explicitly reconstructed.
PLANETS = {"Mercury": (.38709893,.205630,87.9691),
           "Earth": (1.00000011,.01671022,365.25636),
           "Mars": (1.52366231,.09341233,686.97959),
           "Saturn": (9.53707032,.05415060,10759.22)}

# Frozen stellar/Galactic geometry; adapted from the supplied v7.06/v7.09 sources.
def geometry(obj, profile, ratios, e, h, grid):
    mass = ML * obj.luminosity * MSUN
    re = obj.re_pc * PC
    if profile == "Plummer":
        b = re
        rh = b / np.sqrt(2 ** (2 / 3) - 1)
    elif profile == "Tapered cusp":
        _, a_re, rh_re, _, _ = V706.cusp_calibration()
        b, rh = a_re * re, rh_re * re
    else:
        raise ValueError(profile)
    r = np.asarray(ratios) * rh
    polar, wpolar = leggauss(grid.polar)
    az = 2 * np.pi * np.arange(grid.azimuth) / grid.azimuth
    n = np.stack((np.repeat(np.sqrt(1 - polar ** 2), grid.azimuth) *
                  np.tile(np.cos(az), grid.polar),
                  np.repeat(np.sqrt(1 - polar ** 2), grid.azimuth) *
                  np.tile(np.sin(az), grid.polar),
                  np.repeat(polar, grid.azimuth)), axis=1)
    weights = np.repeat(wpolar, grid.azimuth) / (2 * grid.azimuth)
    ne, nhn = n @ e, np.einsum("ki,ij,kj->k", n, h, n)
    hn = n @ h.T
    ehn, hn2 = hn @ e, np.einsum("ki,ki->k", hn, hn)

    if profile == "Plummer":
        gn = G * mass * r / (r * r + b * b) ** 1.5
        lr = G * mass * (b * b - 2 * r * r) / (r * r + b * b) ** 2.5
        lt = G * mass / (r * r + b * b) ** 1.5
        rho = (1 + (r / b) ** 2) ** -2.5
        dlogrho = -5 * r / (r * r + b * b)
    else:
        q, a_re, _, cum, total = V706.cusp_calibration()
        rq = r / re
        rho_q = (a_re / rq) / (1 + rq / a_re) ** 3 * np.exp(-(rq / 10) ** 4)
        rho_mass = mass * rho_q / (4 * np.pi * re ** 3 * total)
        contained = np.interp(rq, q, cum, left=0, right=1)
        gn = G * mass * contained / r ** 2
        lt = gn / r
        lr = 4 * np.pi * G * rho_mass - 2 * lt
        rho = (b / r) / (1 + r / b) ** 3 * np.exp(-(r / (10 * re)) ** 4)
        dlogrho = -1 / r - 3 / (r + b) - 4 * r ** 3 / (10 * re) ** 4

    # s = psi,ij psi,ij  (full Hessian: stellar part + Galactic tide)
    t2 = (lr[:, None] ** 2 + 2 * lt[:, None] ** 2 +
          2 * lt[:, None] * np.trace(h) +
          2 * (lr - lt)[:, None] * nhn + np.trace(h @ h))
    rr = r[:, None]
    # |grad psi|^2 with grad psi = gn n + E + r H n
    x2 = (gn[:, None] ** 2 + e @ e + 2 * gn[:, None] * ne +
          2 * rr * gn[:, None] * nhn + 2 * rr * ehn + rr * rr * hn2)
    radial_grad = gn[:, None] + ne + rr * nhn          # n . grad psi
    return dict(r=r, rh=rh, b=b, gn=gn, lr=lr, lt=lt, rho=rho, dlogrho=dlogrho,
                n=n, weights=weights, t2=t2, g2=x2, radial_grad=radial_grad,
                trh=np.trace(h), nhn=nhn)


def first_gradient(obj, profile, grid, e, h):
    geo = geometry(obj, profile, V706.radial_mesh(profile,grid),e,h,grid)
    nu = (1+A0**2/np.maximum(geo['g2'],1e-200))**.25
    st = {k:geo[k] for k in ('r','rh','b','gn','rho','dlogrho')}
    st.update(lead=(nu*geo['radial_grad'])@geo['weights'],
              pnn=np.zeros_like(geo['r']), trp=np.zeros_like(geo['r']))
    work, force = V706.work_from_state(st)
    jeans,_ = V706.aperture_jeans(st,force,obj.re_pc)
    return {'virial_ratio':work['virial'],
            'los_kms':jeans['formal_aperture_rms_kms'],
            'negative_pressure':jeans['has_negative_pressure']}

def polynomial_b(r):
    q = GM/(r*r*A0)
    return A0*np.sqrt(2*q/(1+np.sqrt(1+4*q)))

def saturated_residual(r):
    # Gbare=Glab/2 and a*=2 A0. Compute b-gbare without cancellation.
    astar=2*A0
    q=GM/(2*r*r*astar)
    root=np.hypot(q,2.)
    y=np.sqrt(q*(q+root)/2)
    return astar*(2*q/(root+q))/(y+q)

def precession(a_au,e,days,acc,n):
    if n<5 or n%2==0: raise ValueError('Orbit nodes must be odd and >=5')
    f=np.linspace(0,2*np.pi,n)
    r=a_au*AU*(1-e*e)/(1+e*np.cos(f))
    per_orbit=simpson(acc(r)*r*r*np.cos(f),x=f)/(e*GM)
    return float(per_orbit*(180/np.pi)*3600*36525/days)

def run(out,skip_dwarfs=False):
    frozen_hash='a64440e1350ae8bf10eb3188ec34829b6d586abf275de5cdc841a50ad3478284'
    actual_hash=hashlib.sha256(Path(V706.__file__).read_bytes()).hexdigest()
    if actual_hash != frozen_hash:
        raise RuntimeError('Frozen v7.06 dependency changed; restore the bundled file')
    out.mkdir(parents=True,exist_ok=True)
    results={'status':'post-hoc reconstructed structural audit; no empirical validation',
             'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
             'orbit_elements':PLANETS,'tests':{}}
    tests=results['tests']
    if not skip_dwarfs:
        tests['1_first_gradient']=[]
        for name,obj in V706.DWARFS.items():
            e,h,_=V706.galactic_field(obj,V706.Grid())
            for profile in V706.PROFILES:
                row={'object':name,'profile':profile,'archived_los_kms':obj.observed_kms}
                for label,n,np_,na in [('virial_coarse',180,16,24),('virial_fine',360,24,36),
                                     ('jeans_coarse',450,16,24),('jeans_fine',900,24,36)]:
                    row[label]=first_gradient(obj,profile,V706.Grid(radial=n,polar=np_,azimuth=na),e,h)
                tests['1_first_gradient'].append(row)
                print(name,profile,row['virial_fine']['virial_ratio'],row['jeans_fine']['los_kms'],flush=True)
    t=np.array([.01,1.,10.])
    tests['2_linear_massive']={'beta':1.,'r_over_lambda':t.tolist(),
        'g_over_gN':(1+(1+t)*np.exp(-t)).tolist(),
        'slope':(-t*np.exp(-t)).tolist(),
        'linear_stability_condition':'abs(alpha)<1, m^2>=0; not nonlinear proof'}
    m0=1e10*MSUN
    masses=np.array([.01,1.,10.])
    v0=(G*m0*A0)**.25/1000
    tests['3_planar']={'mass_over_M0':masses.tolist(),'v_planar_kms':(v0*np.sqrt(masses)).tolist(),
        'v_deep_MOND_kms':(v0*masses**.25).tolist(),'v4_over_GMa0':masses.tolist(),
        'disk_R_over_Rd':[2,5,10], 'enclosed_fraction':(1-(1+np.array([2,5,10]))*np.exp(-np.array([2,5,10]))).tolist(),
        'vertical_return_force':0.,'p2':'flat, v^4 proportional to M^2',
        'p3':'v^2 proportional to sqrt(M R), not flat'}
    for key,acc,names in [('4_polynomial',polynomial_b,['Mercury','Earth','Saturn']),
                          ('5_saturated',saturated_residual,['Mercury','Earth','Mars'])]:
        rows=[]
        for name in names:
            a,e,period=PLANETS[name]
            shifts={str(n):precession(a,e,period,acc,n) for n in [257,1025,4097,16385]}
            rows.append({'planet':name,'acceleration_at_a_m_s2':float(acc(a*AU)),
                         'precession_arcsec_century':shifts})
        tests[key]=rows
    dep=Path(V706.__file__)
    results['dependency_sha256']=hashlib.sha256(dep.read_bytes()).hexdigest()
    (out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    lines=['# Reconstructed v7.07 calculations','',
           'These are new runs of reconstructed equations, not recovered historical output.','']
    if not skip_dwarfs:
        lines+=['| Dwarf | Profile | V/V_N fine | LOS fine (km/s) |','|---|---|---:|---:|']
        for row in tests['1_first_gradient']:
            lines.append(f"| {row['object']} | {row['profile']} | {row['virial_fine']['virial_ratio']:.6f} | {row['jeans_fine']['los_kms']:.6f} |")
    for key in ['4_polynomial','5_saturated']:
        lines+=['',f'## {key}','| Planet | acceleration (m/s²) | arcsec/century (16385 nodes) |','|---|---:|---:|']
        for row in tests[key]:
            lines.append(f"| {row['planet']} | {row['acceleration_at_a_m_s2']:.6e} | {row['precession_arcsec_century']['16385']:.9g} |")
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in tests.items() if k!='1_first_gradient'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results'))
    parser.add_argument('--skip-dwarfs',action='store_true',help='Only analytic/orbit tests 2-5')
    args=parser.parse_args()
    run(args.out,args.skip_dwarfs)
