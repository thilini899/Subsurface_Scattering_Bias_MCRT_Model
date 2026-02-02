## Scattering Bias in ATL06 products in ICESat 2 Using the Monte Carlo Radiative Transfer model

This repository implements a **Monte Carlo radiative transfer (MCRT)** simulation for **photon transport in ice**. It is designed to model how laser light propagates, scatters, and is absorbed in glacier/ice-sheet media, and to produce outputs such as  **time-of-flight (ToF) / return histograms** relevant to photon-counting LiDAR and remote-sensing applications.

The simulator tracks individual photons through a scattering/absorbing medium using randomly determined sampling of step lengths and scattering angles, optionally including **surface Fresnel reflection**,**total internal reflection**, **scattering lengths**, and **angular incidence**.

---

## Table of contents
- [Physical model](#Physical-model)
- [Algorithm overview](#algorithm-overview)
- [Repository layout](#repository-layout)
- [Inputs](#inputs)
- [Contact](#contact)

---

## Physical model

The medium is parameterized by optical properties commonly used in radiative transfer:

- **Scattering length**:    l_s  [m]
- **Absorption length**:    l_a  [m]
- **Phase function**:   **Henyey–Greenstein (HG)** with anisotropy parameter **g=0.75** 
- **Refractive index**: n (ice) **n(ice)=1.33**
-**Angle of Incidence**:     The angle between the surface normal and the light beam.
Photon free paths are sampled from an exponential distribution:
- step length:  s ~ Exp(μ_t)  →  s = −ln(1-random uniform number)*g

Scattering angles are sampled from the chosen phase function (e.g., HG), and absorption is a constant.

The simulation includes **interface physics** at the air–ice boundary:
- Fresnel reflectance and transmittance
- total internal reflection (when applicable)

---

## Algorithm overview

For each photon:

1. **Initialize** position, direction, time, and angle of incidence.
2. **Launch** Launch photons from a source model (e.g., emission times sampled from an ATLAS instrument response function) at a specified incidence angle.
3. **Surface interaction**:
   - compute Fresnel reflection
   - transmit into ice or reflect back to air
4. **Propagate in ice**:
   - sample step length `s`
   - update position: r ← r + s·u
   - update time-of-flight: t ← t + (n/c)·s 
5. **Absorption**:
   - R > exp(−s/l_a) (R is a random number between (0 and 1))
   - probabilistic termination 
6. **Scatter**:
   - sample scattering polar angle θ and azimuth φ
   - rotate direction vector according to θ, φ
7. **Detection**:
   - if photon exits medium, record:
     - ToF
8. **Terminate** when:
   - photon is detected
   - photon is absorbed
   - max interactions reached

---

## Repository layout

```text
.
├── MCRT_photons_inice.py                  
               # Core MCRT implementation (photon, medium, geometry, phase function)
               # Plot incident and backscattered pulse distribution

├── lightpen_depths.py             
               # Plot ligth penetration depth vs Angle of incidence graph
├── gridplot.py                
               #  Plot grid plot of Scattering lengths vs Angle of incidence graph indicating light penetration depth significance.
└── README.md

```
---

## Inputs

ATLAS's impulse response function is hosted at the NASA National Snow and Ice Data Center Distributed Active Archive Center (\url{https://doi.org/10.5067/EVKXYHW95FPJ})


## Contact

Thilini Bamunu Arachchige (t.thilakarathne@und.edu)
Markus Allgaier (markus.allgaier@und.edu)