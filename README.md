## The roles of latent heating and dust in the structure and variability of the northern Martian polar vortex
### analysis of temperature, winds, PV, overturning streamfunction and latent heat release
Used for analysis in the paper "The roles of latent heating and dust in the structure and variability of the northern Martian polar vortex"
***
### Components:
**[analysis_functions]:**
Functions for analysis of temperature, wind, and PV data.

**[PVmodule]:**
Functions to allow calculation of potential vorticity and interpolation to isentropic surfaces.
***
### Scripts:
**[calc_streamfunction.py]:**
Calculates the meridional overturning streamfunction from OpenMARS data.

**[calc_streamfunction_from_dsr.py]:**
Calculates the meridional overturning streamfunction from zonal ensemble mean Isca data (as created by create_dsr_Isca.py).

**[calculate_PV_Isca.py]:**
Calculates isobaric and isentropic PV from Isca data on full pressure levels.

**[calculate_PV_Isca_old.py]:**
Calculates isobaric and isentropic PV from Isca data on full pressure levels. Not parallelized.

**[calculate_PV_OpenMARS.py]:**
Calculates isobaric and isentropic PV from OpenMARS data.

**[create_dsr_Isca.py]:**
Calculates the ensemble mean for a given Isca simulation.

**[dust_distribution.py]:**
Plots Figure 1.

**[eddy_enstrophy_all_new.py]:**
Plots Figure 8.

**[eddy_enstrophy_EMARS.py]:**
Plots Figure 12.

**[Isca_30_sol_average_PV.py]:**
Plots scaled PV on an isentropic surface for all process attribution experiments.

**[Isca_30_sol_average_PV_all_years.py]:**
Plots scaled PV on an isentropic surface for all yearly simulations (Figures 6 & 11).

**[Isca_OpenMARS_profiles.py]:**
Plots cross-sectional profiles of temperature, zonal wind and PV, comparing Isca simulations to OpenMARS (Figures 4 & 5).

**[lh_evolution_all.py]:**
Plots Figure 9.

**[OpenMARS_30_sol_average_PV.py]:**
Plots scaled PV on an isentropic surface for all years of OpenMARS data (Figure 3).

**[plot_streamfn_cross_OpenMARS_compare.py]:**
Plots Figure 13.

**[plot_streamfn_cross_all.py]:**
Plots Figure 7.

**[plot_streamfunction_OpenMARS.py]:**
Plots Figure 10.

**[polar_PV_evolution_Isca.py]:**
Plots ensemble spread of temperature, zonal wind and scaled PV from Isca simulations.

**[PV_cross-section_OpenMARS.py]:**
Plots Figure 2.

***
### Dependencies:
- python
- Windspharm
- MetPy
***
Other Python libraries:
- numpy
- xarray
- scipy
- colorcet
