## The roles of latent heating and dust in the structure and variability of the northern Martian polar vortex
### analysis of temperature, winds, PV, overturning streamfunction and latent heat release
Used for analysis in the paper "The roles of latent heating and dust in the structure and variability of the northern Martian polar vortex"

[![DOI](https://zenodo.org/badge/348445677.svg)](https://zenodo.org/badge/latestdoi/348445677)

***
### Components:
**[analysis_functions](analysis_functions.py):**
Functions for analysis of temperature, wind, and PV data.

**[PVmodule](PVmodule.py):**
Functions to allow calculation of potential vorticity and interpolation to isentropic surfaces.
***
### Scripts:
**[calc_streamfunction.py](calc_streamfunction.py):**
Calculates the meridional overturning streamfunction from OpenMARS data.

**[calc_streamfunction_from_dsr.py](calc_streamfunction_from_dsr.py):**
Calculates the meridional overturning streamfunction from zonal ensemble mean Isca data.

**[calculate_PV_Isca.py](calculate_PV_Isca.py):**
Calculates isobaric and isentropic PV from Isca data on full pressure levels.

**[calculate_PV_Isca_old.py](calculate_PV_Isca_old.py):**
Calculates isobaric and isentropic PV from Isca data on full pressure levels. Not parallelized.

**[calculate_PV_OpenMARS.py](calculate_PV_OpenMARS.py):**
Calculates isobaric and isentropic PV from OpenMARS data.

**[data_reduct_EMARS.py](data_reduct_EMARS.py)**
Reduces volume of EMARS data needed to plot figures.

**[data_reduct_Isca.py](data_reduct_Isca.py)**
Reduces volume of Isca data needed to plot figures on the 300K surface.

**[data_reduct_Isca_350K.py](data_reduct_Isca_350K.py)**
Reduces volume of Isca data needed to plot figures on the 350K surface.

**[data_reduct_Isca_t_tend.py](data_reduct_Isca_t_tend.py)**
Reduces volume of Isca data needed to plot temperature tendency figures.

**[data_reduct_Isca_zonal.py](data_reduct_Isca_zonal.py)**
Reduces volume of Isca data needed to plot zonally averaged figures.

**[data_reduct_OM_300K.py](data_reduct_OM_300K.py)**
Reduces volume of OpenMARS data needed to plot PV on the 300K surface.

**[data_reduct_OM_zonal.py](data_reduct_OM_zonal.py)**
Reduces volume of OpenMARS data needed to plot zonal mean figures.

**[dust_distribution.py](dust_distribution.py):**
Plots Figure 1.

**[eddy_enstrophy_all_new.py](eddy_enstrophy_all_new.py):**
Plots Figure 10.

**[EMARS_evolution_all_diags.py](EMARS_evolution_all_diags.py):**
Plots Figure 14.

**[Hadley_and_jet_strength.py](Hadley_and_jet_strength.py):**
Plots Figure 12.

**[Hadley_evolution_all.py](Hadley_evolution_all.py):**
Plots strength and edge of the Hadley cell, not shown in paper.

**[Isca_30_sol_average_PV.py](Isca_30_sol_average_PV.py):**
Plots scaled PV on an isentropic surface for all process attribution experiments (Figure 6).

**[Isca_30_sol_average_PV_all_years.py](Isca_30_sol_average_PV_all_years.py):**
Plots scaled PV on an isentropic surface for all yearly simulations (Figure 7).

**[Isca_30_sol_average_PV_all_years_old.py](Isca_30_sol_average_PV_all_years_old.py):**
Plots scaled PV on an isentropic surface for all high dust simulations (Figure 13).

**[Isca_OpenMARS_profiles.py](Isca_OpenMARS_profiles.py):**
Plots cross-sectional profiles of temperature, zonal wind and PV, comparing Isca simulations to OpenMARS (Figures 4 & 5).

**[jet_evolution_all.py](jet_evolution_all.py):**
Plots evolution of jet strength and location for reanalysis and simulations, not shown.

**[lh_evolution_all.py](lh_evolution_all.py):**
Plots Figure 11.

**[OpenMARS_30_sol_average_PV.py](OpenMARS_30_sol_average_PV.py):**
Plots scaled PV on an isentropic surface for all years of OpenMARS data (Figure 3).

**[plot_streamfn_cross_all.py](plot_streamfn_cross_all.py):**
Plots Figure 8.

**[polar_PV_evolution_Isca.py](polar_PV_evolution_Isca.py):**
Plots ensemble spread of temperature, zonal wind and scaled PV from Isca simulations, not shown in paper.

**[polar_PV_evolution_all.py](polar_PV_evolution_all.py):**
Plots scaled PV from reanalysis and simulations (Figure 9).

**[PV_cross-section_OpenMARS.py](PV_cross-section_OpenMARS.py):**
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
***
### Other:
**[environment.yml](environment.yml):**
Python environment file to create an identical Python environment to the one used for analysis herein.
