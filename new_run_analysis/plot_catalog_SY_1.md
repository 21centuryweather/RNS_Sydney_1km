# Plot Catalog

## Contents

- [abl_ht](#abl_ht) (Height above ground level of the top of the planetary boundary layer (i.e. boundary layer depth).)
- [av_ht_flx_soil1](#av_ht_flx_soil1) (HT FLUX FROM SURF TO DEEP SOIL LEV 1)
- [av_lat_hflx](#av_lat_hflx) (Latent heat flux at the surface (upward outgoing radiation values are positive).)
- [av_lwsfcdown](#av_lwsfcdown) (Downward flux of longwave radiation at the ground or ocean surface.)
- [av_netlwsfc](#av_netlwsfc) (Net downward (total downward minus upward) radiative longwave flux at the surface (ground or ocean).)
- [av_netswsfc](#av_netswsfc) (Net downward (total downward minus upward) radiative shortwave flux at the surface (ground or ocean).)
- [av_olr](#av_olr) (Longwave radiation flux leaving the top of the atmosphere.)
- [av_rate_ls_rain](#av_rate_ls_rain) (LARGE SCALE RAINFALL RATE    KG/M2/S)
- [av_rate_ls_snow](#av_rate_ls_snow) (LARGE SCALE SNOWFALL RATE    KG/M2/S)
- [av_sens_hflx](#av_sens_hflx) (Sensible heat flux at the surface (upward outgoing radiation values are positive).)
- [av_swsfcdown](#av_swsfcdown) (Downward flux of shortwave radiation at the ground or ocean surface.)
- [bl_type_comb](#bl_type_comb) (COMBINED BOUNDARY LAYER TYPE)
- [fric_vel](#fric_vel) (Friction velocity (a scalar measure of the magnitude of the surface stress).)
- [geop_ht_rho](#geop_ht_rho) (GEOPOTENTIAL HEIGHT ON RHO LEVELS)
- [h_eff_ruff](#h_eff_ruff) (EFFECTIVE ROUGHNESS LEN FOR SCALARS)
- [lnd_mask](#lnd_mask) (Land mask (takes the value 1 or -1 if any land is in the grid box, otherwise 0).)
- [max_wndgust10m](#max_wndgust10m) (Maximum three second wind speed (wind gust) at 10m above ground level.)
- [maxcol_refl](#maxcol_refl) (MAX REFLECTIVITY IN COLUMN  (dBZ))
- [mslp](#mslp) (Atmospheric pressure at mean sea level.)
- [olr](#olr) (Longwave radiation flux leaving the top of the atmosphere.)
- [qsair_scrn](#qsair_scrn) (Atmospheric specific humidity at 1.5m above-ground-level (screen level). Calculated by integrating the similarity equations from the surface to 1.5m (surface value taken as saturated specific humidity at the surface temperature).)
- [radar_refl_1km](#radar_refl_1km) (RADAR REFLECTIVITY AT 1KM AGL (dBZ))
- [rate_ls_rain](#rate_ls_rain) (LARGE SCALE RAINFALL RATE    KG/M2/S)
- [rate_ls_snow](#rate_ls_snow) (LARGE SCALE SNOWFALL RATE    KG/M2/S)
- [rh_scrn](#rh_scrn) (RELATIVE HUMIDITY AT 1.5M)
- [roughness_len](#roughness_len) (ROUGHNESS LEN. AFTER B.L. (SEE DOC))
- [sfc_pres](#sfc_pres) (Atmospheric pressure at the surface.)
- [sfc_sw_dif](#sfc_sw_dif) (Downward diffuse shortwave radiation flux at the surface.)
- [sfc_sw_dir](#sfc_sw_dir) (Downward direct shortwave radiation flux at the surface.)
- [sfc_temp](#sfc_temp) (Temperature of the land or sea/sea-ice surface.  On land points this is the surface "skin" temperature.  On ice-free sea points it is the temperature of the sea surface, and on sea points with ice it is a gridbox mean given by: [(ice fraction)*(temperature of top ice layer computed by the atmosphere surface/boundary layer scheme)] + [(1 - ice fraction)*(freezing point of sea water)].)
- [soil_mois](#soil_mois) (Total (frozen and unfrozen) soil moisture content in soil layer 1 (surface to 0.1 m depth), soil layer 2 (0.10 m to 0.35 m depth), soil layer 3 (0.35 m to 1 m depth), and soil layer 4 (1 m to 3 m depth).)
- [soil_temp](#soil_temp) (Soil/land-ice temperature in soil layer 1 (surface to 0.1 m depth), soil layer 2 (0.10 m to 0.35 m depth), soil layer 3 (0.35 m to 1 m depth), and soil layer 4 (1 m to 3 m depth).)
- [swsfcdown](#swsfcdown) (Downward flux of shortwave radiation at the ground or ocean surface.)
- [temp_scrn](#temp_scrn) (Atmospheric air temperature at 1.5m above ground-level (screen level).)
- [topog](#topog) (Height of topography (above the geoid).)
- [ttl_cld](#ttl_cld) (Total cloud coverage calculated with a maximum-random overlap assumption.)
- [ustar](#ustar) (Surface friction velocity in air, a scalar measure of surface stress)
- [uwnd10m_b](#uwnd10m_b) (10 METRE WIND U-COMP         B GRID)
- [vertical_wnd](#vertical_wnd) (Vertical component of the wind velocity in pressure co-ordinates (often referred to as "omega").)
- [vwnd10m_b](#vwnd10m_b) (10 METRE WIND V-COMP         B GRID)
- [wnd_ucmp](#wnd_ucmp) (Zonal (U) component of the wind velocity in pressure co-ordinates.)
- [wnd_vcmp](#wnd_vcmp) (Meridional (V) component of the wind velocity in pressure co-ordinates.)
- [wndgust10m](#wndgust10m) (Maximum three second wind speed (wind gust) at 10m above ground level.)
- [wndgust10m_scale](#wndgust10m_scale) (SCALE-DEPENDENT WIND GUST (M/S))

## abl_ht

**bom_description:** Height above ground level of the top of the planetary boundary layer (i.e. boundary layer depth).

**stash_long_name:** BOUNDARY LAYER DEPTH AFTER TIMESTEP

**Spatial:** ![Spatial](plots_SY_1/SY_1_abl_ht_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_abl_ht_diurnal_CTRL_NO-URBAN.png)

## av_ht_flx_soil1

**bom_description:** None

**stash_long_name:** HT FLUX FROM SURF TO DEEP SOIL LEV 1

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_ht_flx_soil1_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_ht_flx_soil1_diurnal_CTRL_NO-URBAN.png)

## av_lat_hflx

**bom_description:** Latent heat flux at the surface (upward outgoing radiation values are positive).

**stash_long_name:** SURFACE LATENT HEAT FLUX        W/M2

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_lat_hflx_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_lat_hflx_diurnal_CTRL_NO-URBAN.png)

## av_lwsfcdown

**bom_description:** Downward flux of longwave radiation at the ground or ocean surface.

**stash_long_name:** DOWNWARD LW RAD FLUX: SURFACE

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_lwsfcdown_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_lwsfcdown_diurnal_CTRL_NO-URBAN.png)

## av_netlwsfc

**bom_description:** Net downward (total downward minus upward) radiative longwave flux at the surface (ground or ocean).

**stash_long_name:** NET DOWN SURFACE LW RAD FLUX

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_netlwsfc_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_netlwsfc_diurnal_CTRL_NO-URBAN.png)

## av_netswsfc

**bom_description:** Net downward (total downward minus upward) radiative shortwave flux at the surface (ground or ocean).

**stash_long_name:** NET DOWN SURFACE SW FLUX : CORRECTED

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_netswsfc_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_netswsfc_diurnal_CTRL_NO-URBAN.png)

## av_olr

**bom_description:** Longwave radiation flux leaving the top of the atmosphere.

**stash_long_name:** OUTGOING LW RAD FLUX (TOA)

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_olr_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_olr_diurnal_CTRL_NO-URBAN.png)

## av_rate_ls_rain

**bom_description:** None

**stash_long_name:** LARGE SCALE RAINFALL RATE    KG/M2/S

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_rate_ls_rain_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_rate_ls_rain_diurnal_CTRL_NO-URBAN.png)

## av_rate_ls_snow

**bom_description:** None

**stash_long_name:** LARGE SCALE SNOWFALL RATE    KG/M2/S

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_rate_ls_snow_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_rate_ls_snow_diurnal_CTRL_NO-URBAN.png)

## av_sens_hflx

**bom_description:** Sensible heat flux at the surface (upward outgoing radiation values are positive).

**stash_long_name:** SURFACE SENSIBLE  HEAT FLUX     W/M2

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_sens_hflx_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_sens_hflx_diurnal_CTRL_NO-URBAN.png)

## av_swsfcdown

**bom_description:** Downward flux of shortwave radiation at the ground or ocean surface.

**stash_long_name:** TOTAL DOWNWARD SURFACE SW FLUX

**Spatial:** ![Spatial](plots_SY_1/SY_1_av_swsfcdown_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_av_swsfcdown_diurnal_CTRL_NO-URBAN.png)

## bl_type_comb

**bom_description:** None

**stash_long_name:** COMBINED BOUNDARY LAYER TYPE

**Spatial:** ![Spatial](plots_SY_1/SY_1_bl_type_comb_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_bl_type_comb_diurnal_CTRL_NO-URBAN.png)

## fric_vel

**bom_description:** Friction velocity (a scalar measure of the magnitude of the surface stress).

**stash_long_name:** EXPLICIT FRICTION VELOCITY

**Spatial:** ![Spatial](plots_SY_1/SY_1_fric_vel_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_fric_vel_diurnal_CTRL_NO-URBAN.png)

## geop_ht_rho

**bom_description:** None

**stash_long_name:** GEOPOTENTIAL HEIGHT ON RHO LEVELS

**Spatial:** ![Spatial](plots_SY_1/SY_1_geop_ht_rho_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_geop_ht_rho_diurnal_CTRL_NO-URBAN.png)

## h_eff_ruff

**bom_description:** None

**stash_long_name:** EFFECTIVE ROUGHNESS LEN FOR SCALARS

**Spatial:** ![Spatial](plots_SY_1/SY_1_h_eff_ruff_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_h_eff_ruff_diurnal_CTRL_NO-URBAN.png)

## lnd_mask

**bom_description:** Land mask (takes the value 1 or -1 if any land is in the grid box, otherwise 0).

**stash_long_name:** LAND MASK (No halo) (LAND=TRUE)

**Spatial:** ![Spatial](plots_SY_1/SY_1_lnd_mask_CTRL_NO-URBAN.png)

**Diurnal:** (missing)

## max_wndgust10m

**bom_description:** Maximum three second wind speed (wind gust) at 10m above ground level.

**stash_long_name:** WIND GUST  (M/S)

**Spatial:** ![Spatial](plots_SY_1/SY_1_max_wndgust10m_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_max_wndgust10m_diurnal_CTRL_NO-URBAN.png)

## maxcol_refl

**bom_description:** None

**stash_long_name:** MAX REFLECTIVITY IN COLUMN  (dBZ)

**Spatial:** ![Spatial](plots_SY_1/SY_1_maxcol_refl_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_maxcol_refl_diurnal_CTRL_NO-URBAN.png)

## mslp

**bom_description:** Atmospheric pressure at mean sea level.

**stash_long_name:** PRESSURE AT MEAN SEA LEVEL

**Spatial:** ![Spatial](plots_SY_1/SY_1_mslp_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_mslp_diurnal_CTRL_NO-URBAN.png)

## olr

**bom_description:** Longwave radiation flux leaving the top of the atmosphere.

**stash_long_name:** OUTGOING LW RAD FLUX (TOA)

**Spatial:** ![Spatial](plots_SY_1/SY_1_olr_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_olr_diurnal_CTRL_NO-URBAN.png)

## qsair_scrn

**bom_description:** Atmospheric specific humidity at 1.5m above-ground-level (screen level). Calculated by integrating the similarity equations from the surface to 1.5m (surface value taken as saturated specific humidity at the surface temperature).

**stash_long_name:** SPECIFIC HUMIDITY  AT 1.5M

**Spatial:** ![Spatial](plots_SY_1/SY_1_qsair_scrn_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_qsair_scrn_diurnal_CTRL_NO-URBAN.png)

## radar_refl_1km

**bom_description:** None

**stash_long_name:** RADAR REFLECTIVITY AT 1KM AGL (dBZ)

**Spatial:** ![Spatial](plots_SY_1/SY_1_radar_refl_1km_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_radar_refl_1km_diurnal_CTRL_NO-URBAN.png)

## rate_ls_rain

**bom_description:** None

**stash_long_name:** LARGE SCALE RAINFALL RATE    KG/M2/S

**Spatial:** ![Spatial](plots_SY_1/SY_1_rate_ls_rain_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_rate_ls_rain_diurnal_CTRL_NO-URBAN.png)

## rate_ls_snow

**bom_description:** None

**stash_long_name:** LARGE SCALE SNOWFALL RATE    KG/M2/S

**Spatial:** ![Spatial](plots_SY_1/SY_1_rate_ls_snow_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_rate_ls_snow_diurnal_CTRL_NO-URBAN.png)

## rh_scrn

**bom_description:** None

**stash_long_name:** RELATIVE HUMIDITY AT 1.5M

**Spatial:** ![Spatial](plots_SY_1/SY_1_rh_scrn_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_rh_scrn_diurnal_CTRL_NO-URBAN.png)

## roughness_len

**bom_description:** None

**stash_long_name:** ROUGHNESS LEN. AFTER B.L. (SEE DOC)

**Spatial:** ![Spatial](plots_SY_1/SY_1_roughness_len_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_roughness_len_diurnal_CTRL_NO-URBAN.png)

## sfc_pres

**bom_description:** Atmospheric pressure at the surface.

**stash_long_name:** SURFACE PRESSURE AFTER TIMESTEP

**Spatial:** ![Spatial](plots_SY_1/SY_1_sfc_pres_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_sfc_pres_diurnal_CTRL_NO-URBAN.png)

## sfc_sw_dif

**bom_description:** Downward diffuse shortwave radiation flux at the surface.

**stash_long_name:** DIFFUSE SURFACE SW FLUX : CORRECTED

**Spatial:** ![Spatial](plots_SY_1/SY_1_sfc_sw_dif_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_sfc_sw_dif_diurnal_CTRL_NO-URBAN.png)

## sfc_sw_dir

**bom_description:** Downward direct shortwave radiation flux at the surface.

**stash_long_name:** DIRECT SURFACE SW FLUX : CORRECTED

**Spatial:** ![Spatial](plots_SY_1/SY_1_sfc_sw_dir_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_sfc_sw_dir_diurnal_CTRL_NO-URBAN.png)

## sfc_temp

**bom_description:** Temperature of the land or sea/sea-ice surface.  On land points this is the surface "skin" temperature.  On ice-free sea points it is the temperature of the sea surface, and on sea points with ice it is a gridbox mean given by: [(ice fraction)*(temperature of top ice layer computed by the atmosphere surface/boundary layer scheme)] + [(1 - ice fraction)*(freezing point of sea water)].

**stash_long_name:** SURFACE TEMPERATURE AFTER TIMESTEP

**Spatial:** ![Spatial](plots_SY_1/SY_1_sfc_temp_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_sfc_temp_diurnal_CTRL_NO-URBAN.png)

## soil_mois

**bom_description:** Total (frozen and unfrozen) soil moisture content in soil layer 1 (surface to 0.1 m depth), soil layer 2 (0.10 m to 0.35 m depth), soil layer 3 (0.35 m to 1 m depth), and soil layer 4 (1 m to 3 m depth).

**stash_long_name:** SOIL MOISTURE CONTENT IN A LAYER

**Spatial:** ![Spatial](plots_SY_1/SY_1_soil_mois_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_soil_mois_diurnal_CTRL_NO-URBAN.png)

## soil_temp

**bom_description:** Soil/land-ice temperature in soil layer 1 (surface to 0.1 m depth), soil layer 2 (0.10 m to 0.35 m depth), soil layer 3 (0.35 m to 1 m depth), and soil layer 4 (1 m to 3 m depth).

**stash_long_name:** DEEP SOIL TEMP. AFTER HYDROLOGY DEGK

**Spatial:** ![Spatial](plots_SY_1/SY_1_soil_temp_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_soil_temp_diurnal_CTRL_NO-URBAN.png)

## swsfcdown

**bom_description:** Downward flux of shortwave radiation at the ground or ocean surface.

**stash_long_name:** TOTAL DOWNWARD SURFACE SW FLUX

**Spatial:** ![Spatial](plots_SY_1/SY_1_swsfcdown_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_swsfcdown_diurnal_CTRL_NO-URBAN.png)

## temp_scrn

**bom_description:** Atmospheric air temperature at 1.5m above ground-level (screen level).

**stash_long_name:** TEMPERATURE AT 1.5M

**Spatial:** ![Spatial](plots_SY_1/SY_1_temp_scrn_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_temp_scrn_diurnal_CTRL_NO-URBAN.png)

## topog

**bom_description:** Height of topography (above the geoid).

**stash_long_name:** OROGRAPHY (/STRAT LOWER BC)

**Spatial:** ![Spatial](plots_SY_1/SY_1_topog_CTRL_NO-URBAN.png)

**Diurnal:** (missing)

## ttl_cld

**bom_description:** Total cloud coverage calculated with a maximum-random overlap assumption.

**stash_long_name:** TOTAL CLOUD AMOUNT MAX/RANDOM OVERLP

**Spatial:** ![Spatial](plots_SY_1/SY_1_ttl_cld_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_ttl_cld_diurnal_CTRL_NO-URBAN.png)

## ustar

**bom_description:** Surface friction velocity in air, a scalar measure of surface stress

**stash_long_name:** IMPLICIT FRICTION VELOCITY

**Spatial:** ![Spatial](plots_SY_1/SY_1_ustar_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_ustar_diurnal_CTRL_NO-URBAN.png)

## uwnd10m_b

**bom_description:** None

**stash_long_name:** 10 METRE WIND U-COMP         B GRID

**Spatial:** ![Spatial](plots_SY_1/SY_1_uwnd10m_b_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_uwnd10m_b_diurnal_CTRL_NO-URBAN.png)

## vertical_wnd

**bom_description:** Vertical component of the wind velocity in pressure co-ordinates (often referred to as "omega").

**stash_long_name:** W COMPNT (OF WIND) ON PRESSURE LEVS

**Spatial:** ![Spatial](plots_SY_1/SY_1_vertical_wnd_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_vertical_wnd_diurnal_CTRL_NO-URBAN.png)

## vwnd10m_b

**bom_description:** None

**stash_long_name:** 10 METRE WIND V-COMP         B GRID

**Spatial:** ![Spatial](plots_SY_1/SY_1_vwnd10m_b_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_vwnd10m_b_diurnal_CTRL_NO-URBAN.png)

## wnd_ucmp

**bom_description:** Zonal (U) component of the wind velocity in pressure co-ordinates.

**stash_long_name:** U WIND ON PRESSURE LEVELS    B GRID

**Spatial:** ![Spatial](plots_SY_1/SY_1_wnd_ucmp_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_wnd_ucmp_diurnal_CTRL_NO-URBAN.png)

## wnd_vcmp

**bom_description:** Meridional (V) component of the wind velocity in pressure co-ordinates.

**stash_long_name:** V WIND ON PRESSURE LEVELS    B GRID

**Spatial:** ![Spatial](plots_SY_1/SY_1_wnd_vcmp_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_wnd_vcmp_diurnal_CTRL_NO-URBAN.png)

## wndgust10m

**bom_description:** Maximum three second wind speed (wind gust) at 10m above ground level.

**stash_long_name:** WIND GUST  (M/S)

**Spatial:** ![Spatial](plots_SY_1/SY_1_wndgust10m_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_wndgust10m_diurnal_CTRL_NO-URBAN.png)

## wndgust10m_scale

**bom_description:** None

**stash_long_name:** SCALE-DEPENDENT WIND GUST (M/S)

**Spatial:** ![Spatial](plots_SY_1/SY_1_wndgust10m_scale_CTRL_NO-URBAN.png)

**Diurnal:** ![Diurnal](plots_SY_1/SY_1_wndgust10m_scale_diurnal_CTRL_NO-URBAN.png)
