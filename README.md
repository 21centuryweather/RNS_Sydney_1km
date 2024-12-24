# Sydney_1km
Various ACCESS RNS runs to 1km resolution

## Experiments running

| ID | Name                      | resolution | domain  | start    | end       | levst    | land ICS | science | land cover | Notes                                        | first output                                                                                           |
|----|---------------------------|------------|---------|----------|-----------|----------|----------|---------|------------|----------------------------------------------|--------------------------------------------------------------------------------------------------------|
| 1  | BARRA_CCI_SY_12p2         | 12.2 km    | 320x400 | 1/1/2017 | 31/1/2017 | L70_80km | BARRA-R2 | GAL9    | CCI        | parent model                                 | /scratch/ce10/mjl561/cylc-run/rns_ostia/share/cycle/20170101T0000Z/BARRA_CCI/SY_12p2/GAL9/um           |
| 2  | BARRA_CCI_SY_5            | 5 km       | 320x400 | 1/1/2017 | 31/1/2017 | L70_80km | BARRA-R2 | RAL3.2  | CCI        | 5km                                          | /scratch/ce10/mjl561/cylc-run/rns_ostia/share/cycle/20170101T0000Z/BARRA_CCI/SY_5/RAL3P2/um            |
| 3  | BARRA_CCI_WC_SY_5         | 5 km       | 320x400 | 1/1/2017 | 31/1/2017 | L70_80km | BARRA-R2 | RAL3.2  | CCI+WC     | 5km WorldCover                               | /scratch/ce10/mjl561/cylc-run/rns_ostia/share/cycle/20170101T0000Z/BARRA_CCI_WC/SY_5/RAL3P2/um         |
| 4  | BARRA_CCI_SY_1            | 1 km       | 360x450 | 1/1/2017 | 31/1/2017 | L90_40km | BARRA-R2 | RAL3.2  | CCI        | 1km                                          | /scratch/ce10/mjl561/cylc-run/rns_ostia/share/cycle/20170101T0000Z/BARRA_CCI/SY_1/RAL3P2/um            |
| 5  | BARRA_CCI_WC_SY_1         | 1 km       | 360x450 | 1/1/2017 | 31/1/2017 | L90_40km | BARRA-R2 | RAL3.2  | CCI+WC     | 1km WorldCover                               | /scratch/ce10/mjl561/cylc-run/rns_ostia/share/cycle/20170101T0000Z/BARRA_CCI_WC/SY_1/RAL3P2/um         |
| ~~6~~  | ~~BARRA_CCI_WC_SY_1_alb_anc~~ | 1 km       | 360x450 | 1/1/2017 | 31/1/2017 | L90_40km | BARRA-R2 | RAL3.2  | CCI+WC     | 1km WorldCover with alb_anc option turned on **ERROR: did not run** | /scratch/ce10/mjl561/cylc-run/rns_ostia/share/cycle/20170101T0000Z/BARRA_CCI_WC/SY_1_alb_anc/RAL3P2/um |
| 7  | BARRA_CCI_SY_1_L       | 1 km large | 720x960 | 1/1/2017 | 31/1/2017 | L90_40km | BARRA-R2 | RAL3.2  | CCI        | 1km large domain                             | /scratch/ce10/mjl561/cylc-run/rns_ostia/share/cycle/20170101T0000Z/BARRA_CCI/SY_1_L/RAL3P2/um          |
| 8  | BARRA_CCI_WC_SY_1_L       | 1 km large | 720x960 | 1/1/2017 | 31/1/2017 | L90_40km | BARRA-R2 | RAL3.2  | CCI+WC        | 1km large domain with WorldCover       | /scratch/ce10/mjl561/cylc-run/rns_ostia/share/cycle/20170101T0000Z/BARRA_CCI_WC/SY_1_L/RAL3P2/um          |

## Domain and ancillaries
Current domain is:

|               | y_npts | x_npts | Science | levset   | nproc | CPUS | SU 24hrs | walltime 24hrs |
|---------------|--------|--------|---------|----------|-------|------|----------|----------------|
| ec_recon      | 340    | 420    | -       | -        | 12x16 | 192  | 192      | 1 min          |
| SY_11p1       | 320    | 400    | GAL9    | L70_80km | 12x16 | 192  | 60       | 10 mins        |
| SY_5          | 320    | 400    | RAL3p2  | L70_80km | 12x16 | 192  | 110      | 18 mins        |
| SY_1          | 360    | 450    | RAL3p2  | L90_40km | 18x16 | 288  | 450      | 36 mins        |
| SY_1_L        | 720    | 960    | RAL3p2  | L90_40km | 32x36 | 1152 | 1700     | 45 mins        |
|               |        |        |         |          |       |      |          |                |
| Total (small) |        |        |         |          |       |      | 740      | SU             |
| Total (large) |        |        |         |          |       |      | 1990     | SU             |

![Sydney domains](plotting_code/figures/SY_domain_surface_altitude.png)

Note the SY_1 and SY_1_L domains are both nested within SY_5 parent domain.
The larger domain may be used for offshore wind experiments, the smaller for assessment of Sydney urban effects.

The ancillary generation suite for SY_1km (u-dl705) is based on the Regional Ancillary Suite (u-bu503), located at:
https://code.metoffice.gov.uk/trac/roses-u/browser/d/l/7/0/5/trunk

u-dl705 includes a new optional configuration file as at:
https://code.metoffice.gov.uk/trac/roses-u/browser/d/l/7/0/5/trunk/opt/rose-suite-SY_1km.conf

u-dl705 also checks out a new branch of the ANTS contrib code to allow a larger spiral search radius, as documented here:
https://forum.access-hive.org.au/t/aus2200-vegetation-fraction-ancil-creation-issues/1972/20

The new ANTS contrib branch is located at:
https://code.metoffice.gov.uk/trac/ancil/log/contrib/branches/dev/mathewlipson/r14121_RAS_CONTRIB_1p0_large_spiral


### To create ancillaries

```
# log into MOSRS
module use /g/data/hr22/modulefiles
module load cylc7/23.09
mosrs-auth

# checkout updated version of the Regional Ancillary Suite (u-bu503)
rosie co u-dl705
cd u-dl705

# run with optional file to create 4 regions
rose suite-run -O SY_1km
```

Ancillaries will be created in  `$HOME/cylc-run/u-dl705/share/data/ancils/SY_CCI`. Generation takes ~ 30 minutes.

### Run Regional Nesting Suite
