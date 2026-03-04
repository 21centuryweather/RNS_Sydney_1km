#!/usr/bin/env python

"""
Description:
    Zeros the urban land-cover tile (id=6) and re-proportions selected
    non-urban tiles so each grid cell sums to 1.0. Some tiles are exluded
    from re-proportioning (ice, lake, needleleaf) to preserve their values.

Usage:
    Run the Regional Nesting Suite as normal, but "hold" the task:
        [model]_ancil_lct_postproc_c4, then run this script with the following command:

    python ancil_lct_postproc_no_urban.py $PATH_TO_ANCILS/qrparm.veg.frac_cci_pre_c4 --output $PATH_TO_ANCILS/qrparm.veg.frac_cci_no_urban
    
    where:
        PATH_TO_ANCILS is the path to the ancillary files

    You must rename the output manually:

        cp $PATH_TO_ANCILS/qrparm.veg.frac_cci_no_urban.nc $PATH_TO_ANCILS/qrparm.veg.frac_cci_pre_c4.nc

    NOTE THE ".nc" EXTENSION IN THE OUTPUT FILENAME. 

    Then release the held tasks.
"""

__version__ = "2024-12-25"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"
__institution__ = "UNSW Sydney"

import ants
import iris
import numpy as np

def main():

    # 1. load cubes
    target_lct = ants.load_cube(target_lct_path)
    # remove urban tile (id=6)
    new_lct = remove_tile(target_lct, tile_id=6)

    # report any sum-to-one mismatches
    report_sum_mismatch(new_lct)
    report_ice_nonzero(new_lct)

    # plot before/after
    plot_lct(target_lct, output_path, 'pre_no_urban')
    plot_lct(new_lct, output_path, 'no_urban')

    # 5. save to ancil and netcdf
    print(f'saving to {output_path}')

    # ants.io.save.ancil(new_lct, output_path)
    ants.io.save.netcdf(new_lct, output_path+'.nc')

    return

def remove_tile(target_lct, tile_id=6):

    # copy target_lct to retain original and extract urban tile
    new_lct = target_lct.copy()
    urb_frac = target_lct.extract(iris.Constraint(pseudo_level=tile_id))

    # zero the urban fraction but preserve mask
    urb_frac.data = np.ma.array(np.zeros_like(urb_frac.data), mask=urb_frac.data.mask)

    # rebind and normalise so total fractions sum to 1
    new_lct = bind_lct_fractions(new_lct, urb_frac)
    new_lct = normalise_lct(new_lct)

    return new_lct

def normalise_lct(lct_frac):
    """
    Normalise fractions across pseudo_level to sum to 1, preserving masks.
    """

    totals = lct_frac.collapsed('pseudo_level', iris.analysis.SUM)
    totals.data = np.ma.masked_equal(totals.data, 0.0)
    lct_frac.data = lct_frac.data / totals.data

    return lct_frac

def bind_lct_fractions(lct_orig, urb_frac):
    """
    Description:
        Update single land cover tile urban fraction with new source.
        Updated from code by Vinod Kumar: Vinod.Kumar@bom.gov.au

    Parameters
    ----------

    lct_orig: iris.cube.Cube
        The original land cover fraction data (all tiles)
    urb_frac: iris.cube.Cube
        The new urban land cover fraction data (one tile)

    Returns
    -------
    lct_out: iris.cube.Cube
        The modified land cover fraction data.

    """

    URBAN_ID = 6

    # these tiles will not be adjusted, so will be excluded from normalisation
    ICE_ID, LAKE_ID,  NEEDLE_ID = 9, 7, 2

    # copy lct cube to retain original
    lct_frac = lct_orig.copy()

    # Split the impervious and vegetated fractions.
    surface_type_ids = lct_frac.coord('pseudo_level').points[:]
    urb_sfc_idx = np.where(surface_type_ids == URBAN_ID)[0]
    if urb_sfc_idx.size == 0:
        raise ValueError(f'Urban tile id {URBAN_ID} not found in pseudo_level coordinate.')
    veg_sfc_idx = np.where((surface_type_ids != URBAN_ID) &
                              (surface_type_ids != NEEDLE_ID) &
                              (surface_type_ids != LAKE_ID) &
                              (surface_type_ids != ICE_ID))

    veg_sfc_type_ids = surface_type_ids[veg_sfc_idx]
    veg_frac = lct_frac.extract(iris.Constraint(pseudo_level=veg_sfc_type_ids))

    # Replace urban fractions
    lct_frac.data[urb_sfc_idx, :, :] = urb_frac.data[None, ...]

    # Adjust the land fraction values according to the percentage of each
    # fraction in the total natural fraction.
    lct_frac_totals = lct_frac.collapsed('pseudo_level', iris.analysis.SUM)
    spill = lct_frac_totals - 1.0
    veg_frac_totals = veg_frac.collapsed('pseudo_level', iris.analysis.SUM)
    veg_frac_totals.data = np.ma.masked_equal(veg_frac_totals.data, 0.0)
    increments = (veg_frac / veg_frac_totals) * spill

    # Set the values at locations where a division by zero occurs to zero.
    masked_data_indices = np.where(increments.data.mask)
    increments.data[masked_data_indices] = veg_frac.data[masked_data_indices]
    adjusted = veg_frac - increments

    # Bound the values at locations where the fractions spill outside the [0,1] range.
    adjusted.data[masked_data_indices] = np.nan
    ll_bound_indices = np.where(adjusted.data < 0.)
    hh_bound_indices = np.where(adjusted.data > 1.)
    adjusted.data[ll_bound_indices] = 0.
    adjusted.data[hh_bound_indices] = 1.

    # Re-assign the mask.
    adjusted.data[masked_data_indices] = veg_frac.data[masked_data_indices]

    # Update the land conver fraction data.
    lct_out = lct_frac.copy()
    lct_out.data[veg_sfc_idx, :, :] = adjusted.data[:, ...]

    return lct_out

def report_sum_mismatch(lct_frac, tol=1.0e-6, max_report=20):
    """
    Print grid cells where the sum across tiles deviates from 1.
    """

    totals = lct_frac.collapsed('pseudo_level', iris.analysis.SUM)
    data = totals.data
    mask = np.ma.getmaskarray(data)
    bad = (~mask) & (np.abs(data - 1.0) > tol)
    count = int(np.count_nonzero(bad))

    if count == 0:
        print(f'All grid cells sum to 1 within tol={tol}.')
        return

    print(f'{count} grid cells do not sum to 1 (tol={tol}); showing up to {max_report}.')
    idxs = np.argwhere(bad)
    for idx in idxs[:max_report]:
        value = float(data[tuple(idx)])
        print(f'  idx={tuple(idx)} sum={value:.6g}')

def report_ice_nonzero(lct_frac, ice_id=9, tol=0.0, max_report=20):
    """
    Print grid cells where the ice tile is non-zero.
    """

    surface_type_ids = lct_frac.coord('pseudo_level').points[:]
    ice_idx = np.where(surface_type_ids == ice_id)[0]
    if ice_idx.size == 0:
        raise ValueError(f'Ice tile id {ice_id} not found in pseudo_level coordinate.')

    ice = lct_frac.data[ice_idx[0], :, :]
    mask = np.ma.getmaskarray(ice)
    bad = (~mask) & (ice > tol)
    count = int(np.count_nonzero(bad))

    if count == 0:
        print(f'All ice tile values are zero within tol={tol}.')
        return

    print(f'{count} grid cells have non-zero ice (tol={tol}); showing up to {max_report}.')
    idxs = np.argwhere(bad)
    for idx in idxs[:max_report]:
        value = float(ice[tuple(idx)])
        print(f'  idx={tuple(idx)} ice={value:.6g}')

def plot_lct(lct, output_base, label):

    import matplotlib.pyplot as plt

    tile_titles = ['Broadleaf', 'Needleleaf', 'C3', 'C4', 'Shrubs', 'Urban', 'Lake', 'Soil', 'Ice']

    plt.close('all')
    fig, axes = plt.subplots(3,3, figsize=(15, 15), sharex=True, sharey=True)
    for i,ax in enumerate(axes.flatten()):
        ax.imshow(lct.data[i, :, :], interpolation='none', origin='lower', vmin=0, vmax=1, cmap='magma')
        # xr.DataArray().from_iris(new_lct[i, :, :]).plot(ax=ax, vmin=0, vmax=0.5)
        ax.set_title(tile_titles[i])
        ax.xaxis.label.set_visible(False)

    # tight layout with some padding
    fig.tight_layout(h_pad=7)
    fig.savefig(f'{output_base}_{label}.png', dpi=200, bbox_inches='tight')

    return

if __name__ == "__main__":

    parser = ants.AntsArgParser()
    args = parser.parse_args()

    target_lct_path = args.sources[0]
    output_path = args.output

    main()
