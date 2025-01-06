#!/usr/bin/env python

"""
Description:
    Replaces target urban tiles with nearest non-urban neighbour.

Usage:
    Run the Regional Nesting Suite as normal, but "hold" the task:
        [model]_ancil_lct_postproc_c4, then run this script with the following command:

    python ancil_lct_postproc_no_urban.py $PATH_TO_ANCILS/qrparm.veg.frac_cci_pre_c4 \
    --target-lsm $PATH_TO_ANCILS/qrparm.mask \
    --output $PATH_TO_ANCILS/qrparm.veg.frac_cci_pre_c4
    
    where:
        PATH_TO_ANCILS is the path to the ancillary files
    Then release the held tasks.
"""

__version__ = "2024-12-25"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"
__institution__ = "UNSW Sydney"

import ants
import ants.decomposition as decomp
import iris
import numpy as np

def main():

    # 1. load cubes
    target_lct = ants.load_cube(target_lct_path)
    target_lsm = ants.load_cube(target_lsm_path)

    # remove urban tile (id=6)
    new_lct = remove_tile(target_lct, target_lsm, tile_id=6)

    # plot
    plot_new_lct(new_lct)

    # 5. save to ancil and netcdf
    print(f'saving to {output_path}')
    ants.save(target_lct, output_path+'.pre_no_urban')
    ants.save(new_lct, output_path)

    return

def remove_tile(target_lct, target_lsm, tile_id=6):

    # copy target_lct to retain original and extract urban tile
    new_lct = target_lct.copy()
    urb_frac = target_lct.extract(iris.Constraint(pseudo_level=tile_id))

    # urb_frac.data.mask to include any urb_frac.data > 0
    new_mask = np.logical_or(urb_frac.data.mask, urb_frac.data.data > 0)

    # update new_lct mask to exclude any urban areas
    new_lct.data.mask = new_mask

    # use nearest neighbour to fill urban areas and ensure consistency with lsm
    ants.analysis.make_consistent_with_lsm(new_lct, target_lsm, invert_mask=True)
    ants.analysis.cover_mapping.normalise_fractions(new_lct)

    return new_lct

def plot_new_lct(new_lct):

    import matplotlib.pyplot as plt

    tile_titles = ['Broadleaf', 'Needleleaf', 'C3', 'C4', 'Shrubs', 'Urban', 'Lake', 'Soil', 'Ice']

    plt.close('all')
    fig, axes = plt.subplots(3,3, figsize=(15, 15), sharex=True, sharey=True)
    for i,ax in enumerate(axes.flatten()):
        ax.imshow(new_lct.data[i, :, :], interpolation='none', origin='lower')
        # xr.DataArray().from_iris(new_lct[i, :, :]).plot(ax=ax, vmin=0, vmax=0.5)
        ax.set_title(tile_titles[i])
        ax.xaxis.label.set_visible(False)

    # tight layout with some padding
    fig.tight_layout(h_pad=7)
    fig.savefig(output_path + '_no_urban.png', dpi=200, bbox_inches='tight')

    return

if __name__ == "__main__":

    parser = ants.AntsArgParser(target_lsm=True)
    args = parser.parse_args()

    target_lct_path = args.sources[0]
    target_lsm_path = args.target_lsm
    output_path = args.output

    main()