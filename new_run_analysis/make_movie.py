'''
Usage: python make_movie.py <glob_pattern> [output_name] [fps] [quality]
Example: python make_movie.py "/scratch/.../plots/temp_scrn/temp_scrn_sydney_*.png" temp_scrn_sydney 24
Gadi environment: module purge; module use /g/data/xp65/public/modules; module load conda/analysis3
'''
import glob
import os
import sys

import imageio.v2 as imageio

def make_mp4(fnamein, fnameout, fps=12, quality=26):
    '''
    Uses ffmpeg to create mp4 with custom codec and options for maximum compatability across OS.
        fnamein (string): The image files to create animation from, with glob wildcards (*) accepted.
        fnameout (string): The output filename (excluding extension)
        fps (float): The frames per second. Default 12.
        quality (float): quality ranges 0 to 51, 51 being worst. Default 26.
    '''

    # collect animation frames
    fnames = sorted(glob.glob(fnamein))
    img_shp = imageio.imread(fnames[0]).shape
    out_h, out_w = img_shp[0],img_shp[1]

    # resize output to blocksize for maximum capatability between different OS
    macro_block_size=16
    if out_h % macro_block_size > 0:
        out_h += macro_block_size - (out_h % macro_block_size)
    if out_w % macro_block_size > 0:
        out_w += macro_block_size - (out_w % macro_block_size)

    # quality ranges 0 to 51, 51 being worst.
    assert 0 <= quality <= 51, "quality must be between 1 and 51 inclusive"

    # use ffmpeg command to create mp4
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{fnamein}" \
        -vcodec libx264 -crf {quality} -s {out_w}x{out_h} -pix_fmt yuv420p -y {fnameout}.mp4'
    os.system(command)

    print(f"Movie saved: {fnameout}.mp4")
    return f"completed, see: {fnameout}.mp4"


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python make_movie.py <glob_pattern> [output_name] [fps] [quality]")
        print('  e.g. python make_movie.py "/scratch/gb02/mjl561/um2nc/SY_djf/SY_1/plots/temp_scrn/*.png" temp_scrn_diff 12 26')
        sys.exit(1)

    fnamein = sys.argv[1]

    # default output name: derived from input pattern directory
    prefix = fnamein.split("*")[0]
    plot_dir = os.path.abspath(prefix) if prefix.endswith(os.sep) else os.path.dirname(os.path.abspath(prefix))
    if len(sys.argv) > 2:
        fnameout = os.path.join(plot_dir, sys.argv[2])
    else:
        fnameout = os.path.join(plot_dir, "movie")

    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    quality = int(sys.argv[4]) if len(sys.argv) > 4 else 26

    matched = sorted(glob.glob(fnamein))
    if not matched:
        print(f"No files matched: {fnamein}")
        sys.exit(1)
    print(f"Found {len(matched)} frames")

    result = make_mp4(fnamein, fnameout, fps=fps, quality=quality)
    print(result)