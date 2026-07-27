#!/bin/bash
#PBS -N plot_all_vars
#PBS -q normal
#PBS -P gb02
#PBS -l walltime=06:00:00
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l storage=gdata/xp65+scratch/gb02+gdata/gb02
#PBS -l wd
#PBS -l jobfs=10GB

set -euo pipefail

module use /g/data/xp65/public/modules
module load conda/analysis3-25.08

# location of post-processed netcdf data
output_root_dir="/g/data/gb02/mjl561/ram3_SY_urban/SY_djf/SY_1"
# output_root_dir="/scratch/gb02/mjl561/um2nc/SY_djf/SY_1"
variables_to_plot=(wind wndgust10m_scale wndgust10m)
# use "wind" to plot 10 m wind speed from uwnd10m_b/vwnd10m_b
# variables_to_plot="temp_scrn"
# movie settings (used for hour-by-hour frames)
movie_fps=6
movie_quality=26
movie_output_subdir="movies"
# # diurnal for Summer Hill
# lat="-33.891"
# lon="151.137"
# # diurnal for Parramatta CBD
lat="-33.813"
lon="151.003"

# Optional: provide AEST hours to plot hourly means, e.g. hours=(0 6 12 18).
# Leave empty to plot a single all-timesteps mean.
hours=()
hours=($(seq 0 23))

# spatial
if (( ${#hours[@]} > 0 )); then
  hours_csv=$(IFS=,; echo "${hours[*]}")
  echo "Processing AEST hours: ${hours_csv}"
  python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_vars.py \
    "$output_root_dir" "hours=${hours_csv}" "lat=${lat}" "lon=${lon}" "${variables_to_plot[@]}"
else
  echo "No hours passed: plotting all-timesteps mean"
  python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_vars.py \
    "$output_root_dir" "lat=${lat}" "lon=${lon}" "${variables_to_plot[@]}"
fi

# make movie from hourly frames (only when hour plots were generated)
if (( ${#hours[@]} > 0 )); then
  plots_dir="${output_root_dir}/plots"
  movie_output_dir="${plots_dir}/${movie_output_subdir}"
  vars_manifest="${plots_dir}/plotted_vars.txt"
  prefix="$(basename "${output_root_dir}")"
  mkdir -p "${movie_output_dir}"

  shopt -s nullglob
  target_vars=()

  # Use plot_vars.py manifest so movie targets match parsed/plotted vars exactly.
  if [[ -s "${vars_manifest}" ]]; then
    while IFS= read -r var_name; do
      [[ -n "${var_name}" ]] || continue
      target_vars+=("${var_name}")
    done < "${vars_manifest}"
    echo "Loaded $((${#target_vars[@]})) vars from manifest: ${vars_manifest}"
  else
    echo "Vars manifest missing or empty (${vars_manifest}); skipping movie creation"
    target_vars=()
  fi

  movies_created=0
  for var_name in "${target_vars[@]}"; do
    var_dir="${plots_dir}/${var_name}"
    [[ -d "${var_dir}" ]] || continue

    frames=("${var_dir}/${prefix}_${var_name}_"*"_hour"*.png)
    if (( ${#frames[@]} == 0 )); then
      continue
    fi

    frame_glob="${var_dir}/${prefix}_${var_name}_*_hour*.png"
    movie_name="${movie_output_dir}/${var_name}_hours"
    echo "Creating movie for ${var_name}: ${frame_glob} -> ${movie_name}.mp4"
    python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/make_movie.py \
      "$frame_glob" "$movie_name" "$movie_fps" "$movie_quality"
    movies_created=$((movies_created + 1))
  done

  if (( movies_created == 0 )); then
    echo "No hourly frames found in variable subdirectories of ${plots_dir}; skipping movie creation"
  fi
fi

# # diurnal for Sydney CBD
# lat="-33.8688"
# lon="151.2093"

# # diurnal for Summer Hill
# lat="-33.891"
# lon="151.137"

# python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_diurnal.py \
#   "$output_root_dir" "$lat" "$lon" $variables_to_plot
