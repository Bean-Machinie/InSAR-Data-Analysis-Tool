# InSAR Data Analysis Tool

Utilities for inspecting, clipping, and interactively viewing exported SBAS
InSAR products.

The scripts accept either the outer export bundle folder or the inner
`outputs/<project>_<orbit>` product folder. If no folder is provided they use
`INSAR_PROJECT_DIR`, then fall back to `Data/project_D_results_only` when it
exists.

```powershell
python inspect_insar_outputs.py Data\project_D_results_only
python clip_insar_to_aoi.py Data\project_D_results_only
python insar_deformation_viewer.py Data\project_D_results_only
```

Current standard SBAS variables include `sbas_displacement_raw`,
`sbas_displacement_masked`, `sbas_displacement_segmented_same_pixel`,
`sbas_velocity_raw`, `sbas_velocity_masked`, `coherence_median`, and
`valid_pixel_mask`.

The interactive viewer starts from a satellite basemap and lets you toggle
ground maps separately from SBAS data overlays. The overlays are built from the
NetCDF arrays, so selecting pixels still drives the displacement time-series
inspector rather than flattening the product into static raster images.
