# Polytunnel-Irradiance-Model

[![GitHub release](https://img.shields.io/github/release/CLOVER-energy/Polytunnel-Irradiance-Model)](https://GitHub.com/CLOVER-energy/Polytunnel-Irradiance-Model/releases/)

Python-based model capable of simultaing the spectral irradiance within curved geometries.

## Simulation

The script can be executed by either importing individual functions (see the documentation) or by running from the command-line interface as an integrated Python package. You can either specify parameters governing the model on the command-line or within an input file depending on which works best with your workflow. If importing the individual functions, you can also pass these values in provided that you use the correct structure (see the documentation for details).

### Command-line parameters

If you're passing all of the parameters needed in on the command-line interface (CLI), your command will look something like the following:
```bash
python -m polytunnel_irradiance_model --start_time_str 2024-06-06T00:00:00Z --end_time_str 2024-06-06T23:59:59Z --latitude 51.249814 --longitude 0.347779 --res_minutes 15 --length 2.5 --radius1 2.5 --radius2 1.5 --xy_angle 0 --z_angle 0 --transmissivity 1 --material_list ag ag ag ag ag ag --material_thick 80 80 80 80 80 80 --multistack 1 --cell_thickness 0.35 --cell_gap 2.8 --initial_cell_gap 2.1 --res_meshgrid 0.35
```

Where the individual parameters are given below in Table 1.

**Table 1.** Parameters included when running a simulation.

| Parameter        | Explanation                                                  |
| ---------------- | ------------------------------------------------------------ |
| `start_time_str` | The date and time that you want the simulation(s) to begin should be passed in with this keyword argument. The format is `YYYY-MM-DD` followed by `T` then `HH:MM:SS` with `Z` at the end to close the string. |
| `end_time_str`   | The end date and time for the simulation(s) also needs to be specified. The format is `YYYY-MM-DD` followed by `T` then `HH:MM:SS` with `Z` at the end to close the string. |
| `latitude`       | The latitude for which solar data should be used, specified in degrees North; _i.e._, positive number are for locations North of the Equator |
| `longitude`      | The longitude for which solar data should be used, specified in degrees East. |
| `res_minutes`    | The resolution, in minutes, to use for the modelling         |
| `length`         | The length of polytunnel section to model. Longer sections will increase the computation time but improve the ray-tracing modelling. In general, a repeated unit (or an integer number of repeated units) within a polytunnel can be modelled with sufficient Physics captured. |
| `z_angle`        |                                                              |
| `transmissivity` | The (optical) transmissivity of the material that makes up the polytunnel structure. |
| `material_list`  | The list of materials, defined in the `materials` data directory. |
| `material_thick` | The thickness of the various materials.                      |
| `multistack`     | The number of donor and acceptor layers that should be repeated. This defaults to `1` if it isn't defined. |
| `cell_thickness` | The thickness of the cell, _i.e._, its width along the length of the polytunnel, in metres. |
| `cell_gap`       | The "gap" between consecutive cells along the length of the polytunnel, _i.e._, the length of space where just polytunnel material, uncovered, is exposed, in metres. |
| `res_mesh_grid`  | The resolution of the mesh grid to use, in metres.           |

### File-based parameters

If you're using a file, then these parameters can be specified using a `YAML` file (a file which ends with the `.yaml` extension) and then simply point the package to this file:

```bash
python -m polytunnel_irradiance_model -f <input_file.yaml>
```

The parameters that need to be specified in the file are given in the above table and should use a key-value structure. An example is provided in the Wiki pages which can be downloaded.

**NOTE:** In the example input above, silver (`ag`) is used as all of the layers of the solar cell. 6 layers are used as this corresponds to the number of layers in an organic solar cell.

### Running without PV modules

If you want to simulate the light pattern within a polytunnel without any PV modules on its surface, then several variables need to be set to zero, either on the CLI,

```bash
--cell-thickness 0.0 --cell-gap 0.0
```

or within the YAML input file

```yaml
  ...
  cell_thickness: 0.0
  cell_gap: 0.0
  ...
```



## Modelling considerations

When validating the profiles considered, the spectral sensitivity of any device needs to be included. Here, a BF5 sensor is included (a Pyranometer---a device which detects and measures solar irradiance) has been included which can be used.

Spectral data can be found in the additional-resources directory.