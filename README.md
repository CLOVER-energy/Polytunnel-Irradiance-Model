# Polytunnel-Irradiance-Model

[![GitHub release](https://img.shields.io/github/release/CLOVER-energy/Polytunnel-Irradiance-Model)](https://GitHub.com/CLOVER-energy/Polytunnel-Irradiance-Model/releases/)

Python-based model capable of simultaing the spectral irradiance within curved geometries.

## Simulation

The script can be executed by either importing individual functions (see the documentation) or by running from the command-line interface as an integrated Python package. You can either specify parameters governing the model on the command-line or within an input file depending on which works best with your workflow. If importing the individual functions, you can also pass these values in provided that you use the correct structure (see the documentation for details).

### Command-line parameters

If you're passing all of the parameters needed in on the command-line interface (CLI), your command will look something like the following:
```bash
python -m src.polytunnel_irradiance_model -pt circular_narrow_short -mres 10 -st 2024-03-01T00:00:00Z -et 2024-09-30T23:59:59Z -d 0.55 -vi 1050 -wf corrected_renewables_ninja_weather.csv -wado -mtr 60 --latitude 51.249814 --longitude 0.347779 
```

Where the individual parameters are given below in Table 1.

**Table 1.** Parameters included when running a simulation.

| Parameter                 | Explanation                                                          |
| ------------------------- | -------------------------------------------------------------------- |
| `--meshgrid-resolution`   | The resolution of the mesh grid to use, in metres. |
| `--weather-file`          | The name of the weather file to use. |
| `--weather-file-error`    | The error in the weather data file variables to use for calculating error bars. |
| `-wado`                   | "Weather as diffusivity only"; _i.e._, use the weather data only for diffusuvity information. |
| `--regenerate`            | Regenerate the profiles for the surface irradiance. Use this flag when the location has changed or irradiance information has otherwise altered. |
| `--regenerate-mesh`       | Regenerate (re-compute) the mesh sheets used for computing distances between the surface of the polytunnel and the ground. Use this flag when the resolution of the mesh has changed or other geometry factors need altering. |
| `--start-time`            | The start date and time for the simulation(s) also needs to be specified. The format is `YYYY-MM-DD` followed by `T` then `HH:MM:SS` with `Z` at the end to close the string. |
| `--end-time`              | The end date and time for the simulation(s) also needs to be specified. The format is `YYYY-MM-DD` followed by `T` then `HH:MM:SS` with `Z` at the end to close the string. |
| `--latitude`              | The latitude for which solar data should be used, specified in degrees North; _i.e._, positive number are for locations North of the Equator |
| `--longitude`             | The longitude for which solar data should be used, specified in degrees East. |
| `--altitude`              | The altitude for the location where the weather data should be used. |
| `-mtr`                    | "Modelling temporal resolution": The resolution, in minutes, to use for the modelling throughout the day. |
| `--validation-filename`   | The name of the file to use for validation purposes. |
| `--validation-index`      | The element to use for validation. |
| `--diffusivity`           | The diffusivity of the polytunnel material to use. This parameter governs the extent to which the material scatters incoming direct irradiance as diffuse irradiance. |
| `--polytunnel-input-file` | The name of the polytunnels input file to use. |
| `--polytunnel`            | The name of the polytunnel to use. |
| `--solar-cells-file`      | The path to the solar-cells materials inputs file. |

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
  cell_spacing: 0.0
  ...
```



## Modelling considerations

When validating the profiles considered, the spectral sensitivity of any device needs to be included. Here, a BF5 sensor is included (a Pyranometer---a device which detects and measures solar irradiance) has been included which can be used.

Spectral data can be found in the additional-resources directory.
