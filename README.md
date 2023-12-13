# Earth-Mars Distance Calculator
*Lemonaizeris*

*Python Version 3.0+*

## Description
A program to:
1) Simulate Solar System's inner planets' orbits for the period of 2024-2034.
2) Calculate distance between Earth and Mars for the same period 2024-2034.

As an output it creates:
1) A folder *current_plots* with images of each simulated day of 2024-2034.
2) A video file of the images of all simulated days for the 2024-2034.


## Requirements

### Python modules

Python dependencies are in the file *pythonDependencyRequirements.txt*

### NASA NAIF SPICE kernels

To have this working you need following SPICE kernel files that you can get from NASA's NAIF
https://naif.jpl.nasa.gov/pub/naif/

The following SPICE files are required and can be found un generic_kernels folder:

1) naif0012.tls
2) de430.bsp
3) mars_iau2000_v1.tpc 
4) pck00011.tpc

From MARS2020 mission folder you need:
1) mar097s.bsp

## Usage

Run Command:

	python EarthMarsDistanceCalculator.py