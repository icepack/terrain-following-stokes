# Terrain-following Stokes

This repository contains the text and source code for the paper *Simulating the flow of viscous, incompressible fluids with a free surface using terrain-following coordinates*, to be submitted to Geoscientific Model Development.

## Contents

The source code and demonstrations all require a working installation of the [Firedrake package](https://www.firedrakeproject.org).
This repository contains a Python library called `zetastokes` that lives under the directory `package/`.
Provided you have a working Firedrake installation, you can install `zetastokes` from the top-level directory with
```
pip install --editable package/
```
The `demo/` directory contains several demonstrations of the methods and mathematics.
These demonstrations all require the `zetastokes` package.


## License

The text of the paper in `stokes-tfc.tex` and `stokes-tfc.bib` and figures are released under the Creative Commons Attribution 4.0 International (CC BY 4.0).
The zetastokes library under `package/` and demonstrations under `demo/` are released under the GNU General Public License version 3.0 or later.
The `copernicus.cls`, `copernicus.cfg`, and `copernicus.bst` files are copyright Copernicus Publications and not covered by the above, and are necessary for compilation of the manuscript.
