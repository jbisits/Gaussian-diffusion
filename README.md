# Gaussian-diffusion
Notebooks to look at diffusion of Gaussian initial conditions via the diffusion equaiton.

## Running the notebook
To run the notebooks first clone the repository 

```
git clone https://github.com/jbisits/QG_tracer_advection.git
```

then activate and instantiate to build the dependencies from the julia repl

```
julia>]
(@v1.6) pkg> activate .
(QG_tracer_advection) pkg> instantiate
```
The jupyter notebook `gaussdiff.ipynb` should now be able to be run provided the julia kernel provided by `IJulia` is intsalled.

There is also a pluto notebook `Gaussdiff.jl` of the exact same material.
