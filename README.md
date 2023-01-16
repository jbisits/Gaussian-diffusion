# Gaussian-diffusion

This repository has been archived as it is not maintained.

Notebooks to look at diffusion of Gaussian initial conditions via the diffusion equaiton.

**Note: There was an update to `Plots.jl` which has changed the way colour limits of the `heatmap` are set.
This has resulted in the `.gif` images not matching what is presented in my thesis.
When I have the time I will fix this and the output will match that as I present in the thesis.**

## Running the notebook
To run the notebooks first clone the repository 

```
git clone https://github.com/jbisits/QG_tracer_advection.git
```

then activate and instantiate to build the dependencies from the julia repl

```julia
julia>]
(@v1.6) pkg> activate .
(QG_tracer_advection) pkg> instantiate
```
The jupyter notebook `gaussdiff.ipynb` should now be able to be run provided the julia kernel provided by `IJulia` is intsalled.
Alternatively the notebook can be viewd by clicking on the `gaussdiff.ipynb` file above.

There is also a pluto notebook `Gaussdiff.jl` of the exact same material which can be opened prodvided `Pluto.jl` is installed.
