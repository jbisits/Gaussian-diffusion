### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ e85e0d54-9e76-4dfa-b0ea-233f0df7a8b5
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 82fd8d73-ec8c-4e26-99d4-e905db517b57
using Plots, Distributions, PlutoUI, PassiveTracerFlows

# ╔═╡ e70828ec-f1ad-11eb-0950-39b77264160a
md"""
# Diffusion of Gaussians

This notebook contains simulations of Gaussian initial conditions using the diffusion equation in one and two dimensions.
The purpose is to demonstrate how the area diagnostic (length in ond dimension) proposed in my thesis works for simple cases of diffusion.
For the full explanation plus diagnostics derivation look to chapter some number in my thesis (reference repo).

## One dimension

In one dimension we start with the initial condition for tracer concentration
```math
\begin{equation*}
    C_{0}(x, t; K) = \frac{1}{\sqrt{4\pi Kt}}\exp\Big(-\frac{x^2}{4Kt}\Big),
\end{equation*}
```
which is diffused by the diffusion equation
```math
\begin{equation*}
    \frac{\partial C}{\partial t} = K\frac{\partial^2 C}{\partial x^2}.
\end{equation*}
```

Here $x(t)$ is the position of the tracer at time $t$ and it is the second moment for $x(t)$  that grows linearly with time accroding to 
```math
    \sigma_{x}^{2} = 2Kt,
```
This corresponds to the mean squared displacement of a piece of tracer growing linearly with time.
We now show how a diagnostic of the second moment of the length of a tracer can be used to calculate the diffusion parameter $K$.

Reordering the concentration data from **highest concentration to lowest concentration** means that each tracer concentration has an associated cumulative tracer length.
We then calculate an approximation for the second moment of this length using this reordere data by
```math
\begin{equation*}
    \sigma_{l}^{2} \approx \frac{(\Delta l)^2 \sum_{i = 1}^{N}i^2C_{i}}{\sum_{i = 1}^{N}C_{i}}.
\end{equation*}
```

This is related to the mean squared displacement by
```math
    (2\sigma_{x})^{2} = 8Kt = \sigma_{l}^{2}
```
hence
```math
    K = \frac{\Delta\sigma_{l}^{2}}{8\Delta t}.
```

To verify that this diagnostic $(\sigma_{l}^{2})$ does grow linearly and can predict a diffusivity using we run a diffusion of a Gaussian in one dimension using the same spectral methods that will be used when in the presence of the QG flow.

### Setting up the simulation
First setup the problem using the `PassiveTracerFlows.jl` package and write a function to plot the output.
"""

# ╔═╡ 0f0396b5-06ac-4ad1-b68d-60e3a79f2147
begin
	dev = CPU() 
	nx, ny = 32, 1            
	stepper = "RK4"         
	dt = 0.002          
	nsteps = 7000            
	nsubs  = 50  
	Lx, Ly = 16, 1
	κ = 0.25
	uvel(x, y) = 1
	vvel(x, y) = 0
end

# ╔═╡ eb138be1-5e8d-4300-a8bd-864cc1af967e
md"""
Now setup a one dimensional advection diffusion problem from the `PassiveTracerFlows` package.
"""

# ╔═╡ 93e0ed3f-34ea-437e-ba14-b495172e7f90
begin
	prob_1d = TracerAdvectionDiffusion.Problem(dev; nx=nx, ny=ny, Lx=Lx, Ly = Ly, κ=κ,
                                        #steadyflow=true, u=uvel, v=vvel,
                                        dt=dt, stepper=stepper)
	sol, clock, vars, params = prob_1d.sol, prob_1d.clock, prob_1d.vars, prob_1d.params
	xgrid, ygrid = prob_1d.grid.x, prob_1d.grid.y

	C₀ = Array{Float64}(undef, prob_1d.grid.nx, prob_1d.grid.ny)
	for i in 1:prob_1d.grid.ny
		C₀[:, i] = pdf(Normal(0, 1), xgrid)
	end
	set_c!(prob_1d, C₀)
end

# ╔═╡ e8f4c98b-6987-43ef-a978-3d022fc6c9ff
function plot_output(prob)
	
	l = @layout Plots.grid(2, 1)
  	p = plot(layout=l, size = (700, 700))
	
	c = prob.vars.c
	grid = prob.grid
	x, y = grid.x, grid.y
	
	plot!(p[1], x, c,
			 legend = :false,
			 xlims = (-grid.Lx/2, grid.Lx/2),
			 xlabel = "x",
			 ylabel = "concentration, c(x)",
			  title = "Initial concentration",
            color = :green)
	
	t = prob.clock.t
	sorted_concentration = sort(reshape(c, :), rev = true)
	plot!(p[2], 1:length(sorted_concentration), sorted_concentration,
			label = false,
			xlabel = "Δl",
			ylabel = "concentration",
			title = "Initial concentration ordered highest to lowest",
			ylims = (0, findmax(sorted_concentration)[1]),
			color = :red)

	return p
	
end

# ╔═╡ a410047d-0df1-433e-8048-541d1d2dd547
md"""
### Diagnostic
Now code the diagnostic up, have used `k` for indexing rather than `i` but **this `k` is not realated to the diffusion that is being estimated.**
"""

# ╔═╡ 06560e7d-8d6f-4080-b2e7-82bc323ebe73
function second_mom(prob::FourierFlows.Problem)
	
	C = reshape(prob.vars.c, :)
	sort!(C, rev = true)
	N = length(C)
	
	Δl = prob.grid.Lx / prob.grid.nx
	
	Σk²Cₖ = (Δl)^2 * sum( [ k^2 * C[k] for k ∈ 1:N ] )
	ΣCₖ = sum(C)
	
	return Σk²Cₖ/ΣCₖ
	
end

# ╔═╡ ddca45d3-3211-433c-987b-d20bd5461c09
md"""
Now step the problem forwards and compute the diagnostic at each step. 
"""

# ╔═╡ a1b9097f-3dc1-44e2-b5f7-ca6114dcefcc
begin
	startwalltime = time()

	p = plot_output(prob_1d)
    savefig(p, "Initial1dconc.png") #Save initial concentration
	
	second_mom1d = Vector{Float64}(undef, round(Int, nsteps/nsubs) + 1)
	second_mom1d[1] = second_mom(prob_1d)
	tvec_1d = Vector{Float64}(undef, round(Int, nsteps/nsubs) + 1)
	tvec_1d[1] = clock.t

	anim = @animate for j = 0:round(Int, nsteps/nsubs)

	  p[1][1][:y] = vars.c
	  p[1][:title] = "Concentration, t=" * string(clock.t)
	  p[2][1][:y] = sort(reshape(vars.c, :), rev = true)
	
	  second_mom1d[j + 1] = second_mom(prob_1d)
	  tvec_1d[j + 1] = clock.t

	  stepforward!(prob_1d, nsubs)
	  TracerAdvectionDiffusion.updatevars!(prob_1d)
      
      if j == round(Int, nsteps/nsubs)
        p[1][:title] = "Final concentration"
        p[2][:title] = "Final concentration ordered highest to lowest"
        savefig(p, "Finia1dconc.png") #Save final concnetration
      end
	end
	mp4(anim, "1d_gaussiandiff.mp4", fps=12)
end

# ╔═╡ ab932243-db00-4688-b02e-94563601faac
md"""
### Looking at second moment and computing diffusivity
We have saved second moments from this simulation that we can plot.
"""

# ╔═╡ 17b860c1-7df7-4f20-8cb1-45462dba6b1f
begin
	oneddiffplot = plot(tvec_1d, second_mom1d, label = false,
						xlabel = "t", ylabel = "σ̂²ₗ(t)", 
						title = "Growth of σ̂ₗ² during a diffusion \n simulation of a Gaussian initial condition")
	oneddiffplot
end

# ╔═╡ a0d77063-1584-46b2-93a9-11660bff68f4
md"""
The linear growth suggests we have constant diffusivity which we now calculate using
```math
\begin{equation*}
    K = \frac{\Delta\hat{\sigma}_{l}^{2}}{8\Delta t}.
\end{equation*}
```
"""

# ╔═╡ eee83a1f-fc65-4dba-9db9-3281c8e5944e
begin
	Δσ²_1d = second_mom1d[end] - second_mom1d[1]
	Δt_1d = tvec_1d[end] - tvec_1d[1]
	K1d = Δσ²_1d / (8 * Δt_1d)
end

# ╔═╡ 01b9622e-8a71-4fd7-b2db-96db24194c5b
md"""
Here we see that this is very good approximation to the parameterised value of $\kappa = 0.25$ that was set in the simulation problem.
"""

# ╔═╡ 727f3f63-453f-475b-b1ee-2f1b14fdf757
md"""
## Two dimensions

In two dimesions we consider two initial conditions of tracer concentration, a "Gaussian blob" and a "Gaussian band".
As they are different distribtuions they require different diagnostics.
The band will use something similar to the one dimensional case but will only look at the **growing of the meridional second moment** $\hat{\sigma}_{y}^{2}$.
The blob will look at the growth of the area of circle with radius equal to the radial second moment $\hat{\sigma}_{r}^{2}$ or $\langle r^2 \rangle.

### Gaussian blob

Here the initial condition for tracer concentration is
```math
\begin{equation*}
    C_{0}(x, y, t; K) = \frac{1}{4\pi Kt}\exp-\Big(\frac{r^2}{4Kt} \Big)
\end{equation*}
```
which is diffused isotropically by the two dimensional diffusion equaiton
```math
    \frac{\partial c}{\partial t} = \nabla^{2}c.
```
The theory is laid out in the thesis so instead jump to the result that the area of the tracer grows as a circle linearly.
By ordering the tracer concentration from highest to lowest each tracer concentration will have an associated cumulative area.
This data reordering then allows the average area $\langle A \rangle$ that a piece of tracer will fall in which we can relate to a diffusivity
```math
\begin{equation*}
    \frac{\langle A \rangle}{\int C(A)dA} = 4\pi K t \approx \frac{\Delta A \sum_{i = 1}^{N}i C_{i}}{\sum_{i = 1}^{N}C_{i}}
\end{equation*}
```
Now setup a simulation, plot function and the function to calculate the average area for a diffusion simulation of a Gaussian blob.
"""

# ╔═╡ 88d2ad2a-e7ac-4999-9201-f2ff6c7c524f
begin
	prob_2d = TracerAdvectionDiffusion.Problem(dev; nx=nx, Lx=Lx, κ=κ,
                                                #steadyflow=true, u=uvel, v=vvel,
                                                dt=dt, stepper=stepper)
	strip = Normal(0, 1)
	strip_IC(x) = pdf(strip, x)
	C₀2d = Array{Float64}(undef, prob_2d.grid.nx, prob_2d.grid.ny)
	xgrid2d, ygrid2d = prob_2d.grid.x, prob_2d.grid.y

	#Gaussian blob
	C₀2d = [pdf(MvNormal([0, 0], [1 0;0 1]), [x, y]) for y in ygrid2d, x in xgrid2d]
	set_c!(prob_2d, C₀2d)
end

# ╔═╡ 83561a5a-dd69-4fd2-aaca-8811bbc295d1
function plot_output_2d(prob)
  	
	l = @layout Plots.grid(2, 1)
  	p = plot(layout=l, size = (600, 500))
	
	c = prob.vars.c
  	grid = prob.grid
	x, y = grid.x, grid.y
	
  	heatmap!(p[1], x, y, c',
			 aspectratio = 1,
				  c = :balance,
		   colorbar = true,
			 legend = :false,
			  xlims = (-grid.Lx/2, grid.Lx/2),
			  ylims = (-grid.Ly/2, grid.Ly/2),
			 xlabel = "x",
			 ylabel = "y",
			  title = "Initial concentration",
		 framestyle = :box)
	
	t = prob.clock.t
	sorted_concentration = sort(reshape(c, :), rev = true)
	plot!(p[2], 1:length(sorted_concentration), sorted_concentration,
			label = false,
			xlabel = "ΔA",
			ylabel = "Concentration",
			title = "Initial concentration ordered highest to lowest",
			ylims = (0, findmax(sorted_concentration)[1]),
			color = :green)
	
	return p
end

# ╔═╡ 5b7b359a-0971-43b5-941f-bbf58049afc2
function first_mom(prob::FourierFlows.Problem)
	
	C = reshape(prob.vars.c, :)
	sort!(C, rev = true)
	N = length(C)
	
	Δx = prob.grid.Lx / prob.grid.nx
	Δy = prob.grid.Ly / prob.grid.ny
    ΔA = Δx * Δy
	
	ΣkCₖ = ΔA * sum( [ k * C[k] for k ∈ 1:N ] )
	ΣCₖ = sum(C)
	
	
	return ΣkCₖ/ΣCₖ
	
end

# ╔═╡ a1dfb19a-ddf6-4ffa-8a78-7e0c01076af9
begin
	p_2d = plot_output_2d(prob_2d)

	savefig(p_2d, "Initialblob2d.png")

	first_mom2d = Vector{Float64}(undef, round(Int, nsteps/nsubs) + 1)
	first_mom2d[1] = first_mom(prob_2d)
	tvec_2d = Vector{Float64}(undef, round(Int, nsteps/nsubs) + 1)
	tvec_2d[1] = prob_2d.clock.t

	anim_2d = @animate for j = 0:round(Int, nsteps/nsubs)

		p_2d[1][1][:z] = prob_2d.vars.c
		p_2d[1][:title] = "Concentration, t=" * string(prob_2d.clock.t)
		p_2d[2][1][:y] =  sort(reshape(prob_2d.vars.c, :), rev = true)

		first_mom2d[j + 1] = first_mom(prob_2d)
		tvec_2d[j + 1] = prob_2d.clock.t

		stepforward!(prob_2d, nsubs)
		TracerAdvectionDiffusion.updatevars!(prob_2d)
		if j == round(Int, nsteps/nsubs)
			p_2d[1][:title] = "Final concentration"
			p_2d[2][:title] = "Final concentration ordered highest to lowest"
			savefig(p_2d, "Finialblob2d.png")
		end
	end

	mp4(anim_2d, "2d_gaussiandiff_blob.mp4", fps=12)
end

# ╔═╡ 3da404c0-7264-454c-bb1d-7d91d37ea410
md"""
Now look at the growth of the average area and if it is linear as the theory suggests we calculate the diffusivity by 
```math
\begin{equation*}
    K = \frac{\Delta \langle A \rangle}{4 \pi \Delta t}
\end{equation*}
```
"""

# ╔═╡ 895c0d83-5799-4169-9d4d-fb042d808823
begin
	twoddiffplot = plot(tvec_2d, first_mom2d, label = false, xlabel = "t", ylabel = "⟨A⟩(t)", title = "Growth of average area during an isotropic diffusion \n simulation of a Gaussian blob")
	twoddiffplot
end

# ╔═╡ 8ab3a069-9de3-41da-b26a-9a7ead36bd6d
begin
	Δt_2d = tvec_2d[end] - tvec_2d[1]
	Δ𝔼A2d = first_mom2d[end] - first_mom2d[1]
	K2d_blob = Δ𝔼A2d / (4 * π * Δt_2d) 
end

# ╔═╡ 41357cfa-4fed-4d8a-b73b-55df53c2aa6b
md"""
Again we see that the diffusion calculated is a very good estimation to the diffusion set in the problme $\kappa = 0.25$.

### Gaussian band simulation
To look at the band we reset the two dimensional problem to have the initial condition and run the simulation using this.
**Be sure to rerun correct cells prior to running simulaitons!**.

The area of the Gaussian band grows as a rectanlge that has width $L_{x}$ (zonal doamain) and length $2\sigma_{y}$ (meridional width of the Gaussian set at each zonal gridpoint).
This does not grow linearly in time (it grows proportional to $\sqrt{t}) so instead we consider how the area squared grows
```math
A^{2} = L_{x}^{2}8\sigma_{y}^{2}.
```
By ordering the concetration from higest to lowest we then calculate and estimate for the second moment of the area by
```math
\sigma_{A}^{2} \approx \frac{(\Delta A)^{2}\sum_{i=1}^{N}i^{2}C_{i}}{\sum_{i=1}^{N}C_{i}}
```
which is related to a diffusivity by
```math
K \approx \frac{\Delta \sigma_{A}^{2}}{8L_{x}^{2}\Delta t}.
```
Now setup a simulation of the diffusion of a Gaussian band initial condition via the two dimensional diffusion equation and see if it accurately calculates the diffusivity.
"""

# ╔═╡ dc1f17c4-c02e-44b7-80eb-c1c5380d2a3c
begin
	prob_2d_band = TracerAdvectionDiffusion.Problem(dev; nx=nx, Lx=Lx, κ=κ,
                                                #steadyflow=true, u=uvel, v=vvel,
                                                dt=dt, stepper=stepper)
	C₀2d_band = Array{Float64}(undef, prob_2d_band.grid.nx, prob_2d_band.grid.ny)
	xgrid2d_band, ygrid2d_band = prob_2d_band.grid.x, prob_2d_band.grid.y

	for i in 1:prob_2d_band.grid.ny
	C₀2d_band[i, :] = pdf(Normal(0, 1), ygrid2d_band)
	end

	set_c!(prob_2d_band, C₀2d_band)
end

# ╔═╡ c02215b6-ea5e-4984-9d2a-88a9f9ad5afb
function second_mom_2d(prob::FourierFlows.Problem)
	
    C = reshape(prob.vars.c, :)
	sort!(C, rev = true)
	N = length(C)
	
	Δx = prob.grid.Lx / prob.grid.nx
    Δy = prob.grid.Ly / prob.grid.ny
    ΔA = (Δx * Δy) 
	
	Σk²Cₖ = (ΔA)^2 * sum( [ k^2 * C[k] for k ∈ 1:N ] )
	ΣCₖ = sum(C)
	
	return Σk²Cₖ/ΣCₖ
	
end

# ╔═╡ 0316a606-8f76-4145-9962-8ca10dbf3a7a
begin
	p_2d_band = plot_output_2d(prob_2d_band)

	savefig(p_2d_band, "Initialband2d.png")

	second_mom2d = Vector{Float64}(undef, round(Int, nsteps/nsubs) + 1)
	second_mom2d[1] = second_mom_2d(prob_2d)
	tvec_2d_band = Vector{Float64}(undef, round(Int, nsteps/nsubs) + 1)
	tvec_2d_band[1] = prob_2d_band.clock.t

	anim_2d_band = @animate for j = 0:round(Int, nsteps/nsubs)

    p_2d_band[1][1][:z] = prob_2d_band.vars.c
    p_2d_band[1][:title] = "Concentration, t=" * string(prob_2d_band.clock.t)
    p_2d_band[2][1][:y] =  sort(reshape(prob_2d_band.vars.c, :), rev = true)

    second_mom2d[j + 1] = second_mom_2d(prob_2d_band)
    tvec_2d[j + 1] = prob_2d.clock.t
    
    stepforward!(prob_2d_band, nsubs)
    TracerAdvectionDiffusion.updatevars!(prob_2d_band)
    TracerAdvectionDiffusion.updatevars!(prob_2d_band)
    if j == round(Int, nsteps/nsubs)
        p_2d[1][:title] = "Final concentration"
        p_2d[2][:title] = "Final concentration ordered highest to lowest"
        savefig(p_2d_band, "Finialband2d.png")
    end
end

mp4(anim_2d_band, "2d_gaussiandiff_strip.mp4", fps=12)
end

# ╔═╡ 47f0ebc2-7cdd-4f3b-a76e-6594ddce2e8a
md"""
Again we look to find a linear relationship and use this to estimate the diffusivity.
"""

# ╔═╡ 1776e7f6-d3d8-4650-9e4b-dea3e5cca5a3
begin
	twoddband = plot(tvec_2d_band, second_mom2d, label = false, xlabel = "t", ylabel = "σₐ²(t)", title = "Growth of σₐ² during an isotropic diffusion \n simulation of a Gaussian band")
	twoddband
end

# ╔═╡ 7d8632cc-be39-4a02-aefe-6cd263cb5c46
begin
	σ²ₐ = second_mom2d
	Δσ²ₐ = σ²ₐ[end] - σ²ₐ[1]
	Δt_2d_band = tvec_2d[end] - tvec_2d[1]
	K2d_band = Δσ²ₐ / (prob_2d.grid.Lx^2 * 8 * Δt_2d)
end

# ╔═╡ e95b9003-08e7-4aa8-864e-9e397a18bc14
TableOfContents(title = "Diffusion of Gaussians")

# ╔═╡ Cell order:
# ╠═e85e0d54-9e76-4dfa-b0ea-233f0df7a8b5
# ╠═82fd8d73-ec8c-4e26-99d4-e905db517b57
# ╟─e70828ec-f1ad-11eb-0950-39b77264160a
# ╠═0f0396b5-06ac-4ad1-b68d-60e3a79f2147
# ╟─eb138be1-5e8d-4300-a8bd-864cc1af967e
# ╠═93e0ed3f-34ea-437e-ba14-b495172e7f90
# ╠═e8f4c98b-6987-43ef-a978-3d022fc6c9ff
# ╟─a410047d-0df1-433e-8048-541d1d2dd547
# ╠═06560e7d-8d6f-4080-b2e7-82bc323ebe73
# ╟─ddca45d3-3211-433c-987b-d20bd5461c09
# ╠═a1b9097f-3dc1-44e2-b5f7-ca6114dcefcc
# ╟─ab932243-db00-4688-b02e-94563601faac
# ╠═17b860c1-7df7-4f20-8cb1-45462dba6b1f
# ╟─a0d77063-1584-46b2-93a9-11660bff68f4
# ╠═eee83a1f-fc65-4dba-9db9-3281c8e5944e
# ╟─01b9622e-8a71-4fd7-b2db-96db24194c5b
# ╟─727f3f63-453f-475b-b1ee-2f1b14fdf757
# ╠═88d2ad2a-e7ac-4999-9201-f2ff6c7c524f
# ╠═83561a5a-dd69-4fd2-aaca-8811bbc295d1
# ╠═5b7b359a-0971-43b5-941f-bbf58049afc2
# ╠═a1dfb19a-ddf6-4ffa-8a78-7e0c01076af9
# ╟─3da404c0-7264-454c-bb1d-7d91d37ea410
# ╠═895c0d83-5799-4169-9d4d-fb042d808823
# ╠═8ab3a069-9de3-41da-b26a-9a7ead36bd6d
# ╟─41357cfa-4fed-4d8a-b73b-55df53c2aa6b
# ╠═dc1f17c4-c02e-44b7-80eb-c1c5380d2a3c
# ╠═c02215b6-ea5e-4984-9d2a-88a9f9ad5afb
# ╠═0316a606-8f76-4145-9962-8ca10dbf3a7a
# ╟─47f0ebc2-7cdd-4f3b-a76e-6594ddce2e8a
# ╠═1776e7f6-d3d8-4650-9e4b-dea3e5cca5a3
# ╠═7d8632cc-be39-4a02-aefe-6cd263cb5c46
# ╟─e95b9003-08e7-4aa8-864e-9e397a18bc14
