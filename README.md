# slab_fitter
slab_fitter is a set of python codes to perform MCMC slab model fits to CO rotation digrams.  Code is currently
under development and is not ready for use.

# Requirements
Requires internet access to utilize astroquery.hitran and access HITRAN partition function files

Requires the molmass and astropy packages

## Functions
sf_run.init creates a model run

compute_fluxes calculates line fluxes from relevant HITRAN info and model parameters (column density, temperature and solid angle)

make_rotation_diagram makes a rotation diagram from line fluxes

## Usage

```python
myrun=sf_run(data)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

