# hypercube-sampler
Sampling from a polytope on a hypercube with non-linear constraints.

To install:
- Clone repository
- Create your favorite virtual environment with python 3.7
- In the repository root, run `pip install --editable .`

This package features a command line interface. To run the sampler:
- Within your virtual environment, run `sampler <in-file> <out-file> <n_samples>`
- `<in-file>` is the path and name of a formatted constraint definition file
- `<out-file>` is the path and name of the output file
- `<n_samples>` is the number of points to find on the hypercube
