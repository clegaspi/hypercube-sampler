# hypercube-sampler
Created by Christian Legaspi, September 2019.

### Description
This package samples the feasible region (polytope) for a problem 
defined on a unit hypercube with non-linear constraints.

### Approach
This package uses a random walk to explore the feasible region (i.e.
the region that satisfies the constraints). From a valid starting point
specified in the constraint definition file, the algorithm selects a
direction uniformly at random and selects a magnitude for the step
uniformly at random, either forward or backward, such that it stays 
within the feasible region. If no such step size exists, then a new 
direction is selected. Once a valid direction and step size is found, 
the algorithm steps to the new point and process is repeated.

Currently, the algorithm only searches for valid step sizes such that
it does not step over any region along the selected direction that
violates the constraints. This algorithm could be improved by 
considering step sizes which may step over those regions and consider
all valid locations along that direction within the unit hypercube.

The implementation was created to be conscious of memory. The sampler
can be iterated so that points are generated on the fly instead of
being stored. However, the `sample(n)` function does return
a list with `n` points for convenience.

### Installation
- Clone repository
- Create your favorite virtual environment with python 3.7
  (I prefer conda, myself)
- Within the virtual environment and the repository root, 
  run `pip install --editable .`

### Running the sampler
This package features a command line interface. To run the sampler:
- Within your virtual environment, run `sampler <in-file> <out-file> <n_samples>`
- `<in-file>` is the path and name of a formatted constraint definition file
- `<out-file>` is the path and name of the output file
- `<n_samples>` is the number of points to find on the hypercube

Additionally the sampler can be used in a Python project via:

```
from hypercube_sampler.sampler import Sampler
sampler = Sampler.from_constraint_file('/path/to/constraint_file.txt')

# Get a set number of samples
samples = sampler.sample(100)

# Or get samples one at a time ad infinitum
for sample in sampler:
    my_processing_function(sample)
```

