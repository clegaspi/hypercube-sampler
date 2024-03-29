import click

from hypercube_sampler.sampler import Sampler
from hypercube_sampler.constraints import Constraint


@click.command()
@click.argument('input_file', required=True)
@click.argument('output_file', required=True)
@click.argument('n_results', required=True)
def main(input_file, output_file, n_results):
    """Runs the sampling algorithm on the problem defined in the
    constraint input file INPUT_FILE and outputs N_RESULTS number
    of sampled points to OUTPUT_FILE.
    """
    constraint = Constraint(input_file)
    sampler = Sampler(constraint)
    samples = sampler.sample(int(n_results))

    with open(output_file, 'w') as f:
        for point in samples:
            vector = " ".join(str(v) for v in point)
            f.write(vector + '\n')
