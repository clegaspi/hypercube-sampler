from hypercube_sampler.constraints import Constraint
from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
from functools import partial


class Sampler:
    @staticmethod
    def _apply_constraint_func(func, x: np.array) -> float:
        """Evaluate constraint function for its value.

        :param func: Python code object representing constraint function
        :param x: list or array on which to evaluate the constraints
        :return: value from evaluating the constraint equation
        """
        return eval(func)

    @staticmethod
    def _step_to_next_point(current_point: np.array,
                            step_vector: np.array,
                            magnitude: float) -> np.array:
        """
        Generate the coordinates of a point after stepping from starting point.

        :param current_point: array representing starting point
        :param step_vector: unit vector representing the direction to step
        :param magnitude: step size as float
        :return: array representing end point
        """
        return np.add(current_point, magnitude * step_vector)

    @staticmethod
    def _apply_constraint_to_step_candidate(constraint_func,
                                            current_point: np.array,
                                            step_vector: np.array,
                                            magnitude: float) -> float:
        """
        Apply constraint function to a point created by stepping from
        starting point.
        :param constraint_func: Python code object representing constraint function
        :param current_point: array representing starting point
        :param step_vector: unit vector representing the direction to step
        :param magnitude: step size as float
        :return: value from evaluating the constraint equation at the end point
        """
        return Sampler._apply_constraint_func(
            constraint_func,
            Sampler._step_to_next_point(current_point, step_vector, magnitude)
        )

    @staticmethod
    def sample(constraint_file: str, n_samples: int) -> list:
        """
        Get samples from a polytope created on a hypercube from nonlinear constraints
        stored in a constraint definition file. Randomly samples points using
        random walk.

        :param constraint_file: file path to constraint definition file
        :param n_samples: number of samples to return
        :return: list of lists representing valid points on the polytope
        """

        # Read and compile constraint info
        constraint = Constraint(constraint_file)
        constraint_funcs = constraint.get_constraint_funcs()

        # If the constraint file does not contain a valid example point,
        # search for a valid point (on the edge of the polytope). Otherwise,
        # set the valid example point as the starting point of the random walk.
        if not constraint.apply(constraint.example):
            # This assumes all constraint functions are of the form:
            # g(x) >= 0
            constraint_objs = [
                NonlinearConstraint(
                    partial(Sampler._apply_constraint_func, func),
                    0, np.inf)
                for func in constraint_funcs
            ]
            result = minimize(
                lambda x: 0,
                [0.5]*constraint.n_dim,
                constraints=constraint_objs
            )
            current_pt = result.x
        else:
            current_pt = np.array(constraint.example)

        samples = []

        while len(samples) < n_samples:
            # Calculate random unit vector for direction to step
            step_vector = np.random.uniform(-1, 1, size=(constraint.n_dim,))
            step_vector = step_vector / np.linalg.norm(step_vector)

            # Find maximum magnitude for step within polytope
            get_next_point = partial(Sampler._step_to_next_point, current_pt, step_vector)

            # Assumes g(x) >= 0 form of constraint equation
            constraint_objs = [
                NonlinearConstraint(
                    partial(Sampler._apply_constraint_to_step_candidate,
                            func, current_pt, step_vector),
                    0, np.inf)
                for func in constraint_funcs
            ]
            solution = minimize(
                lambda m: -m,
                [0.0],
                constraints=constraint_objs
            )
            max_magnitude = solution.x[0]

            # Select random magnitude in range from 0 to max magnitude
            rand_magnitude = np.random.uniform(0, max_magnitude)
            n_retries = 10
            while not constraint.apply(get_next_point(rand_magnitude)) \
                    and n_retries > 0:
                # If the magnitude selected falls outside of the polytope,
                # try another, up to 10 times.
                rand_magnitude = np.random.uniform(0, max_magnitude)
                n_retries -= 1

            if not constraint.apply(get_next_point(rand_magnitude)):
                # If we couldn't find a good step to make in this direction,
                # pick a new direction.
                continue

            # Step to new point and pick a new direction
            current_pt = get_next_point(rand_magnitude)
            samples.append(current_pt.tolist())

        return samples