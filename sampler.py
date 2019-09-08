from constraints import Constraint
from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
from functools import partial


class Sampler:
    @staticmethod
    def _apply_constraint_func(func, x):
        """Evaluate constraint function for its value.

        :param func: Python compiled code representing function
        :param x: list or array on which to evaluate the constraints
        """
        return eval(func)

    @staticmethod
    def _step_to_next_point(current_point: np.array,
                            step_vector: np.array,
                            magnitude: float):
        """
        Generate the coordinates of a point after stepping from starting point.

        :param current_point: array representing starting point
        :param step_vector: unit vector representing the direction to step
        :param magnitude: step size as float
        :return: array representing end point
        """
        return np.add(current_point, magnitude * step_vector)

    @staticmethod
    def _apply_constraint_to_step_candidate(constraint_func, current_point,
                                            step_vector, magnitude):
        """
        Apply constraint function to a point created by stepping from
        starting point.
        :param constraint_func:
        :param current_point: array representing starting point
        :param step_vector: unit vector representing the direction to step
        :param magnitude: step size as float
        :return:
        """
        return Sampler._apply_constraint_func(
            constraint_func,
            Sampler._step_to_next_point(current_point, step_vector, magnitude)
        )

    @staticmethod
    def sample(constraint_file: str, n_samples: int) -> list:
        constraint = Constraint(constraint_file)
        constraint_funcs = constraint.get_constraint_funcs()

        if not constraint.apply(constraint.example):
            constraint_objs = [
                NonlinearConstraint(
                    partial(Sampler._apply_constraint_func, func),
                    0, np.inf)
                for func in constraint_funcs
            ]
            result = minimize(
                lambda x: 0,
                [0.1]*constraint.n_dim,
                constraints=constraint_objs
            )
            current_pt = result.x
        else:
            current_pt = np.array(constraint.example)

        samples = []

        while len(samples) < n_samples:
            step_vector = np.random.uniform(-1, 1, size=(constraint.n_dim,))
            step_vector = step_vector / np.linalg.norm(step_vector)

            get_next_point = partial(Sampler._step_to_next_point, current_pt, step_vector)

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

            rand_magnitude = np.random.uniform(0, max_magnitude)
            n_retries = 10
            while not constraint.apply(get_next_point(rand_magnitude)) \
                    and n_retries > 0:
                rand_magnitude = np.random.uniform(0, max_magnitude)
                n_retries -= 1

            if not constraint.apply(get_next_point(rand_magnitude)):
                # If we couldn't find a good step to make in this direction,
                # pick a new direction
                continue

            current_pt = get_next_point(rand_magnitude)
            samples.append(list(current_pt))

        return samples
