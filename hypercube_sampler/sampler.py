from functools import partial

from scipy.optimize import minimize, NonlinearConstraint, Bounds
import numpy as np

from hypercube_sampler.constraints import Constraint


class Sampler:
    """Draws random samples from feasible region for constraint problem on unit hypercube"""
    def __init__(self, constraint: Constraint):
        """
        Build sampler object.

        :param constraint: constraint object to define constraints
        """
        self.constraint = constraint

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

    def sample(self, n_samples: int, step_tol: float = 1e-8) -> list:
        """
        Get samples from a polytope created on a hypercube from nonlinear constraints
        stored in a constraint definition file. Randomly samples points using
        random walk.

        :param constraint_file: file path to constraint definition file
        :param n_samples: number of samples to return
        :param step_tol: magnitude tolerance for step size. If step is smaller
            than the magnitude, will be considered the same point.
        :return: list of lists representing valid points on the polytope
        """

        # Read and compile constraint info
        constraint_funcs = self.constraint.get_constraint_funcs()

        # If the constraint file does not contain a valid example point,
        # search for a valid point (on the edge of the polytope). Otherwise,
        # set the valid example point as the starting point of the random walk.
        if not self.constraint.apply(self.constraint.example):
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
                [0.5]*self.constraint.n_dim,
                constraints=constraint_objs,
                bounds=Bounds([0]*self.constraint.n_dim, [1]*self.constraint.n_dim)
            )
            current_pt = result.x
        else:
            current_pt = np.array(self.constraint.example)

        samples = []

        while len(samples) < n_samples:
            # Calculate random unit vector for direction to step
            step_vector = np.random.uniform(-1, 1, size=(self.constraint.n_dim,))
            step_vector = step_vector / np.linalg.norm(step_vector)

            # Find minimum and maximum magnitude for step within polytope
            # Assumes g(x) >= 0 form of constraint equation
            constraint_objs = [
                NonlinearConstraint(
                    partial(Sampler._apply_constraint_to_step_candidate,
                            func, current_pt, step_vector),
                    0, np.inf)
                for func in constraint_funcs
            ]
            min_solution = minimize(
                lambda m: m,
                [0.0],
                constraints=constraint_objs,
                bounds=Bounds([-np.sqrt(self.constraint.n_dim)], [0])
            )
            max_solution = minimize(
                lambda m: -m,
                [0.0],
                constraints=constraint_objs,
                bounds=Bounds([0], [np.sqrt(self.constraint.n_dim)])
            )
            if not min_solution.success or not max_solution.success or \
                    (abs(min_solution.x[0]) < step_tol and
                     abs(max_solution.x[0]) < step_tol):
                # If unsuccessful minimization or we're on the edge of the
                # polytope and can't move in either direction, find another direction
                continue
            min_magnitude = min_solution.x[0]
            max_magnitude = max_solution.x[0]

            # Select random magnitude in range from 0 to max magnitude
            get_next_point = partial(Sampler._step_to_next_point, current_pt, step_vector)
            rand_magnitude = np.random.uniform(min_magnitude, max_magnitude)
            n_retries = 10
            next_point = get_next_point(rand_magnitude)
            while (not self._is_valid_point(next_point) or
                   abs(rand_magnitude) < step_tol) and n_retries > 0:
                # If the new point selected falls outside of the polytope or hypercube,
                # or is the same as the starting point, try another magnitude,
                # up to 10 times.
                rand_magnitude = np.random.uniform(min_magnitude, max_magnitude)
                next_point = get_next_point(rand_magnitude)
                n_retries -= 1

            if (not self._is_valid_point(next_point) or
                    abs(rand_magnitude) < step_tol):
                # If we couldn't find a good step to make in this direction,
                # pick a new direction.
                continue

            # Step to new point and pick a new direction
            current_pt = next_point
            samples.append(current_pt.tolist())

        return samples

    def _is_valid_point(self, point: np.array):
        """
        Determines if a selected point is on the hypercube and
        meets constraints.

        :param point: point to check
        :return: True if on the hypercube and meets constraints,
            false otherwise
        """
        return self.constraint.apply(point) \
            and np.all(point >= 0) \
            and np.all(point <= 1)
