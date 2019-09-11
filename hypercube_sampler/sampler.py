from functools import partial
from itertools import islice

from scipy.optimize import minimize, NonlinearConstraint, Bounds
import numpy as np

from hypercube_sampler.constraints import Constraint


class Sampler:
    """Draws random samples from feasible region for constraint problem on unit hypercube"""
    def __init__(self, constraint: Constraint, step_tol: float = 1e-8):
        """
        Build sampler object.

        :param constraint: constraint object to define constraints
        :param step_tol: magnitude tolerance for step size. If step is smaller
            than the magnitude, will be considered the same point.
        """
        self.constraint = constraint
        self.step_tol = step_tol
        self._constraint_funcs = self.constraint.get_constraint_funcs()
        self._current_pt = np.array(constraint.get_example())

    @classmethod
    def from_constraint_file(cls, input_file: str, step_tol: float = 1e-8):
        """
        Create sampler directly from constraint input file

        :param input_file: path to constraint definition file
        :param step_tol: magnitude tolerance for step size. If step is smaller
            than the magnitude, will be considered the same point.
        :return: Sampler object built from constraint definition
        """
        return cls(Constraint(input_file), step_tol=step_tol)

    def sample(self, n_samples: int) -> list:
        """
        Get a specific number of samples from a polytope created on a hypercube from nonlinear constraints
        stored in a constraint definition file. Randomly samples points using random walk.

        :param n_samples: number of samples to return
        :return: list of lists representing valid points on the polytope
        """

        return list(islice(self, n_samples))

    def __iter__(self):
        """Makes sampler iterable to generate points on the fly"""
        return self

    def __next__(self):
        """
        Generates points in the feasible region using a random walk.

        :return: vector representing a point in the feasible region.
        """
        # If the constraint file does not contain a valid example point,
        # search for a valid point (on the edge of the polytope). Otherwise,
        # set the valid example point as the starting point of the random walk.
        if not self.constraint.apply(self._current_pt):
            # This assumes all constraint functions are of the form:
            # g(x) >= 0
            constraint_objs = [
                NonlinearConstraint(
                    partial(Sampler._apply_constraint_func, func),
                    0, np.inf)
                for func in self._constraint_funcs
            ]
            result = minimize(
                lambda x: 0,
                [0.5] * self.constraint.n_dim,
                constraints=constraint_objs,
                bounds=Bounds([0] * self.constraint.n_dim, [1] * self.constraint.n_dim)
            )
            self._current_pt = result.x

        successful_step = False

        while not successful_step:
            # Calculate random unit vector for direction to step
            step_vector = np.random.uniform(-1, 1, size=(self.constraint.n_dim,))
            step_vector = step_vector / np.linalg.norm(step_vector)

            # Find minimum and maximum magnitude for step within polytope.
            # Assumes g(x) >= 0 form of constraint equation and that the
            # feasible region is contiguous. Will not step over any region
            # that violates constraints.
            constraint_objs = [
                NonlinearConstraint(
                    partial(Sampler._apply_constraint_to_step_candidate,
                            func, self._current_pt, step_vector),
                    0, np.inf)
                for func in self._constraint_funcs
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
            # If unsuccessful minimization or we're in a region of the
            # polytope where we can't move forward or backward, find another direction
            if not min_solution.success or not max_solution.success or \
                    (abs(min_solution.x[0]) < self.step_tol and
                     abs(max_solution.x[0]) < self.step_tol):
                continue
            min_magnitude = min_solution.x[0]
            max_magnitude = max_solution.x[0]

            # Select random magnitude in range from min to max magnitude
            rand_magnitude = np.random.uniform(min_magnitude, max_magnitude)
            n_retries = 10
            next_point = self._get_new_point_after_step(
                self._current_pt, step_vector, rand_magnitude)
            while (not self._is_valid_point(next_point) or
                   abs(rand_magnitude) < self.step_tol) and n_retries > 0:
                # If the new point selected falls outside of the polytope or hypercube,
                # or is the same as the starting point, try another magnitude,
                # up to 10 times.
                rand_magnitude = np.random.uniform(min_magnitude, max_magnitude)
                next_point = self._get_new_point_after_step(
                    self._current_pt, step_vector, rand_magnitude)
                n_retries -= 1

            if (not self._is_valid_point(next_point) or
                    abs(rand_magnitude) < self.step_tol):
                # If we couldn't find a good step to make in this direction,
                # pick a new direction.
                continue

            # Step to new point and pick a new direction
            self._current_pt = next_point
            successful_step = True

        return self._current_pt.tolist()

    @staticmethod
    def _apply_constraint_func(func, x: np.array) -> float:
        """Evaluate constraint function for its value.

        :param func: Python code object representing constraint function
        :param x: list or array on which to evaluate the constraints
        :return: value from evaluating the constraint equation
        """
        return eval(func)

    @staticmethod
    def _get_new_point_after_step(current_point: np.array,
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
            Sampler._get_new_point_after_step(current_point, step_vector, magnitude)
        )

    def _is_valid_point(self, point: np.array) -> bool:
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
