import unittest
import os

import numpy as np

from hypercube_sampler.sampler import Sampler
from hypercube_sampler.constraints import Constraint

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class SamplerTest(unittest.TestCase):
    def test_sampler_good_starting_point(self):
        """Test running sampler with valid starting point"""
        input_path = os.path.join(TEST_DIR, '2d.txt')
        constraint = Constraint(input_path)
        sampler = Sampler(constraint)
        results = sampler.sample(100)
        self.assertEqual(len(results), 100)
        self.assertTrue(all(constraint.apply(pt) for pt in results))

    def test_sampler_bad_starting_point(self):
        """Test running sampler with invalid starting point, where
        the sampler has to find an edge first"""
        input_path = os.path.join(TEST_DIR, '2d-badstart.txt')
        constraint = Constraint(input_path)
        sampler = Sampler(constraint)
        results = sampler.sample(100)
        self.assertEqual(len(results), 100)
        self.assertTrue(all(constraint.apply(pt) for pt in results))

    def test_apply_constraint_func(self):
        """Test applying constraint function to a point to retrieve the value
        of the constraint function.
        """
        input_path = os.path.join(TEST_DIR, '2d.txt')
        constraint = Constraint(input_path)
        self.assertAlmostEqual(
            Sampler._apply_constraint_func(constraint.get_constraint_funcs()[1], [0, 1]),
            -0.2
        )
        self.assertAlmostEqual(
            Sampler._apply_constraint_func(constraint.get_constraint_funcs()[1], [0, 0.5]),
            0.3
        )

    def test_step_to_next_point(self):
        """Test stepping function"""
        current_pt = np.array([0.5, 0.5])
        step_vector = np.array([0, 1])
        step_vector = step_vector / np.linalg.norm(step_vector)
        mag = 0.2

        next_pt = Sampler._get_new_point_after_step(current_pt, step_vector, mag)
        self.assertTrue(np.allclose(next_pt, [0.5, 0.7]))

    def test_apply_constraint_to_step_candidate(self):
        """Test applying constraint function to a point that is stepped to
        in order to retrieve the value of the constraint function."""
        input_path = os.path.join(TEST_DIR, '2d.txt')
        constraint = Constraint(input_path)

        current_pt = np.array([0.5, 0.5])
        step_vector = np.array([0, 1])
        step_vector = step_vector / np.linalg.norm(step_vector)
        mag = 0.2

        value = Sampler._apply_constraint_to_step_candidate(
            constraint.get_constraint_funcs()[1],
            current_pt, step_vector, mag)
        self.assertAlmostEqual(value, 0.1)

    def test_is_valid_point(self):
        """Test if points lie in the unit hypercube and satisfy constraints"""
        input_path = os.path.join(TEST_DIR, '2d.txt')
        constraint = Constraint(input_path)
        sampler = Sampler(constraint)

        # Satisfies constraints
        self.assertTrue(
            sampler._is_valid_point(np.array([0.5, 0.5]))
        )
        # Does not satisfy constraints
        self.assertFalse(
            sampler._is_valid_point(np.array([0.1, 0.1]))
        )
        # Outside cube
        self.assertFalse(
            sampler._is_valid_point(np.array([-0.1, 0.1]))
        )
        self.assertFalse(
            sampler._is_valid_point(np.array([0.1, -0.1]))
        )
        self.assertFalse(
            sampler._is_valid_point(np.array([1.1, 0.1]))
        )
        self.assertFalse(
            sampler._is_valid_point(np.array([0.1, 1.1]))
        )
