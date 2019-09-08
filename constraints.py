import re

class Constraint:
    """Constraints loaded from a file."""

    def __init__(self, fname):
        """
        Construct a Constraint object from a constraints file

        :param fname: Name of the file to read the Constraint from (string)
        """
        with open(fname, "r") as f:
            lines = f.readlines()
        # Parse the dimension from the first line
        self.n_dim = int(lines[0])
        # Parse the example from the second line
        self.example = [float(x) for x in lines[1].split(" ")[0:self.n_dim]]

        # Run through the rest of the lines and compile the constraints
        self.exprs = []
        self._value_exprs = []
        re_extract_value_str = re.compile(r'(.+)[<>]=?.+')
        for line in lines[2:]:
            # support comments in the first line
            if line.startswith("#"):
                continue

            value_expr_str = re_extract_value_str.search(line).groups()
            if not value_expr_str:
                raise ValueError(f"Error parsing constraint: {line}")
            self._value_exprs.append(
                compile(value_expr_str[0], "<string>", "eval")
            )
            self.exprs.append(compile(line, "<string>", "eval"))

    def get_example(self):
        """Get the example feasible vector"""
        return self.example

    def get_ndim(self):
        """Get the dimension of the space on which the constraints are defined"""
        return self.n_dim

    def get_constraint_funcs(self):
        return self._value_exprs

    def evaluate_constraints(self, x):
        """
        Apply the constraints to a vector, returning the values of the constraint
        expressions.

        :param x: list or array on which to evaluate the constraints
        """
        values = []
        for expr in self._value_exprs:
            values.append(eval(expr))
        return values

    def apply(self, x):
        """
        Apply the constraints to a vector, returning True only if all are satisfied

        :param x: list or array on which to evaluate the constraints
        """
        for expr in self.exprs:
            if not eval(expr):
                return False
        return True   
