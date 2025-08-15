import sympy
import jax
import jax.numpy as jnp
import inspect

# --- Define symbolic residual ---

# Define symbolic variables
x, y = sympy.symbols("x y")

# Define the residual as a symbolic Matrix
residual_expr = sympy.Matrix(
    [
        sympy.sin(x + y) + sympy.sin(x),
        (sympy.cos(x) ** 2 + sympy.sin(x) ** 2) * sympy.cos(x + y)
        + sympy.cos(y),
    ]
)

print(f"\nResidual expression {residual_expr}")

residual_expr = sympy.simplify(residual_expr)

print(f"\nSimplified residual expression {residual_expr}")


# --- Convert symbolic parts to Python functions ---

# Create a numeric function from the symbolic expression
expr_func = sympy.lambdify((x, y), residual_expr, cse=True, modules="jax")


# --- Create and JIT-compile the final wrapper function ---


@jax.jit
def residual_fn(input_values):
    """
    JIT-compiled residual function.
    Takes a JAX array [x, y] as input.
    """
    x_in, y_in = input_values[0], input_values[1]

    # Execute the expression function
    result_matrix = expr_func(x_in, y_in)

    # Return a flattened JAX array
    return result_matrix.flatten()


# Create the Jacobian function using JAX
jacobian_fn = jax.jacobian(residual_fn)

# JIT-compile the Jacobian function for performance
jit_jacobian_fn = jax.jit(jacobian_fn)

# Create a sample input point
input_values = jnp.array([1.0, 2.0])

# Calculate the residual
residual_value = residual_fn(input_values)
print(f"\nResidual at {input_values}: {residual_value}")

# Calculate the Jacobian
jacobian_value = jit_jacobian_fn(input_values)
print(f"\nJacobian at {input_values}:\n{jacobian_value}")
