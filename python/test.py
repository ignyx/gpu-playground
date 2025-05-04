import numpy as np
import complexmodule

# Create a complex NumPy array
array = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)

# Call the C function
result = complexmodule.complex_operation(array)

print(result)
