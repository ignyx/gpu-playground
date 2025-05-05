import numpy as np
import complexmodule
import unittest

class TestComplexOperation(unittest.TestCase):
    def test_ensure_square(self):
        array = np.array([[1+2j, 3+4j]], dtype=np.complex128)
        with self.assertRaises(TypeError):
            result = complexmodule.complex_operation(array)

# Create a complex NumPy array
array = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)

# Call the C function
result = complexmodule.complex_operation(array)

print(result)
print(array.dot(array))

if __name__ == '__main__':
    unittest.main()
