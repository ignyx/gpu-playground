import numpy as np
import complexmodule
import unittest

class TestComplexOperation(unittest.TestCase):
    def test_ensure_square(self):
        array = np.array([[1+2j, 3+4j]], dtype=np.complex128)
        with self.assertRaises(TypeError):
            result = complexmodule.complex_operation(array)

    def test_result(self):
        DIM = 1000
        matrix = np.random.rand(DIM, DIM) + 1j * np.random.rand(DIM, DIM)
        A = matrix.astype(np.complex128)

        result = complexmodule.complex_operation(A)
        expected = A.dot(A)
        # expected = A.dot(np.diag(np.diag(A))).dot(np.linalg.inv(A))

        np.testing.assert_allclose(result, expected)

# Create a complex NumPy array
array = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)

# Call the C function
result = complexmodule.complex_operation(array)

print(result)
print(array.dot(array))

if __name__ == '__main__':
    unittest.main()
