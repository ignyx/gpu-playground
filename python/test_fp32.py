import numpy as np
import complexmodule
import unittest
from time import time

dtype=np.float32

class TestComplexOperation(unittest.TestCase):
    def test_ensure_square(self):
        array = np.array([[1, 3]], dtype=dtype)
        with self.assertRaises(TypeError):
            result = complexmodule.complex_operation(array)

    def test_result(self):
        DIM = 1000
        matrix = np.random.rand(DIM, DIM)
        A = matrix.astype(dtype) # maybe better to use random.Generator

        start = time()
        result = complexmodule.complex_operation(A)
        end_gpu = time()

        expected = A.dot(np.diag(np.diag(A))).dot(A)
        end_cpu = time()

        print(f"GPU took {format(end_gpu-start, '.6f')} sec")
        print(f"CPU took {format(end_cpu-start, '.6f')} sec")
        np.testing.assert_allclose(result, expected, rtol=1e-5)

# Create a complex NumPy array
array = np.array([[1, 3], [5, 7]], dtype=dtype)

# Call the C function
result = complexmodule.complex_operation(array)

print(result)
print(array.dot(array))

if __name__ == '__main__':
    unittest.main()
