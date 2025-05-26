import numpy as np
from scipy.fft import ifft as scipy_ifft # Importar ifft do SciPy

print(f"Versão do NumPy: {np.__version__}")
# (Se você tiver SciPy instalado, pode adicionar: import scipy; print(f"Versão do SciPy: {scipy.__version__}"))


# Use exatamente o mesmo scaled_vector do seu log
scaled_vector_from_log = np.array([
    -8.96000000e+01, -7.68000000e+01, -6.40000000e+01, -5.12000000e+01,
    -3.84000000e+01, -2.56000000e+01, -1.28000000e+01,  1.42108547e-14,
     1.28000000e+01,  2.56000000e+01,  3.84000000e+01,  5.12000000e+01,
     6.40000000e+01,  7.68000000e+01,  8.96000000e+01,  1.02400000e+02
])

print("\n--- Teste NumPy IFFT ---")
result_numpy_ifft = np.fft.ifft(scaled_vector_from_log)
print(f"Parte Real (NumPy IFFT):\n{np.real(result_numpy_ifft)}")

print("\n--- Teste SciPy IFFT ---")
result_scipy_ifft = scipy_ifft(scaled_vector_from_log) # Usar a ifft do SciPy
print(f"Output da scipy.fft.ifft (result_scipy_ifft):\n{result_scipy_ifft}")
print(f"Parte Real do Output (np.real(result_scipy_ifft)):\n{np.real(result_scipy_ifft)}")
print("--- Fim do Teste SciPy IFFT ---")