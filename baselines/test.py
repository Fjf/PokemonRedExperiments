import numpy as np

dtype = np.int16
ftype = np.float16
if __name__ == "__main__":
    a = dtype(-2**11)
    while a != 2**11:
        a = dtype(a + 1)
        assert dtype(ftype(a)) == a, f"{a} ~= {dtype(ftype(a))}"
