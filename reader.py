import numpy as np
import pandas as pd
import os
import gc

def UA3P_reader(source = "XX.csv", dim = "auto", direction = "X"): # dim = 2, 3, "auto"
    assert type(source) == type(""), "worng type input"
    assert dim == 2 or dim == 3 or dim == "auto", "dim must be  2, 3, \"auto\""
    assert source[-4:].upper() == ".CSV" or source[-4:].upper() == ".TXT", ""
    if not os.path.isfile(source):
        raise TypeError("can't find the file:\n\t" + source)
    if source[-4:].upper() == ".CSV":
        df = pd.read_csv(source, header = 1)
        UA3P_array = df.to_numpy()
    else:
        with open(source, "w") as f:
            lines = f.readlines()
            # lines = [for line in lines]
            UA3P_array = np.empty((10, 4))
    UA3P_x = UA3P_array[:, 0]
    UA3P_y = UA3P_array[:, 1]
    UA3P_z = UA3P_array[:, 2]
    UA3P_zd = UA3P_array[:, 3]

    if dim != 3:
        X = UA3P_array[:, :2]
        eigenvalues, eigenvectors = np.linalg.eig(np.cov(X.T))
        eigenvalues = np.array(eigenvalues)
        eigenvectors = np.array(eigenvectors)
        if eigenvalues[1] == 0:
            dim = 2
            direction = eigenvectors[0]
        else:
            temp = eigenvalues[0] / eigenvalues[1]
            if temp > 20:
                dim = 2
                direction = eigenvectors[0]
            elif temp < 0.05:
                dim = 2
                direction = eigenvectors[1]
            else:
                dim = 3
                direction = None

    return UA3P_x, UA3P_y, UA3P_z, UA3P_zd, dim, direction
    # del UA3P_array
    # gc.collect()
