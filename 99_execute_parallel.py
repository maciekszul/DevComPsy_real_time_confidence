import sys
import subprocess as sp
from joblib import Parallel, delayed

try:
    n_jobs = int(sys.argv[1])
except:
    raise IndexError("no jobs :(")

try:
    path = str(sys.argv[2])
except:
    raise IndexError("no file path")

try:
    range_of_files = int(sys.argv[3])
except:
    raise IndexError("no range of files")


def job_to_do(index, path, json_file):
    sp.call([
        "python",
        path,
        str(index)
    ])

Parallel(n_jobs=n_jobs)(delayed(job_to_do)(index, path) for index in range(range_of_files))