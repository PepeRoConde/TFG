import pickle
import sys
from src.plots.metrics import plot_coding_rate, plot_sparsity

fns = {"plot_coding_rate": plot_coding_rate, "plot_sparsity": plot_sparsity}
for path in sys.argv[1:]:
    d = pickle.load(open(path, "rb"))
    fns[d["fn"]](**d["kwargs"])
