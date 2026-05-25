"""
src/plots/utils.py
~~~~~~~~~~~~~~~~~~
Shared plotting utilities used by plot_logs.py and sparse_y_crate.py.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colour palette – evenly spaced samples from the 'summer' colormap.
# Call get_colors(n) to get exactly n colours.
# ---------------------------------------------------------------------------


def get_colors(n):
    """Return *n* colours sampled uniformly from the 'summer' colormap."""
    cmap = plt.get_cmap("tab20")
    return [
        cmap(i / max(n - 1 + 0.1, 1.1)) for i in range(n)
    ]  # el .1 es pal negro del cobre


# ---------------------------------------------------------------------------
# Field-name shortcuts used in run labels
# ---------------------------------------------------------------------------
FIELD_SHORTCUTS = {
    "batch_size": "b",
    "learning_rate": "lr",
    "lr": "lr",
    "weight_decay": "wd",
    "label_smoothing": "ls",
    "tamano_patch": "tp",
    "tamano_token": "tt",
    "num_sigmas": "ns",
    "sigma": "s",
    "contador_aumento": "ca",
    "overlap_rate": "or",
    "label_mode": "lm",
    "aumento_datos": "aug",
    "use_amp": "amp",
    "paciencia": "pac",
    "optimizer": "opt",
    "epochs": "e",
    "arch": "a",
    "order": "orden",
    "shared_dict": "dicionario compartido",
    "shared_u": "$U$ compartido",
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_varying_fields(configs):
    """
    Return the config keys whose values differ across the given iterable of
    config dicts (missing keys are treated as NaN).
    """
    configs = list(configs)
    if not configs:
        return []

    all_keys: set = set()
    for cfg in configs:
        if cfg:
            all_keys.update(cfg.keys())

    all_keys.discard("runs_dir")

    varying = []
    for key in all_keys:
        values = set()
        for cfg in configs:
            if cfg:
                val = cfg.get(key, np.nan)
                if isinstance(val, (list, dict)):
                    val = str(val)
                values.add(val)
        if len(values) > 1:
            varying.append(key)

    return sorted(varying)


def config_to_label(config, varying_fields):
    """Build a concise run label from the fields that vary across runs."""
    if not config or not varying_fields:
        return "unknown"

    parts = []
    for field in varying_fields:
        val = config.get(field, np.nan)
        display = FIELD_SHORTCUTS.get(field, field)

        if isinstance(val, float) and np.isnan(val):
            parts.append(f"{display}:NaN")
        elif isinstance(val, bool):
            parts.append(f"{display}:{int(val)}")
        else:
            parts.append(f"{display}:{val}")

    return ", ".join(parts)


def get_marker_and_linewidth(config, varying_fields):
    """Derive marker style and line-width from the run config."""
    label_mode = config.get("label_mode", "vainilla")
    sigma = config.get("sigma", 1)

    marker = "o"
    if label_mode == "gaussian":
        linewidth = sigma * 0.5 if isinstance(sigma, (int, float)) else 1.5
    else:
        linewidth = 1.0

    return marker, linewidth
