"""MC-TIRE (Cao, Seeuws, De Vos and Bertrand) dissimilarity curves for a bivariate series.

Runs in a separate environment with TensorFlow 2.15 and TensorFlow Probability 0.23; the method's code
is the authors' repository (github.com/caozhenxiang/MC-TIRE, commit f538a86), placed at
00_SRC/_ext/MC-TIRE and used unchanged. The pipeline follows the repository's MC_main.py for its
multi-channel ("MC") datasets: window size 40, nfft 30, "timeseries" normalization, both time and
frequency domains, rank 1, n_filter 2, loss weights 1e-2 / 1e-2 / 0.1, at most 200 epochs with early
stopping (patience 10), batch size 64. The only additions are a fixed seed per series and the return of
the two smoothed dissimilarity curves ("A&S", cross-channel; "B", channel-specific) instead of the
repository's peak-merging heuristic, so that a single-break scan can be calibrated like the other
statistics. Curve index i corresponds to time i + 40 (the repository reports peaks + window size).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ext", "MC-TIRE")
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from functions import preprocessing, utils, evaluation  # noqa: E402

net = utils.load_architecture("MC-TIRE")
WS, NFFT = 40, 30
PARAMS = dict(w_as=1e-2, w_b=1e-2, w_uncor=0.1, n_filter=2, rank=1)


def windows(Z):
    td, fd = [], []
    for c in range(Z.shape[1]):
        w = preprocessing.ts_to_windows(Z[:, c], 0, WS, 1, normalization="timeseries")
        fd.append(utils.calc_fft(w, NFFT, norm_mode="timeseries"))
        td.append(utils.norm_windows(w))
    return np.transpose(np.array(td), [1, 2, 0]), np.transpose(np.array(fd), [1, 2, 0])


def curves(Z, seed):
    """Smoothed A&S and B dissimilarity curves of the series Z (n x channels)."""
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    wtd, wfd = windows(np.asarray(Z, float))
    p = PARAMS
    as_td, b_td, _, _ = net.train_model(wtd, p["w_as"], p["w_b"], p["w_uncor"], p["n_filter"], 0, False, p["rank"])
    as_fd, b_fd, _, _ = net.train_model(wfd, p["w_as"], p["w_b"], p["w_uncor"], p["n_filter"], 0, False, p["rank"])
    d = evaluation.smoothened_dissimilarity_measures(as_td, as_fd, b_td, b_fd, WS)
    return np.asarray(d["A&S"], float), np.asarray(d["B"], float)
