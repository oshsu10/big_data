# big_data

Source code for the paper *"Development and optimization of deep learning
algorithms for real-time big data stream processing using computational
intelligence"* (submitted to Alexandria Engineering Journal).

N.K. Arkabaev, Osh State University, Kyrgyz Republic.

## Contents

| file | paper | description |
|------|-------|-------------|
| `asp.py`     | Sec. 3.4, Eq. (1) | `ASP` - adaptive stream processor: pruning, int8 quantization, early exits, resource monitor, run-time adaptation |
| `eem.py`     | Sec. 4.2 | `EEM` - cascade early-exit model with genuine per-sample compute saving |
| `bal.py`     | Sec. 4.3, Eq. (2) | reward function `rw()` and tabular Q-learning controller `QB` |
| `kstream.py` | Sec. 4.5 | asynchronous consumer with adaptive batching (Apache Kafka or a built-in mock stream) |
| `exp.py`     | Table 1  | repeated-run experiment driver; prints mean +/- std rows in LaTeX form |

## Install

```bash
pip install -r requirements.txt
```

## Quick start (smoke test on synthetic data)

```bash
python exp.py --task txt --reps 3 --quick        # end-to-end pipeline check
python kstream.py                                # adaptive batching demo
```

## Reproducing Table 1

On the experimental cluster, point the driver at the real datasets
(each `.npz` contains arrays `x` and `y`):

```bash
python exp.py --task iot --data iot.npz --reps 10 --latex
python exp.py --task txt --data txt.npz --reps 10 --latex
python exp.py --task fin --data fin.npz --reps 10 --latex
```

The `--latex` flag prints rows ready for insertion into Table 1 of the
manuscript (`mean $\pm$ std` over the given number of runs).

Note: full pruning and int8 quantization require `tensorflow < 2.16`
(Keras 2, as in the paper: TF 2.7); on Keras 3 these steps are skipped
with a message while early exits and adaptation remain active.

Note: numbers produced on synthetic data are for pipeline verification
only; the values reported in the paper are measured on the real
datasets and hardware described in Sec. 3.
