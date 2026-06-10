# Marquetry Samples

Self-contained scripts showing the main features of Marquetry.
Each one runs directly:

```shell
python samples/01_autograd_basics.py
```

| Sample | What it shows | Notes |
|---|---|---|
| `01_autograd_basics.py` | Define-by-run autodiff: forward math, `backward()`, `no_backprop_mode()` | |
| `02_spiral_classification.py` | Standard training loop: Dataset → DataLoader → MLP → loss → optimizer | synthetic data, no download |
| `03_fashion_mnist_mlp.py` | Training on a real dataset with transformers | downloads ~30MB once, cached in `~/.marquetry` |
| `04_lstm_curve_forecast.py` | Stateful LSTM with truncated BPTT (`unchain_backward`) | |
| `05_classic_ml.py` | Non-neural models: RandomForest and SVM via `fit` / `predict` | |
| `06_model_export.py` | Persisting models: `save_params` (npz) / marquetry archive (`.mq`) / ONNX | ONNX part needs `pip install onnx onnxruntime` |

Generated files (trained models, exports) are written to `samples/output/`.
