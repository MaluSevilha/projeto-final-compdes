# CIFAR Pipeline Project

Organização por experimentos:

- `common/` contém código compartilhado;
- `experiments/baseline/` contém a versão baseline em Python puro;
- `experiments/vectorized/` contém a versão NumPy;
- `experiments/multiprocessing/` contém a versão paralela;
- `experiments/gpu/` contém a versão com PyTorch/CUDA;
- `benchmark/` compara todas as versões.

## Execução

Na raiz do projeto:

```bash
python3 -m experiments.baseline.run
python3 -m experiments.baseline.profiling
python3 -m experiments.vectorized.run
python3 -m experiments.multiprocessing.run
python3 -m experiments.gpu.run
python3 -m benchmark.compare_versions
```

## Dependências

- numpy
- matplotlib

Opcional para GPU:
- torch
