# Pipeline de pré-processamento da CIFAR-10: análise e otimização

Projeto final da eletiva de Computação de Alto Desempenho (Insper).

**Integrantes:** Giulia Roggero, Maria Luiza Sevilha Seraphico e Thais Sabaini

## O problema

Construir e otimizar um pipeline de processamento de imagens da CIFAR-10, aplicando conversão para escala de cinza e detecção de bordas (operador de Sobel) sobre todas as imagens selecionadas.

O pipeline é o mesmo em todas as versões:

1. carregar os batches da CIFAR-10 (pickle);
2. reconstruir cada vetor de 3.072 posições em uma imagem `32x32x3`;
3. converter RGB para escala de cinza (luminância);
4. aplicar Sobel horizontal e vertical, calcular a magnitude do gradiente e normalizar por imagem.

**Base de dados:** CIFAR-10 (60.000 imagens de 32x32x3 em 10 classes; 50.000 de treino e 10.000 de teste). Download: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

## Resultado

A versão sequencial em Python puro processa 5.000 imagens em **28,2 s**. A versão final processa **50.000 imagens em 0,765 s**, produzindo saída bit a bit idêntica à da baseline.

| Versão | 50.000 imagens | Speedup vs. baseline |
|---|---|---|
| baseline (Python puro) | 228,22 s | 1x |
| multiprocessing | 105,13 s | 2,2x |
| vetorizada (NumPy) | 1,927 s | 118x |
| numba | 0,715 s | 319x |
| gpu (PyTorch/CUDA) | 0,554 s | 412x |
| **numba paralelo (`prange`)** | **0,765 s** | 1,37x sobre o numba serial* |

\* O Numba paralelo foi medido em outra sessão de benchmark, junto das versões rápidas. Os tempos absolutos entre sessões não são comparáveis (veja "Cuidados de medição"), por isso o ganho é reportado dentro da própria sessão: 1,37x sobre o Numba serial e 3,14x sobre a vetorizada, chegando a 7,5% do tempo da GPU.

A análise completa está em **`docs/relatorio-final.ipynb`**, que é o relatório do projeto.

## Estrutura

```
common/                     código compartilhado (config e leitura da base)
experiments/
  baseline/                 versão sequencial em Python puro (referência)
  vectorized/               versão NumPy
  numba/                    versão Numba (@njit)
  numba_parallel/           versão Numba paralela (prange)
  multiprocessing/          versão com multiprocessing
  gpu/                      versão PyTorch/CUDA
benchmark/
  verify_pipeline.py        corretude: compara todas as versões com a baseline
  compare_versions.py       comparação em um volume fixo
  generate_scaling_csv.py   escalonamento das 5 primeiras versões
  scaling_parallel.py       escalonamento das versões rápidas + varredura de threads
outputs/
  benchmarks/               CSVs de resultado
  profiling/                perfis .prof do cProfile
  figures/                  figuras
  verification/             preview visual (RGB / gray / edges)
docs/                    
  apresentacao.pdf          apresentação final em formato PDF
  apresentacao.pptx         apresentação em formato PPT
  discovery.md              anotações sobre discovery da base de dads
  relatorio-final.ipynb               relatório final do projeto
```

Cada pasta em `experiments/` segue o mesmo formato: `preprocessing.py` (as operações), `pipeline.py` (orquestração e medição de tempo), `run.py` (execução pela linha de comando) e `profiling.py` (perfil com cProfile).

## Como reproduzir

### 1. Ambiente

Python 3.10 ou superior.

```bash
pip install -r requirements.txt
```

A versão GPU precisa do PyTorch, que **não** está em `requirements.txt` porque a instalação depende da versão de CUDA da máquina. Veja <https://pytorch.org/get-started/locally/>. Sem PyTorch, todos os scripts continuam funcionando: a versão GPU é detectada como indisponível e removida do benchmark com um aviso.

### 2. Base de dados

```bash
mkdir -p data
cd data
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
cd ..
```

Isso cria `data/cifar-10-batches-py/`, que é onde `common/config.py` espera encontrar a base. A pasta `data/` está no `.gitignore` e não vem no repositório.

### 3. Corretude, antes de qualquer medição

```bash
python3 -m benchmark.verify_pipeline
```

Roda todas as versões disponíveis sobre um subconjunto pequeno e compara cada uma com a baseline, que é a referência de corretude. As versões Numba e multiprocessing são comparadas com tolerância zero, porque preservam a ordem das operações; vetorizada e GPU usam tolerância de `1e-4` e `1e-3`, porque reassociam as somas do Sobel e diferem na última casa do `float32`. O script também verifica que o Numba paralelo é **bit a bit idêntico** ao serial, ou seja, que paralelizar não alterou o resultado. Sai com código 1 se algo falhar.

### 4. Execução individual de cada versão

```bash
python3 -m experiments.baseline.run --limit 5000
python3 -m experiments.vectorized.run --limit 5000
python3 -m experiments.numba.run --limit 5000 --warmup
python3 -m experiments.numba_parallel.run --limit 5000 --warmup
python3 -m experiments.multiprocessing.run --limit 5000
python3 -m experiments.gpu.run --limit 5000
```

Cada `run.py` imprime o tempo por etapa, o total e o throughput. Nas versões Numba, `--warmup` compila antes de medir, para que o tempo reflita a execução e não o compilador. `--limit` aceita qualquer valor até 50.000 (tamanho do split de treino).

### 5. Profiling

```bash
python3 -m experiments.baseline.profiling --limit 5000
python3 -m experiments.numba.profiling --limit 5000
python3 -m experiments.numba_parallel.profiling --limit 5000 --warmup
```

Os perfis vão para `outputs/profiling/*.prof`. Para inspecionar visualmente:

```bash
pip install snakeviz
snakeviz outputs/profiling/baseline_profile.prof
```

O perfil do Numba **sem** `--warmup` mostra a compilação JIT dominando o tempo; **com** `--warmup`, mostra o custo real de execução. Os dois são usados no relatório, com propósitos diferentes.

### 6. Benchmarks

Os três podem ser executados da raiz do projeto ou de dentro de `benchmark/`.

```bash
# comparação em volume fixo (gera outputs/benchmarks/compare_versions.csv)
python3 -m benchmark.compare_versions --limit 5000 --repeat 3

# escalonamento das 5 primeiras versões, Numba já compilado
# (gera outputs/benchmarks/scaling_hot.csv)
python3 -m benchmark.generate_scaling_csv

# custo fixo da compilação JIT: Numba recompilando em cada ponto
# (gera outputs/benchmarks/scaling_numba_cold.csv)
python3 -m benchmark.generate_scaling_csv --numba-mode cold --counts 100 500 1000 2000 5000

# escalonamento das versões rápidas + varredura de threads
# (gera scaling_parallel.csv, scaling_parallel_detail.csv e threads_parallel.csv)
python3 -m benchmark.scaling_parallel
```

Tempos aproximados de execução completa, com a base inteira: `generate_scaling_csv` no modo `hot` leva cerca de 30 minutos, quase todo consumido pela baseline em volumes altos; no modo `cold`, cerca de 15 minutos, quase todo em compilação. O `scaling_parallel`, que não inclui a baseline, leva poucos minutos.

Opções úteis:

- `--counts 1000 10000 50000` escolhe os volumes a medir (valores acima do teto do split são ajustados);
- `--repeat N` define as repetições por ponto;
- `--threads N` fixa as threads do Numba paralelo em `scaling_parallel`;
- `--include-slow` acrescenta baseline e multiprocessing ao `scaling_parallel`;
- `--cold-numba` acrescenta ao `compare_versions` uma linha com o Numba medido a frio.

### 7. Relatório

```bash
pip install jupyter
jupyter notebook final.ipynb
```

O notebook localiza a raiz do projeto sozinho e lê os CSVs e perfis de `outputs/`. As células de análise rodam **sem** a base baixada; apenas duas células de execução ao vivo (inspeção da base e execução da baseline) dependem de `data/`, e elas se pulam sozinhas com um aviso.

## Cuidados de medição

Três pontos que afetam a leitura de qualquer número deste projeto:

**Compilação JIT não é tempo de processamento.** O Numba paga um custo fixo de compilação na primeira chamada de cada assinatura de função, independente do volume de dados. Medido a frio, esse custo é de cerca de 140 s, enquanto processar 5.000 imagens já compilado custa 0,078 s. O regime principal do relatório é o **quente** (com warm-up, ou com o binário lido do cache em disco), e a compilação é reportada separadamente como custo de setup, do mesmo jeito que não se contabiliza a inicialização do contexto CUDA no custo por imagem da GPU.

**Comparações valem dentro da mesma sessão.** As versões presentes tanto em `scaling_hot.csv` quanto em `scaling_parallel.csv` ficaram de 14% a 41% mais lentas na segunda medição, na mesma máquina e com o mesmo código, por variação térmica, de frequência e de cache. Não compare colunas entre arquivos diferentes.

**O split de treino tem 50.000 imagens.** Pedir `--limit 60000` não gera erro, apenas devolve as 50.000 disponíveis. Os scripts de benchmark ajustam os volumes ao teto do split para não registrar dois pontos medindo o mesmo volume.

## Arquivos de resultado

| Arquivo | Conteúdo |
|---|---|
| `outputs/benchmarks/compare_versions.csv` | volume fixo de 5.000 imagens, tempo por etapa e desvio padrão |
| `outputs/benchmarks/scaling_hot.csv` | escalonamento de 100 a 50.000 imagens, 5 versões, Numba compilado |
| `outputs/benchmarks/scaling_parallel.csv` | escalonamento das 4 versões rápidas, mesma sessão |
| `outputs/benchmarks/scaling_parallel_detail.csv` | o mesmo, com desvio padrão, mínimo e nº de threads |
| `outputs/benchmarks/threads_parallel.csv` | varredura de threads em 50.000 imagens, com eficiência paralela |
| `outputs/benchmarks/scaling_numba_cold.csv` | Numba recompilando em cada ponto: custo fixo do JIT |
| `outputs/profiling/baseline_profile.prof` | perfil da versão sequencial |
| `outputs/profiling/numba_profile.prof` | perfil do Numba a frio (compilação dominando) |
| `outputs/profiling/numba_parallel_profile.prof` | perfil da versão final, já compilada |

## Dependências

- `numpy`
- `matplotlib`
- `pandas`
- `numba`

Opcionais: `torch` (versão GPU), `jupyter` (relatório), `snakeviz` (visualização dos perfis).
