# Projeto Final - Planejamento

## Tema

**Pipeline otimizado para processamento e classificação
de imagens utilizando a base CIFAR-10**

## Objetivo

Desenvolver um pipeline de processamento de imagens utilizando a base
**CIFAR-10**, implementando inicialmente uma versão sequencial em CPU e,
posteriormente, versões otimizadas com foco em desempenho.

O projeto atenderá ao requisito da disciplina de aplicar operações de
processamento de imagens (como conversão para escala de cinza e detecção
de bordas) e adicionará uma etapa de classificação utilizando uma CNN
simples. Assim, será possível medir o impacto das otimizações tanto no
processamento das imagens quanto no treinamento/inferência do modelo.

----

# Pipeline proposto

    Carregar imagens CIFAR-10
            ↓
    Converter para escala de cinza
            ↓
    Aplicar detector de bordas (Sobel ou Canny)
            ↓
    Gerar conjunto processado
            ↓
    Treinar uma CNN simples
            ↓
    Avaliar acurácia

A etapa de processamento de imagens atende diretamente ao enunciado da
professora. A CNN utiliza as imagens processadas como entrada, tornando
o projeto mais completo e permitindo avaliar desempenho em um pipeline
real.

------------------------------------------------------------------------

# Tecnologias

-   Python
-   NumPy
-   OpenCV
-   PyTorch
-   Matplotlib
-   cProfile
-   torch.profiler (opcional)

------------------------------------------------------------------------

# Etapas do projeto

## 1. Preparação

-   Baixar a base CIFAR-10.
-   Entender o formato dos dados.
-   Selecionar uma parcela inicial (ex.: 5.000 imagens).

------------------------------------------------------------------------

## 2. Versão sequencial

Implementar todo o pipeline utilizando apenas CPU:

-   carregar imagens;
-   converter para escala de cinza;
-   detectar bordas;
-   salvar ou manter o resultado em memória;
-   treinar uma CNN pequena;
-   avaliar o modelo.

Nesta etapa não serão utilizadas otimizações como paralelismo, GPU ou
múltiplos workers.

----

## 3. Medições

Medir:

-   tempo total;
-   tempo de cada etapa do pipeline;
-   uso de CPU e memória;
-   throughput (imagens por segundo).

Ferramentas sugeridas:

-   time.perf_counter
-   cProfile
-   torch.profiler

------------------------------------------------------------------------

## 4. Identificação dos gargalos

Responder:

-   Qual etapa consome mais tempo?
-   O gargalo está no processamento das imagens ou no treinamento da
    CNN?
-   Há tempo significativo gasto em leitura de dados?

Registrar as evidências com tabelas e gráficos.

------------------------------------------------------------------------

## 5. Otimizações

Aplicar uma otimização por vez e medir seu impacto.

### Processamento de imagens

-   Vetorização com NumPy.
-   Paralelização do processamento utilizando multiprocessing.

### Entrada de dados

-   DataLoader com múltiplos workers.
-   pin_memory.
-   persistent_workers.

### Treinamento

-   Execução em GPU (CUDA).
-   Mixed Precision (AMP).
-   Ajuste do batch size.

Cada otimização deverá ser comparada com a versão anterior.

------------------------------------------------------------------------

## 6. Comparação

Construir tabelas contendo:

  Versão     Tempo total   Speedup   Acurácia
  -------- ------------- --------- ----------

Além disso, gerar gráficos de:

-   tempo total;
-   tempo por etapa;
-   speedup;
-   imagens processadas por segundo.

------------------------------------------------------------------------

## 7. Validação

Verificar que:

-   todas as imagens foram processadas corretamente;
-   a CNN produz resultados equivalentes entre as versões;
-   as otimizações não alteraram o comportamento esperado do pipeline.

------------------------------------------------------------------------

# Organização sugerida do código

    projeto/

    dataset.py
    preprocess.py
    model.py
    train.py
    evaluate.py
    benchmark.py
    profiling.py

    optimizations/
        multiprocessing_pipeline.py
        gpu_training.py
        amp_training.py

    graficos/
    relatorio/
    README.md

------------------------------------------------------------------------

# Cronograma

### Semana 1

-   Download da base.
-   Implementação da versão sequencial.

### Semana 2

-   Profiling.
-   Identificação dos gargalos.

### Semana 3

-   Implementação das otimizações.

### Semana 4

-   Tabelas, gráficos, relatório e apresentação.

------------------------------------------------------------------------

# Justificativa da ideia

A proposta atende integralmente ao requisito da disciplina porque
realiza um pipeline de processamento de imagens contendo conversão para
escala de cinza e detecção de bordas em toda a base selecionada.

A inclusão da CNN acrescenta uma etapa de classificação sobre as imagens
processadas, permitindo explorar otimizações de alto desempenho em
diferentes partes do pipeline (pré-processamento, carregamento de dados
e treinamento), tornando a análise de desempenho mais rica sem fugir da
proposta da professora.
