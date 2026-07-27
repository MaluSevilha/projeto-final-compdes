# Projeto Final --- Computação de Alto Desempenho

## Pipeline otimizado para processamento de imagens utilizando a base CIFAR-10

**Status:** Relatório em desenvolvimento (Versão 0.1)

------------------------------------------------------------------------

# 1. Introdução

O processamento eficiente de grandes conjuntos de imagens é uma tarefa
frequente em aplicações de Visão Computacional e Inteligência
Artificial. Em pipelines de processamento, etapas aparentemente simples,
como leitura dos dados e pré-processamento das imagens, podem
representar uma parcela significativa do tempo total de execução.

O objetivo deste projeto é desenvolver um pipeline para processamento de
imagens utilizando a base CIFAR-10, implementando inicialmente uma
versão sequencial em CPU que servirá como referência para todas as
comparações de desempenho.

Após a implementação da versão inicial, serão utilizados métodos de
profiling para identificar os principais gargalos da aplicação. Com base
nesses resultados, serão propostas e avaliadas diferentes estratégias de
otimização, buscando reduzir o tempo de execução e aumentar o throughput
do pipeline sem alterar seus resultados.

------------------------------------------------------------------------

# 2. Objetivos

## Objetivo geral

Desenvolver e otimizar um pipeline de processamento de imagens
utilizando a base CIFAR-10.

## Objetivos específicos

-   estudar a estrutura da base CIFAR-10;
-   implementar uma versão sequencial do pipeline;
-   medir o tempo de execução de cada etapa;
-   identificar gargalos por meio de profiling;
-   implementar versões otimizadas;
-   comparar desempenho entre todas as versões.

------------------------------------------------------------------------

# 3. Base de dados

Foi utilizada a base **CIFAR-10**, disponibilizada pela Universidade de
Toronto.

Características da base:

-   60.000 imagens coloridas;
-   resolução de 32 × 32 pixels;
-   10 classes distintas;
-   50.000 imagens de treinamento;
-   10.000 imagens de teste.

As classes disponíveis são:

-   airplane
-   automobile
-   bird
-   cat
-   deer
-   dog
-   frog
-   horse
-   ship
-   truck

Na versão Python da base, as imagens encontram-se distribuídas em cinco
arquivos de treinamento (`data_batch_1` até `data_batch_5`) e um arquivo
de teste (`test_batch`).

Cada imagem é armazenada como um vetor de 3072 posições, correspondente
aos três canais RGB concatenados.

------------------------------------------------------------------------

# 4. Exploração da base

Foi desenvolvido um script (`explore_dataset.py`) para validar o
carregamento da base e compreender sua estrutura.

Durante essa etapa foram confirmados os seguintes aspectos:

-   leitura correta dos arquivos pickle;
-   reconstrução correta das imagens RGB;
-   associação correta entre imagens e rótulos;
-   distribuição aproximadamente uniforme entre as classes.

**Figura 1.** Exemplo de imagem reconstruída da base CIFAR-10.

> Inserir captura de tela obtida pelo `explore_dataset.py`.

------------------------------------------------------------------------

# 5. Implementação da versão sequencial

A primeira implementação foi organizada de forma modular.

## Estrutura dos módulos

-   `config.py`: configuração dos caminhos do projeto;
-   `dataset.py`: leitura da base e reconstrução das imagens;
-   `preprocessing.py`: conversão para escala de cinza e detecção de
    bordas;
-   `pipeline.py`: execução do pipeline;
-   `sequencial.py`: execução da versão baseline;
-   `profiling.py`: geração do perfil de execução.

## Pipeline implementado

O pipeline executa as seguintes etapas:

1.  carregamento da base;
2.  reconstrução das imagens RGB;
3.  conversão para escala de cinza;
4.  aplicação do operador de Sobel;
5.  armazenamento dos resultados em memória.

Cada etapa foi implementada em funções independentes para facilitar
futuras otimizações e análises de desempenho.

------------------------------------------------------------------------

# 6. Resultados da versão sequencial

Os testes iniciais foram realizados utilizando **5.000 imagens** do
conjunto de treinamento.

## Tempo por etapa

  Etapa                              Tempo (s)
  -------------------------------- -----------
  Leitura da base                        0,059
  Reconstrução RGB                    0,000008
  Conversão para escala de cinza         0,040
  Detecção de bordas (Sobel)             0,105
  **Tempo total**                    **0,205**

Throughput observado:

**24.420 imagens/s**

Observa-se que a reconstrução das imagens possui custo praticamente
desprezível quando comparada às demais etapas.

------------------------------------------------------------------------

# 7. Profiling

Para identificar os gargalos da aplicação foi utilizado o módulo
`cProfile`, complementado pela ferramenta **Snakeviz**.

**Figura 2.** Flame graph gerado pelo Snakeviz.

> Inserir captura de tela do Snakeviz.

Os resultados mostraram que os maiores custos estão concentrados em:

1.  `sobel_edges()`;
2.  `load_dataset()` / `pickle.load()`;
3.  `rgb_to_grayscale()`.

Em contrapartida, a reconstrução das imagens RGB apresentou custo
praticamente nulo.

Esses resultados indicam que as principais oportunidades de otimização
estão nas operações de pré-processamento e na leitura dos dados.

------------------------------------------------------------------------

# 8. Próximas etapas

A sequência planejada para o desenvolvimento é:

## Etapa 1 --- Baseline puro em Python

Reimplementar a conversão para escala de cinza e o operador de Sobel
utilizando apenas laços `for`, estabelecendo uma implementação
totalmente sequencial que servirá como referência.

## Etapa 2 --- Vetorização com NumPy

Substituir as operações manuais por implementações vetorizadas
utilizando NumPy e medir o ganho de desempenho.

## Etapa 3 --- Paralelização

Aplicar paralelização do processamento das imagens utilizando
`multiprocessing`.

## Etapa 4 --- GPU

Avaliar o uso de CUDA para acelerar partes do pipeline ou a etapa de
classificação com CNN.

------------------------------------------------------------------------

# 9. Conclusão parcial

Até o momento foi implementada e validada a primeira versão funcional do
pipeline, além da exploração da base de dados e da identificação dos
principais gargalos computacionais.

Os resultados do profiling mostraram que a detecção de bordas representa
o maior custo computacional da aplicação, seguida pelo carregamento da
base. Essas evidências orientarão as próximas etapas do projeto, nas
quais diferentes estratégias de otimização serão avaliadas
quantitativamente.
