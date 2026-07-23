# Discovery — Base CIFAR-10 para o Projeto Final

## 1. Contexto da base

A base escolhida para o projeto é a **CIFAR-10**, disponível na página oficial do conjunto CIFAR.

A CIFAR-10 é um conjunto de dados para visão computacional formado por **60.000 imagens coloridas de 32 x 32 pixels**, distribuídas em **10 classes**. A base possui **50.000 imagens de treino** e **10.000 imagens de teste**. As classes são mutuamente exclusivas e incluem: airplane, automobile, bird, cat, deer, dog, frog, horse, ship e truck.

## 2. Por que essa base faz sentido para o projeto

A CIFAR-10 é uma boa escolha para este trabalho porque:

- as imagens são pequenas, o que facilita o processamento local;
- a base já está organizada em classes, permitindo usar a informação de rótulo;
- o volume de dados é suficiente para medir tempo de execução e gargalos;
- a estrutura da base permite construir um pipeline de processamento de imagens com etapas bem definidas;
- depois do pré-processamento, é possível usar uma CNN simples para classificação ou validação do pipeline.

## 3. Formato dos dados

A página da CIFAR informa que o conjunto CIFAR-10 está disponível em diferentes formatos:

- versão **Python**;
- versão **Matlab**;
- versão **binary**.

Na versão Python, a base é dividida em:

- `data_batch_1`
- `data_batch_2`
- `data_batch_3`
- `data_batch_4`
- `data_batch_5`
- `test_batch`

Cada batch contém um dicionário com:

- `data`: matriz `10000 x 3072` com valores `uint8`;
- `labels`: lista com os rótulos das imagens.

A imagem é armazenada em ordem linear, com:

- **1024 valores do canal vermelho**;
- **1024 valores do canal verde**;
- **1024 valores do canal azul**.

Isso significa que cada imagem precisa ser reconstruída a partir do vetor de 3072 valores para voltar ao formato `32 x 32 x 3`.

## 4. Impacto disso no pipeline

Essa estrutura é útil para o projeto porque permite montar um pipeline com etapas bem claras:

1. carregar os batches da CIFAR-10;
2. reconstruir as imagens coloridas;
3. converter para escala de cinza;
4. aplicar detecção de bordas;
5. salvar ou reutilizar as imagens processadas;
6. treinar uma CNN simples com a saída do pipeline;
7. medir o desempenho de cada etapa.

## 5. Proposta de problema para o projeto

O problema escolhido pode ser descrito assim:

**“Construir e otimizar um pipeline de processamento de imagens da CIFAR-10, aplicando conversão para escala de cinza e detecção de bordas em todas as imagens selecionadas, e usar o resultado em uma CNN simples para classificação ou validação do processamento.”**

Essa formulação encaixa bem na proposta da disciplina porque:

- atende ao requisito de processar um conjunto de imagens;
- inclui operações explícitas de pré-processamento;
- permite medir tempo, gargalos e speedup;
- abre espaço para otimizações em leitura de dados, processamento e treinamento.

## 6. Primeira hipótese de gargalos

Antes de implementar, os gargalos mais prováveis são:

- leitura e reconstrução das imagens a partir dos batches;
- conversão das imagens para escala de cinza;
- aplicação do detector de bordas;
- treinamento da CNN, se o conjunto escolhido for grande o suficiente;
- transferência de dados entre etapas do pipeline.

## 7. Estratégia inicial de implementação

A primeira versão deve ser simples e sequencial:

- executar tudo em CPU;
- usar uma parcela reduzida da base para benchmark inicial;
- manter o código fácil de medir e reproduzir;
- registrar tempo total e tempo por etapa;
- validar se o resultado do pré-processamento está correto.

## 8. Estratégia de otimização

Depois da versão inicial, as otimizações podem seguir esta ordem:

- melhorar a leitura e organização dos dados;
- vetorização do pré-processamento;
- paralelização do pipeline de imagens;
- uso de DataLoader com múltiplos workers;
- uso de GPU para a CNN;
- ajuste de batch size;
- eventual uso de mixed precision.

## 9. Entregáveis esperados

Para esta etapa de discovery, o ideal é terminar com:

- descrição da base CIFAR-10;
- definição do problema escolhido;
- mapeamento das etapas do pipeline;
- identificação das possíveis fontes de custo computacional;
- plano de execução da versão sequencial e das otimizações.

## 10. Conclusão

A CIFAR-10 é adequada para o projeto porque combina um conjunto de imagens pequeno o bastante para processamento local com volume suficiente para análise de desempenho.

A base também combina bem com a proposta da professora, já que permite aplicar processamento de imagens de forma explícita e, ao mesmo tempo, construir uma segunda etapa com CNN para explorar otimizações computacionais.
