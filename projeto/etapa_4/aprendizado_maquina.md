# 4. ETAPA 4 —  MODELAGEM E PREVISÃO DE CARGA ELÉTRICA


**Código em:**
|[modelo python](modelo2.py)|

## 1.1. Base de dados

A modelagem utilizou os dados da tabela `gold_fato_carga`, consultada no Amazon Athena. A base contempla o período de janeiro de 2016 a dezembro de 2024, com 13.152 registros diários distribuídos entre os quatro subsistemas.

Para a previsão, os dados foram agregados mensalmente pela média da carga (`val_cargaenergiamwmed`), resultando em 432 registros, sendo 108 para cada subsistema.

| Subsistema           | Carga média (MWmed) |
| -------------------- | ------------------: |
| Norte                |            6.025,39 |
| Nordeste             |           11.018,78 |
| Sudeste/Centro-Oeste |           38.537,02 |
| Sul                  |           11.691,99 |

## 1.2. Análise exploratória

A análise histórica mostrou diferenças no comportamento da carga entre os subsistemas, tanto em relação à escala quanto à variação ao longo dos meses.

Também foram identificados padrões sazonais distintos:

| Subsistema           | Maior média | Menor média |
| -------------------- | ----------- | ----------- |
| Nordeste             | Novembro    | Julho       |
| Norte                | Setembro    | Janeiro     |
| Sudeste/Centro-Oeste | Março       | Julho       |
| Sul                  | Fevereiro   | Maio        |

### Evolução histórica da carga

![Evolução da carga do Nordeste](imagens/evolucao_nordeste.png)
![Evolução da carga do Nordeste](imagens/evolucao_norte.png)
![Evolução da carga do Nordeste](imagens/evolucao_sudeste_centro_oeste.png)
![Evolução da carga do Nordeste](imagens/evolucao_sul.png)




### Sazonalidade mensal


![Sazonalidade da carga do Nordeste](imagens/sazonalidade_nordeste.png)
![Sazonalidade da carga do Nordeste](imagens/sazonalidade_norte.png)
![Sazonalidade da carga do Nordeste](imagens/sazonalidade_sudeste_centro_oeste.png)
![Sazonalidade da carga do Nordeste](imagens/sazonalidade_sul.png)


## 1.3. Modelo utilizado

Foi escolhida uma regressão linear com tendência temporal e sazonalidade mensal, aplicada individualmente a cada subsistema.

O modelo considera duas variáveis principais:

- **Tendência:** representa a evolução da carga ao longo do tempo.
- **Sazonalidade:** utiliza variáveis dummy para representar os meses do ano.

De forma simplificada, o modelo estima a carga a partir da tendência temporal e dos efeitos associados a cada mês.

As variáveis climáticas disponíveis na base não foram incluídas nesta primeira versão, ficando como possibilidade de melhoria para trabalhos futuros.

## 1.4. Treinamento e avaliação

A divisão dos dados respeitou a ordem cronológica:

| Etapa       | Período     |
| ----------- | ----------- |
| Treinamento | 2016 a 2022 |
| Teste       | 2023 a 2024 |
| Previsão    | 2025        |

Para avaliar o desempenho, foram utilizadas as métricas MAE, RMSE e MAPE. Também foi realizada uma comparação com um modelo de referência (*baseline*), que utiliza a carga observada no mesmo mês do ano anterior.

| Subsistema           | MAPE Regressão | MAPE Baseline |
| -------------------- | -------------: | ------------: |
| Nordeste             |          8,56% |         7,66% |
| Norte                |         16,09% |        10,83% |
| Sudeste/Centro-Oeste |          7,47% |         6,82% |
| Sul                  |          7,09% |         8,09% |

A regressão apresentou resultado melhor que o baseline apenas no subsistema Sul. Nos demais, o modelo de referência obteve menor erro percentual.

O Norte apresentou o maior erro relativo, enquanto o Sul teve o melhor resultado da regressão.

## 1.5. Análise dos resultados

A análise dos erros mostrou que o modelo subestimou a carga nos quatro subsistemas durante o período de teste. Essa diferença foi maior em 2024, indicando que a tendência estimada não acompanhou integralmente o crescimento observado.

Esse comportamento demonstra uma limitação da abordagem utilizada, que considera apenas a tendência linear e a sazonalidade mensal, sem incorporar outros fatores que podem influenciar a demanda de energia.

**Comparação entre valores reais e previstos**

\<!-- INSERIR AQUI OS GRÁFICOS DE RESULTADO. Arquivos com nome iniciado por resultado\_ -->

## 1.6. Previsão para 2025

Após a avaliação, o modelo foi ajustado novamente utilizando todo o histórico de 2016 a 2024, gerando previsões mensais para 2025.

Como exemplo, as estimativas para o subsistema Nordeste foram:

| Mês       | Previsão (MWmed) |
| --------- | ---------------: |
| Janeiro   |        12.812,85 |
| Fevereiro |        12.975,56 |
| Março     |        12.986,21 |
| Abril     |        12.626,26 |
| Maio      |        12.577,04 |
| Junho     |        12.210,47 |
| Julho     |        12.155,12 |
| Agosto    |        12.406,14 |
| Setembro  |        12.689,75 |
| Outubro   |        13.255,08 |
| Novembro  |        13.302,38 |
| Dezembro  |        13.231,01 |

As previsões devem ser interpretadas considerando as limitações identificadas na avaliação, principalmente a subestimação da carga nos anos de teste.

## 1.7. Considerações finais

A modelagem permitiu desenvolver uma primeira abordagem de previsão mensal para os quatro subsistemas. Os resultados mostraram que a regressão linear com tendência e sazonalidade teve desempenho variável, superando o baseline apenas no Sul.

Como melhorias futuras, podem ser avaliadas a inclusão de variáveis climáticas, a utilização de outros modelos de séries temporais e a análise de fatores que expliquem as mudanças recentes no comportamento da carga.

# 2. ARQUIVOS E CÓDIGOS DESENVOLVIDOS

O principal código desenvolvido nesta etapa foi o `modelo2.py`, responsável pela consulta, preparação dos dados, treinamento, avaliação e geração das previsões.

A execução do script produz os seguintes arquivos:

| Arquivo | Descrição |
|---|---|
| `modelo2.py` | Script principal da modelagem |
| `metricas_modelo2.csv` | Métricas de avaliação dos modelos |
| `previsoes_teste_modelo2.csv` | Valores reais e previstos no período de teste |
| `previsao_2025_modelo2.csv` | Previsões para 2025 |

Os gráficos de comparação entre valores reais e previstos são gerados separadamente pelo script `graficos_modelo2.py`.

Foram utilizadas as bibliotecas Python `awswrangler`, `pandas`, `numpy` e `matplotlib`, além do Amazon Athena para consulta aos dados.

# 3. RESULTADOS DA MODELAGEM

Ao final da etapa, foram obtidos quatro modelos de regressão, uma avaliação comparativa com o baseline sazonal e previsões mensais para 2025.

A comparação evidenciou que modelos mais simples podem apresentar resultados competitivos e que o desempenho varia entre os subsistemas. Os resultados também apontaram oportunidades de melhoria, principalmente na representação do crescimento recente da carga e na inclusão de novas variáveis explicativas.

![Resultados_Nordeste](imagens/resultado%20nordeste.png)
![Resultados_Nordeste](imagens/resultado%20norte.png)
![Resultados_Nordeste](imagens/resultado%20sudeste_centro-oeste.png)
![Resultados_Nordeste](imagens/resultado%20sul.png)
