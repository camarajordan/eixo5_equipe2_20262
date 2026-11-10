# Comparação de Algoritmos — Etapa 4

Implementação e comparação dos algoritmos de aprendizado de máquina sobre a camada Gold (`db_energia_clima_puc`, período de `2016` a `2024`, integrando dados diários de carga elétrica do **ONS** e variáveis meteorológicas do **INMET**).

---

## Relação com o `modelo2.py` e Estrutura dos Arquivos

O arquivo [`../modelo2.py`](../modelo2.py) (elaborado na entrega inicial da Etapa 4) avalia a Regressão Linear na escala **mensal** utilizando apenas tendência e mês (sem clima). Para comparar a Regressão Linear de forma justa com o **Gradient Boosting** e o **Random Forest**, os scripts desta pasta colocam todos os algoritmos na mesma base **diária** e com as mesmas variáveis climáticas do INMET:

1. **[`preparacao.py`](preparacao.py)**
   Consulta a camada Gold no Amazon Athena, agrega os dados climáticos por subsistema (`SE`, `S`, `NE`, `N`) e cria as variáveis de entrada:
   - **Calendário:** dia da semana, mês, ano, tendência contínua e indicador de feriados nacionais/facultativos;
   - **Defasagens da carga (*lags*):** valores passados da própria carga usados para prever o dia seguinte — carga de ontem (`carga_lag_1`), carga de 7 dias atrás (`carga_lag_7`) e média dos últimos 7 dias (`carga_media_7d`);
   - **Clima (INMET):** temperaturas máxima, média e mínima, precipitação, radiação global, velocidade do vento, temperatura do dia anterior (`temp_media_lag_1`) e média de 3 dias (`temp_media_3d`).

2. **[`01_regressao_linear.py`](01_regressao_linear.py)** — *Regressão Linear (diária com clima)*
   Aplica a Regressão Linear na granularidade diária com variáveis climáticas e de calendário, servindo de base direta de comparação na mesma escala dos modelos em árvore.

3. **[`02_gradient_boosting.py`](02_gradient_boosting.py)** — *Gradient Boosting (`HistGradientBoostingRegressor`)*
   Algoritmo de árvores sequenciais indicado na orientação, em que cada nova árvore corrige os erros das árvores anteriores.

4. **[`03_random_forest.py`](03_random_forest.py)** — *Random Forest (`RandomForestRegressor`)*
   Terceiro algoritmo avaliado pelo grupo, baseado em múltiplas árvores independentes treinadas em paralelo. Também gera as tabelas e gráficos consolidados em [`resultados/`](resultados/).

---

## Divisão Temporal e Etapas de Treinamento

A divisão segue a ordem cronológica da série (`2016–2024`):
1. **Treino inicial (`2016–2022`) e Validação (`2023`):** os modelos são treinados até `2022` e avaliados em `2023` apenas para escolher a melhor combinação de hiperparâmetros sem usar dados de `2024`.
2. **Retreinamento (`2016–2023`):** definida a melhor configuração, o modelo é treinado novamente incluindo o ano de `2023`, para incorporar o histórico mais recente antes de prever `2024`.
3. **Teste final (`2024`):** avaliação final das previsões nos `1.464` registros diários de `2024`.

---

## Execução

```bash
pip install -r requirements.txt
python 01_regressao_linear.py
python 02_gradient_boosting.py
python 03_random_forest.py
```

Os resultados consolidados encontram-se em [`resultados/resumo_comparativo.md`](resultados/resumo_comparativo.md).
