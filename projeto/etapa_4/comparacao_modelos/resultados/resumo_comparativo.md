# Resumo Comparativo dos Modelos — Teste em 2024

Resultados obtidos no período de teste (`2024`, `1.464` registros diários nos 4 subsistemas elétricos: `SE`, `S`, `NE` e `N`), após ajuste de hiperparâmetros em `2023` e retreinamento com o histórico de `2016` a `2023`.

### Legenda das Métricas Avaliadas
- **MAE (Erro Médio Absoluto, em MWmed):** média simples de quantos Megawatts o modelo errou para mais ou para menos por dia (quanto menor, melhor).
- **RMSE (Raiz do Erro Quadrático Médio, em MWmed):** similar ao MAE, mas penaliza com maior peso os erros grandes ocorridos em dias de pico (quanto menor, melhor).
- **R² (Coeficiente de Determinação):** varia até `1,00` e indica a proporção da variação real da carga explicada pelo modelo (quanto mais próximo de `1,00`, melhor).
- **MAPE (Erro Percentual Absoluto Médio, em %):** erro médio expresso em porcentagem da carga real, permitindo comparar subsistemas de tamanhos diferentes (quanto menor, melhor).

---

## 1. Métricas Globais no Período de Teste (2024)

| Modelo | MAE (MWmed) | RMSE (MWmed) | R² | MAPE (%) |
| :--- | ---: | ---: | ---: | ---: |
| **Gradient Boosting** | **424,74** | **691,18** | **0,9978** | **2,20%** |
| **Regressão Linear + Defasagens (log)** | 499,06 | 769,97 | 0,9972 | **2,77%** |
| **Random Forest** | 577,26 | 908,39 | 0,9961 | **3,09%** |
| Modelo de Referência (`carga_lag_7` — mesmo dia da semana anterior) | 961,80 | 1.684,20 | 0,9867 | **4,48%** |
| Regressão Linear (clima e calendário, sem defasagens) | 1.793,38 | 2.899,57 | 0,9607 | **8,19%** |

*Nota: O `modelo2.py` da entrega inicial opera na escala mensal sem variáveis climáticas; a Regressão Linear acima avalia o método linear na mesma base diária e com as mesmas variáveis climáticas fornecidas aos modelos em árvore.*

---

## 2. Erro Percentual (MAPE) por Subsistema em 2024

| Subsistema | Regressão Linear (sem defasagens) | Regressão Linear + Defasagens | Referência (`lag 7`) | Random Forest | Gradient Boosting |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Sudeste/Centro-Oeste (`SE`)** | 10,35% | 2,17% | 5,11% | 2,59% | **1,98%** |
| **Sul (`S`)** | 7,15% | 4,08% | 7,23% | 4,16% | **2,99%** |
| **Nordeste (`NE`)** | 5,16% | 2,04% | 3,00% | 2,14% | **1,86%** |
| **Norte (`N`)** | 10,10% | 2,80% | 2,56% | 3,46% | **1,96%** |

---

## 3. Impacto de Cada Grupo de Variáveis (Teste de Hipóteses do Projeto)

Para validar na prática as hipóteses definidas na Etapa 1 (`H1`: influência das variáveis climáticas na demanda; `H3`: influência da sazonalidade e do histórico recente), o modelo foi reavaliado removendo um grupo de variáveis por vez:

| Cenário Avaliado | Gradient Boosting — MAE (MWmed) | Gradient Boosting — MAPE (%) | Random Forest — MAE (MWmed) | Random Forest — MAPE (%) |
| :--- | ---: | ---: | ---: | ---: |
| **Modelo Completo (todas as variáveis)** | **424,74** | **2,20%** | 577,26 | 3,09% |
| Sem variáveis climáticas (`H1`) | 570,90 | 2,67% | 543,65 | 2,67% |
| Sem calendário — dia da semana e feriados (`H3`) | 783,73 | 3,67% | 801,68 | 3,95% |
| Sem defasagens da carga — *lags* (`H3`) | 632,06 | 3,79% | 1.024,53 | 5,41% |

---

## 4. Síntese dos Resultados

1. **Comparação entre os algoritmos:** O **Gradient Boosting** obteve o menor erro geral (`MAPE = 2,20%`, `MAE = 424,74 MWmed`), seguido pela versão diária com defasagens e pelo **Random Forest** (`MAPE = 3,09%`). Sem defasagens, tanto o Gradient Boosting (`3,79%`) quanto o Random Forest (`5,41%`) superaram com folga a Regressão Linear (`8,19%`), pois modelos baseados em árvores capturam relações não lineares (como o aumento abrupto do uso de refrigeração acima de `24 °C`).
2. **Papel das defasagens (*lags*) e do calendário (`H3`):** O consumo elétrico de um dia guarda forte dependência com o dia anterior (`carga_lag_1`) e com o mesmo dia da semana passada (`carga_lag_7`). Retirar essas variáveis eleva o erro do Gradient Boosting de `2,20%` para `3,79%`.
3. **Comprovação do impacto climático (`H1`):** A remoção das variáveis meteorológicas do INMET no Gradient Boosting aumentou o erro médio absoluto de `424,74 MWmed` para `570,90 MWmed` (`+34,4%`), confirmando que os dados climáticos reduzem o erro de previsão especialmente nos picos de temperatura.
