# TCC — Identificação de Funcionalidades e Sentimentos em Avaliações de Aplicativos

Pipeline de mineração de dados que descobre **funcionalidades** mencionadas em
avaliações da Google Play (via LDA) e as associa aos **sentimentos** dos usuários
(léxico + aprendizado de máquina), gerando uma matriz de quais recursos são
elogiados e quais geram insatisfação.

Autor: Anélio Gonçalves Caldas — CEFET-MG.

## Etapas

| Etapa | Módulo | Capítulo do TCC |
|-------|--------|-----------------|
| 1. Coleta (web scraping) | `src/coleta.py` | 4.3 |
| 2. Pré-processamento | `src/preprocessamento.py` | 2.2.1 / 4 |
| 3. Modelagem de tópicos (LDA) | `src/topicos.py` | 2.3 / 4.4 |
| 4. Análise de sentimentos | `src/sentimentos.py` | 2.4 / 4.4 |
| 5. Associação func. × sentimento | `src/associacao.py` | 4.4 |
| Validação (truth set) | `src/amostragem.py` + rotulagem manual | 4.5 |

## Como rodar (Windows / PowerShell)

```powershell
# 1. Ambiente virtual (já criado em .venv)
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt

# 2. Pipeline completo
py main.py tudo

# ou etapa a etapa:
py main.py coleta
py main.py preprocessa
py main.py topicos
py main.py sentimentos
py main.py associa
```

## Trocar o aplicativo analisado

Edite `APP_ID` e `APP_NOME` em `config.py`. O `APP_ID` é o valor de `id=` na URL
da Play Store (ex.: `com.nu.production` para o Nubank).

## Decisões técnicas

- **Python 3.14** — `gensim` não compila nessa versão, então o **LDA usa
  `scikit-learn`** (mesmo modelo de Blei, Ng & Jordan, 2003) e a **coerência de
  tópicos (UMass)** é calculada manualmente em `src/topicos.py`.
- **Léxico de sentimentos** — coloque o OpLexicon ou SentiLex-PT em `resources/`
  (`resources/lexico.csv` com colunas `palavra,polaridade`, ou
  `resources/oplexicon.txt`). Sem isso, um léxico-semente embutido é usado
  apenas para o pipeline não quebrar.

## Validação (truth set)

1. `py main.py amostra` gera `data/processed/amostra_para_rotular_<APP_ID>.csv`.
2. Preencha as colunas `funcionalidade` e `sentimento` manualmente.
3. Salve como `amostra_rotulada_<APP_ID>.csv` — será usada para treinar os
   classificadores supervisionados e comparar tópicos automáticos × anotação
   humana (precisão, recall, F1).
