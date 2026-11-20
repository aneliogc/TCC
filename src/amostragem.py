"""
Utilitario - Geracao da amostra para rotulagem manual (truth set).

Seleciona uma amostra estratificada por nota para ser rotulada manualmente
quanto a: (a) funcionalidade(s) mencionada(s) e (b) sentimento (positivo/
neutro/negativo). Esse conjunto e a base de validacao descrita na Secao 4.5.
"""
from __future__ import annotations

import pandas as pd

import config


def amostrar(df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    """Amostra estratificada por nota, mantendo representatividade."""
    n = min(n, len(df))
    frac = n / len(df)
    partes = []
    for _, grupo in df.groupby("nota"):
        k = max(1, round(len(grupo) * frac))
        partes.append(grupo.sample(min(k, len(grupo)),
                                   random_state=config.RANDOM_STATE))
    amostra = pd.concat(partes)
    amostra = amostra.sample(frac=1, random_state=config.RANDOM_STATE)  # embaralha
    saida = amostra[["id", "texto", "nota"]].copy()
    # colunas em branco para anotar
    saida["funcionalidade"] = ""   # ex.: "pagamento; login"
    saida["sentimento"] = ""       # positivo | neutro | negativo
    return saida.reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(config.ARQ_PROC)
    amostra = amostrar(df)
    amostra.to_csv(config.ARQ_AMOSTRA_ROTULAR, index=False, encoding="utf-8-sig")
    print(f"{len(amostra)} avaliacoes salvas para rotulagem manual em:")
    print(f"  {config.ARQ_AMOSTRA_ROTULAR}")
    print("\nPreencha as colunas 'funcionalidade' e 'sentimento' e salve como")
    print("  data/processed/amostra_rotulada_<APP_ID>.csv para a validacao.")


if __name__ == "__main__":
    main()
