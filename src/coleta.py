"""
Etapa 1 - Coleta de dados (web scraping da Google Play Store).

Coleta avaliacoes textuais em portugues usando a biblioteca google-play-scraper.
Para cada avaliacao armazena: id, texto, nota (score), data, versao do app e
quantidade de "curtidas" (thumbsUp).

A coleta e paginada: a API do Play retorna lotes e um token de continuacao.
"""
from __future__ import annotations

import time
import pandas as pd
from google_play_scraper import Sort, reviews

import config


def coletar(app_id: str = config.APP_ID,
            lang: str = config.LANG,
            country: str = config.COUNTRY,
            alvo: int = config.MAX_AVALIACOES,
            lote: int = 200) -> pd.DataFrame:
    """Coleta ate `alvo` avaliacoes do app, paginando de `lote` em `lote`."""
    coletadas: list[dict] = []
    token = None
    while len(coletadas) < alvo:
        resultado, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=lote,
            continuation_token=token,
        )
        if not resultado:
            break
        coletadas.extend(resultado)
        print(f"  coletadas {len(coletadas)} avaliacoes...", flush=True)
        if token is None:
            break
        time.sleep(0.5)  # respeita a plataforma

    df = pd.DataFrame(coletadas)
    if df.empty:
        return df

    # Seleciona e renomeia as colunas de interesse (conforme o protocolo do TCC)
    colunas = {
        "reviewId": "id",
        "content": "texto",
        "score": "nota",
        "at": "data",
        "reviewCreatedVersion": "versao",
        "thumbsUpCount": "curtidas",
    }
    df = df[[c for c in colunas if c in df.columns]].rename(columns=colunas)
    df = df.drop_duplicates(subset="id")
    df = df[df["texto"].notna() & (df["texto"].str.strip() != "")]
    return df.reset_index(drop=True)


def main() -> None:
    # Dataset congelado: nao recoleta se o arquivo ja existe, garantindo
    # reprodutibilidade (o LDA e os rotulos dependem de um corpus fixo).
    if config.ARQ_RAW.exists():
        n = len(pd.read_csv(config.ARQ_RAW))
        print(f"Dataset ja existe ({n} avaliacoes): {config.ARQ_RAW}")
        print("Pulando coleta. Para recoletar do zero, apague esse arquivo antes.")
        return
    print(f"Coletando avaliacoes de {config.APP_NOME} ({config.APP_ID})...")
    df = coletar()
    if df.empty:
        print("Nenhuma avaliacao coletada. Verifique o APP_ID e a conexao.")
        return
    df.to_csv(config.ARQ_RAW, index=False, encoding="utf-8")
    print(f"\n{len(df)} avaliacoes salvas em {config.ARQ_RAW}")
    print("\nDistribuicao das notas:")
    print(df["nota"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
