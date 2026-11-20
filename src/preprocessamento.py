"""
Etapa 2 - Pre-processamento textual (PLN).

Sequencia de transformacoes no texto:
  1. Limpeza e remocao de ruido (URLs, numeros, emojis, pontuacao)
  2. Normalizacao (caixa baixa, remocao de acentos opcional)
  3. Tokenizacao
  4. Remocao de stop words (NLTK portugues + stopwords de dominio)
  5. Stemming (RSLP, especifico para o portugues)

Mantem duas colunas de saida:
  - texto_limpo  : texto normalizado, sem stopwords, ainda legivel (para o
                   baseline lexico de sentimentos, que precisa das palavras)
  - tokens       : lista de tokens processados (com stemming) para o LDA
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import nltk
import pandas as pd

import config


# Recursos do NLTK

def _garantir_nltk() -> None:
    for pacote, caminho in [
        ("stopwords", "corpora/stopwords"),
        ("rslp", "stemmers/rslp"),
    ]:
        try:
            nltk.data.find(caminho)
        except LookupError:
            nltk.download(pacote, quiet=True)


_garantir_nltk()
from nltk.corpus import stopwords as _sw          # noqa: E402
from nltk.stem import RSLPStemmer                 # noqa: E402

_STEMMER = RSLPStemmer()
_STOPWORDS = set(_sw.words("portuguese")) | config.STOPWORDS_DOMINIO
# stems das palavras genericas/avaliativas a remover apos o stemming
_STOPSTEMS = {_STEMMER.stem(w) for w in config.STOPWORDS_EXTRA}


# Expressoes regulares de limpeza

_RE_URL = re.compile(r"http\S+|www\.\S+")
_RE_MENCAO = re.compile(r"[@#]\w+")
_RE_NAO_LETRA = re.compile(r"[^a-zà-ÿ\s]")   # mantem letras (com acento) e espaco
_RE_ESPACO = re.compile(r"\s+")
_RE_ALONGAMENTO = re.compile(r"(.)\1{2,}")   # "muitooo" -> "muito"


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def limpar(texto: str) -> str:
    """Limpeza + normalizacao, preservando as palavras (sem stemming)."""
    texto = str(texto).lower()
    texto = _RE_URL.sub(" ", texto)
    texto = _RE_MENCAO.sub(" ", texto)
    texto = _RE_ALONGAMENTO.sub(r"\1\1", texto)   # reduz alongamentos
    texto = _RE_NAO_LETRA.sub(" ", texto)          # remove numeros/emojis/pontos
    texto = _RE_ESPACO.sub(" ", texto).strip()
    return texto


# Pre-calcula stopwords sem acento uma unica vez (otimizacao)
_STOPWORDS_SA = {remover_acentos(s) for s in _STOPWORDS}


def _tokens_validos(texto_limpo: str) -> list[str]:
    out = []
    for tok in texto_limpo.split():
        if len(tok) < 3:
            continue
        if tok in _STOPWORDS or remover_acentos(tok) in _STOPWORDS_SA:
            continue
        out.append(tok)
    return out


def stemizar(tokens: list[str]) -> list[str]:
    return [_STEMMER.stem(t) for t in tokens]


def processar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todo o pre-processamento e devolve o DataFrame enriquecido."""
    df = df.copy()
    df["texto_limpo"] = df["texto"].map(limpar)
    df["tokens_sem_stop"] = df["texto_limpo"].map(_tokens_validos)

    if config.USAR_STEMMER:
        df["tokens"] = df["tokens_sem_stop"].map(
            lambda ts: [s for s in stemizar(ts) if s not in _STOPSTEMS])
    else:
        df["tokens"] = df["tokens_sem_stop"]

    # texto pronto para o Bag-of-Words do LDA (tokens unidos por espaco)
    df["texto_bow"] = df["tokens"].map(lambda ts: " ".join(ts))

    if config.USAR_STEMMER:
        _salvar_stem_map(df["tokens_sem_stop"])

    # descarta avaliacoes curtas demais apos limpeza
    antes = len(df)
    df = df[df["tokens"].map(len) >= config.MIN_TOKENS].reset_index(drop=True)
    print(f"  removidas {antes - len(df)} avaliacoes curtas (< {config.MIN_TOKENS} tokens)")
    return df


def _salvar_stem_map(coluna_tokens_originais) -> None:
    """Constroi e salva o mapa radical -> palavra completa mais frequente.

    Para cada palavra original (antes do stemming), conta com que forma completa
    cada radical aparece; a mais frequente vira a representante para exibicao.
    """
    contadores: dict[str, Counter] = {}
    for originais in coluna_tokens_originais:
        for palavra in originais:
            radical = _STEMMER.stem(palavra)
            contadores.setdefault(radical, Counter())[palavra] += 1
    linhas = [(radical, cont.most_common(1)[0][0])
              for radical, cont in contadores.items()]
    pd.DataFrame(linhas, columns=["stem", "palavra"]).to_csv(
        config.ARQ_STEM_MAP, index=False, encoding="utf-8")


def main() -> None:
    print(f"Lendo {config.ARQ_RAW}...")
    df = pd.read_csv(config.ARQ_RAW)
    print(f"  {len(df)} avaliacoes carregadas")
    df = processar_df(df)
    # Salva sem a coluna lista (tokens) serializada de forma legivel
    df_out = df.copy()
    df_out["tokens"] = df_out["tokens"].map(lambda ts: " ".join(ts))
    df_out.drop(columns=["tokens_sem_stop"], inplace=True)
    df_out.to_csv(config.ARQ_PROC, index=False, encoding="utf-8")
    print(f"\n{len(df_out)} avaliacoes pre-processadas salvas em {config.ARQ_PROC}")
    print("\nExemplo:")
    for _, row in df_out.head(3).iterrows():
        print(f"  original : {row['texto'][:80]}")
        print(f"  bow      : {row['texto_bow'][:80]}\n")


if __name__ == "__main__":
    main()
