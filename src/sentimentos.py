"""
Etapa 4 - Analise de sentimentos.

Estudo comparativo entre duas abordagens (Capitulo 4 do TCC):

  A) Baseline lexico (nao supervisionado): soma a polaridade das palavras a
     partir de um lexico de opiniao para o portugues (OpLexicon / SentiLex-PT),
     com tratamento de negacao e intensificadores.

  B) Supervisionada: Naive Bayes, SVM e Regressao Logistica sobre TF-IDF,
     treinados numa amostra rotulada manualmente quanto ao sentimento.

O lexico deve ser colocado em resources/. Se nenhum for encontrado, um pequeno
lexico-semente embutido e usado para o pipeline nao quebrar (com aviso).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import config

# ===========================================================================
# A) BASELINE LEXICO
# ===========================================================================
NEGACOES = {"nao", "nunca", "jamais", "nem", "nenhum", "nenhuma"}
INTENSIFICADORES = {"muito": 1.5, "super": 1.5, "extremamente": 2.0,
                    "bastante": 1.3, "pouco": 0.5, "meio": 0.7}

# Lexico-semente minimo (fallback).
LEXICO_SEMENTE = {
    "bom": 1, "otimo": 1, "excelente": 1, "maravilhoso": 1, "rapido": 1,
    "facil": 1, "pratico": 1, "adorei": 1, "gostei": 1, "perfeito": 1,
    "seguro": 1, "confiavel": 1, "recomendo": 1, "funciona": 1, "ajuda": 1,
    "ruim": -1, "pessimo": -1, "horrivel": -1, "lento": -1, "dificil": -1,
    "problema": -1, "erro": -1, "trava": -1, "travando": -1, "bug": -1,
    "demora": -1, "demorado": -1, "falha": -1, "pior": -1, "odiei": -1,
    "golpe": -1, "roubo": -1, "cobranca": -1, "instavel": -1, "cai": -1,
}


def carregar_lexico() -> dict[str, float]:
    """
    Tenta carregar um lexico de resources/. Aceita:
      - resources/lexico.csv  com colunas: palavra,polaridade
      - resources/oplexicon.txt  formato OpLexicon v3.0 (palavra,tipo,pol,...)
    Cai no lexico-semente se nada for encontrado.
    """
    csv = config.RESOURCES / "lexico.csv"
    if csv.exists():
        d = pd.read_csv(csv)
        print(f"  lexico carregado: {csv.name} ({len(d)} termos)")
        return dict(zip(d["palavra"].astype(str),
                        d["polaridade"].astype(float)))

    op = config.RESOURCES / "oplexicon.txt"
    if op.exists():
        lex: dict[str, float] = {}
        for linha in op.read_text(encoding="utf-8", errors="ignore").splitlines():
            partes = linha.split(",")
            if len(partes) >= 3:
                try:
                    lex[partes[0].strip()] = float(partes[2])
                except ValueError:
                    continue
        print(f"  lexico OpLexicon carregado ({len(lex)} termos)")
        return lex

    print("  [AVISO] nenhum lexico em resources/ -> usando lexico-semente "
          "embutido (baixe OpLexicon/SentiLex-PT para resultados reais).")
    return dict(LEXICO_SEMENTE)


def polaridade_lexico(tokens: list[str], lex: dict[str, float]) -> str:
    """Classifica um texto (lista de tokens SEM stemming) via lexico."""
    escore = 0.0
    intensidade = 1.0
    negado = False
    for tok in tokens:
        if tok in INTENSIFICADORES:
            intensidade = INTENSIFICADORES[tok]
            continue
        if tok in NEGACOES:
            negado = True
            continue
        if tok in lex:
            valor = lex[tok] * intensidade
            if negado:
                valor = -valor
            escore += valor
        # um modificador vale so para a palavra pontuada seguinte (reseta a cada token)
        intensidade = 1.0
        negado = False
    if escore > 0:
        return "positivo"
    if escore < 0:
        return "negativo"
    return "neutro"


def aplicar_baseline(df: pd.DataFrame) -> pd.Series:
    lex = carregar_lexico()
    # usa texto_limpo (sem stemming) tokenizado por espaco
    return df["texto_limpo"].fillna("").map(
        lambda t: polaridade_lexico(t.split(), lex)
    )


# ===========================================================================
# B) SUPERVISIONADO
# ===========================================================================
def rotulo_fraco_por_nota(nota: int) -> str:
    """Rotulo heuristico a partir da nota (apoio, NAO e o truth set)."""
    if nota <= 2:
        return "negativo"
    if nota == 3:
        return "neutro"
    return "positivo"


def treinar_supervisionado(df: pd.DataFrame, coluna_rotulo: str):
    """
    Treina NB, SVM e Regressao Logistica e imprime as metricas de teste.
    `coluna_rotulo` deve conter os rotulos (positivo/neutro/negativo).
    """
    dados = df[df[coluna_rotulo].isin(config.SENT_CLASSES)].copy()
    X_txt = dados["texto_limpo"].fillna("")
    y = dados[coluna_rotulo]

    Xtr, Xte, ytr, yte = train_test_split(
        X_txt, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )

    vect = TfidfVectorizer(min_df=config.TFIDF_MIN_DF, ngram_range=config.TFIDF_NGRAM)
    Xtr_v = vect.fit_transform(Xtr)
    Xte_v = vect.transform(Xte)

    modelos = {
        "Naive Bayes": MultinomialNB(),
        "SVM (Linear)": LinearSVC(),
        "Regressao Logistica": LogisticRegression(max_iter=1000),
    }
    resultados = {}
    for nome, modelo in modelos.items():
        modelo.fit(Xtr_v, ytr)
        pred = modelo.predict(Xte_v)
        print(f"\n----- {nome} -----")
        print(classification_report(yte, pred, digits=3, zero_division=0))
        resultados[nome] = {
            "modelo": modelo,
            "relatorio": classification_report(
                yte, pred, output_dict=True, zero_division=0),
            "matriz_confusao": confusion_matrix(
                yte, pred, labels=config.SENT_CLASSES),
        }
    return vect, resultados


def prever_corpus_binario(df_corpus: pd.DataFrame,
                          df_rotulada: pd.DataFrame) -> pd.Series:
    """Treina um SVM linear binario (positivo/negativo) sobre as anotacoes
    MANUAIS e preve a polaridade de TODO o corpus.
    """
    treino = df_rotulada[df_rotulada["sentimento"].isin(["positivo", "negativo"])].copy()
    clf = make_pipeline(
        TfidfVectorizer(min_df=config.TFIDF_MIN_DF, ngram_range=config.TFIDF_NGRAM),
        LinearSVC(class_weight="balanced"),
    )
    clf.fit(treino["texto_limpo"].fillna(""), treino["sentimento"])
    print(f"  SVM binario treinado em {len(treino)} avaliacoes anotadas "
          f"(pos/neg) e aplicado a {len(df_corpus)} do corpus")
    return pd.Series(clf.predict(df_corpus["texto_limpo"].fillna("")),
                     index=df_corpus.index)


def main() -> None:
    df = pd.read_csv(config.ARQ_PROC)

    # --- Baseline lexico (roda sem rotulos) ---
    print("== A) Baseline lexico ==")
    df["sent_lexico"] = aplicar_baseline(df)
    print(df["sent_lexico"].value_counts().to_string())

    # --- Supervisionado ---
    print("\n== B) Supervisionado ==")
    if "sentimento" in df.columns and df["sentimento"].notna().any():
        coluna = "sentimento"
        print("  usando rotulos manuais da coluna 'sentimento'")
    else:
        coluna = "sent_nota"
        df[coluna] = df["nota"].map(rotulo_fraco_por_nota)
        print("  [AVISO] sem rotulos manuais -> treinando com rotulo fraco "
              "derivado da nota (apenas demonstrativo, NAO e o truth set).")

    treinar_supervisionado(df, coluna)

    saida = config.DATA_PROC / f"avaliacoes_sentimento_{config.APP_ID}.csv"
    df.to_csv(saida, index=False, encoding="utf-8")
    print(f"\nSaida salva em {saida}")


if __name__ == "__main__":
    main()
