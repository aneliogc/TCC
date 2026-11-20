"""
Etapa 6 - Validacao.

Compara os resultados automaticos com a amostra rotulada manualmente (truth set),
respondendo as questoes de pesquisa:

  QP2 - qual abordagem de sentimento (lexico x supervisionado) e melhor?
        -> metricas do baseline lexico e dos classificadores, agora treinados
           sobre os rotulos MANUAIS (nao mais os rotulos fracos da nota).

  QP3 - a associacao automatica funcionalidade<->sentimento corresponde a
        interpretacao humana?
        -> concordancia entre o topico dominante (rotulado) de cada avaliacao e
           a funcionalidade anotada manualmente.

Requer o arquivo rotulado:
  data/processed/amostra_rotulada_<APP_ID>.csv
com as colunas 'id', 'sentimento' e (opcional) 'funcionalidade' preenchidas.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import config
from src import preprocessamento, sentimentos

ARQ_ROTULADA = config.ARQ_ROTULADA


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def carregar_rotulada() -> pd.DataFrame:
    if not ARQ_ROTULADA.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {ARQ_ROTULADA}\n"
            f"Gere a amostra com 'py main.py amostra', preencha as colunas "
            f"'sentimento' e 'funcionalidade' e salve com esse nome."
        )
    # utf-8-sig le corretamente arquivos salvos pelo Excel (com ou sem BOM)
    df = pd.read_csv(ARQ_ROTULADA, encoding="utf-8-sig")
    df["sentimento"] = df["sentimento"].map(_norm)
    df = df[df["sentimento"].isin(config.SENT_CLASSES)].copy()
    return df



# QP2 - sentimento

def validar_lexico(df: pd.DataFrame) -> None:
    """Baseline lexico avaliado contra os rotulos manuais."""
    df = df.copy()
    df["texto_limpo"] = df["texto"].map(preprocessamento.limpar)
    df["pred"] = sentimentos.aplicar_baseline(df)
    print("\n===== BASELINE LEXICO vs. anotacao humana =====")
    print(classification_report(df["sentimento"], df["pred"],
                                labels=config.SENT_CLASSES,
                                digits=3, zero_division=0))


def validar_supervisionado(df: pd.DataFrame) -> None:
    """Classificadores supervisionados treinados sobre o truth set manual."""
    df = df.copy()
    df["texto_limpo"] = df["texto"].map(preprocessamento.limpar)
    print("\n===== SUPERVISIONADO (treino sobre rotulos manuais) =====")
    sentimentos.treinar_supervisionado(df, "sentimento")

# QP3 - associacao funcionalidade e sentimento

def validar_funcionalidades(df: pd.DataFrame) -> None:
    """
    Concordancia (aproximada) entre a funcionalidade atribuida automaticamente
    (rotulo do topico dominante) e a funcionalidade anotada manualmente.

    Como a anotacao manual e texto livre, a correspondencia e feita por
    casamento de subtring normalizada entre o rotulo do topico e a anotacao.
    """
    if "funcionalidade" not in df.columns or df["funcionalidade"].isna().all():
        print("\n[QP3] coluna 'funcionalidade' vazia - pulei a validacao de topicos.")
        return

    # topico dominante (e distribuicao completa) de cada avaliacao da amostra
    if not config.ARQ_TOPICOS_DOC.exists():
        print("\n[QP3] rode 'py main.py topicos' antes de validar funcionalidades.")
        return
    doc = pd.read_csv(config.ARQ_TOPICOS_DOC)
    prob_cols = sorted([c for c in doc.columns if c.startswith("prob_topico_")],
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    topicos = doc[["id", "topico_dominante"] + prob_cols]
    rotulos = pd.read_csv(config.ARQ_ROTULOS)
    mapa = dict(zip(rotulos["topico"], rotulos["funcionalidade"]))

    # ordem canonica das 8 funcionalidades e suas palavras-chave (para casar a
    # anotacao humana em texto livre com cada funcionalidade)
    topicos_ord = sorted(mapa)
    nomes = [str(mapa[t]) for t in topicos_ord]        # nomes legiveis
    nomes_norm = [_norm(n) for n in nomes]

    def _keywords(nome_norm: str) -> list[str]:
        limpo = nome_norm.replace("/", " ").replace("(", " ").replace(")", " ")
        return [w for w in limpo.split() if len(w) > 3]

    kw = [_keywords(n) for n in nomes_norm]

    m = df.merge(topicos, on="id", how="inner")
    if m.empty:
        print("\n[QP3] nenhuma avaliacao da amostra casou com a base de topicos.")
        return
    m["auto_norm"] = m["topico_dominante"].map(mapa).map(_norm)
    m["func_manual"] = m["funcionalidade"].map(_norm)

    # avaliacoes marcadas como SEM funcionalidade especifica (genericas): nao ha
    # recurso humano para comparar, entao ficam fora e sao reportadas a parte.
    SEM_FUNC = {"", "nan", "none", "nenhuma", "nenhum", "sem funcionalidade",
                "generico", "generica", "nao se aplica", "na", "n a", "-"}
    m["sem_func"] = m["func_manual"].isin(SEM_FUNC)
    com = m[~m["sem_func"]].reset_index(drop=True)

    print("\n===== QP3: identificacao de funcionalidades (auto x humana) =====")
    n_sem = int(m["sem_func"].sum())
    print(f"  avaliacoes na amostra          : {len(m)}")
    print(f"  sem funcionalidade (genericas) : {n_sem} ({n_sem/len(m):.1%}) "
          f"- excluidas das metricas")
    if com.empty:
        print("  nenhuma avaliacao com funcionalidade para comparar.")
        return

    # matrizes indicadoras (n_avaliacoes x 8 funcionalidades)
    #   y_true : 1 em cada funcionalidade que o humano anotou (uma ou mais)
    #   y_dom  : 1 so na funcionalidade do topico DOMINANTE do LDA (uma por aval.)
    #   y_multi: 1 em todo topico com probabilidade >= LDA_MIN_PROB (uma ou mais)
    # topicos_ord[f] e o numero do topico cuja funcionalidade e a coluna f, logo
    # a coluna de probabilidade correspondente e prob_topico_{topicos_ord[f]}.
    n, K = len(com), len(nomes)
    y_true = np.zeros((n, K), dtype=int)
    y_dom = np.zeros((n, K), dtype=int)
    y_multi = np.zeros((n, K), dtype=int)
    tem_prob = len(prob_cols) > 0
    for i in range(n):
        manual = com.loc[i, "func_manual"]
        auto = com.loc[i, "auto_norm"]
        for f in range(K):
            if any(p in manual for p in kw[f]):
                y_true[i, f] = 1
            if auto == nomes_norm[f]:
                y_dom[i, f] = 1
            if tem_prob:
                pcol = f"prob_topico_{topicos_ord[f]}"
                if pcol in com.columns and com.loc[i, pcol] >= config.LDA_MIN_PROB:
                    y_multi[i, f] = 1
        # garante que o multi-rotulo tenha ao menos o topico dominante
        if tem_prob and y_multi[i].sum() == 0:
            y_multi[i] = y_dom[i]

    def _relatorio(y_pred: np.ndarray, titulo: str) -> None:
        acertos = int(((y_pred & y_true).sum(axis=1) > 0).sum())
        media_rotulos = y_pred.sum(axis=1).mean()
        print(f"\n  --- {titulo} ---")
        print(f"  funcionalidades por avaliacao  : {media_rotulos:.2f} (media)")
        print(f"  concordancia (acuracia)        : {acertos/n:.1%}  ({acertos}/{n})")
        print(classification_report(y_true, y_pred, target_names=nomes,
                                    digits=3, zero_division=0))

    print(f"  avaliacoes com funcionalidade  : {n}")
    _relatorio(y_dom, "Versao A: topico DOMINANTE (1 funcionalidade/avaliacao)")
    if tem_prob:
        _relatorio(y_multi,
                   f"Versao B: MULTI-ROTULO (prob >= {config.LDA_MIN_PROB})")
    else:
        print("\n  [multi-rotulo] rode 'py main.py topicos' novamente para gerar "
              "a distribuicao completa (colunas prob_topico_*) e habilitar a "
              "versao multi-rotulo.")

    # Em vez de comparar funcionalidades soltas, compara pares: cada
    # funcionalidade (anotada ou prevista) carrega a polaridade da avaliacao. Um
    # par so e acerto quando funcionalidade E sentimento coincidem. A polaridade
    # prevista vem de validacao cruzada (out-of-fold): cada avaliacao e prevista
    # por um modelo que NAO a viu no treino, evitando inflar o resultado.
    BIN = ("positivo", "negativo")
    yb = com["sentimento"].to_numpy()
    idx_bin = np.where(np.isin(yb, BIN))[0]
    if len(idx_bin) < 10:
        print("\n  [pares] subconjunto binario pequeno demais - pulei.")
        return

    textos = com["texto"].map(preprocessamento.limpar).fillna("").to_numpy()
    clf = make_pipeline(TfidfVectorizer(min_df=3, ngram_range=(1, 2)),
                        LinearSVC(class_weight="balanced"))
    pred_pol = np.full(n, "", dtype=object)
    pred_pol[idx_bin] = cross_val_predict(clf, textos[idx_bin], yb[idx_bin], cv=5)

    def _metr(T: np.ndarray, P: np.ndarray):
        p, r, f, _ = precision_recall_fscore_support(
            T, P, average="micro", zero_division=0)
        conc = float(((P & T).sum(axis=1) > 0).mean())
        return conc, p, r, f

    def _par_relatorio(y_pred: np.ndarray, titulo: str) -> None:
        yt, yp = y_true[idx_bin], y_pred[idx_bin]
        hp, pp = yb[idx_bin], pred_pol[idx_bin]
        pos_t = (hp == "positivo").astype(int)[:, None]
        neg_t = (hp == "negativo").astype(int)[:, None]
        pos_p = (pp == "positivo").astype(int)[:, None]
        neg_p = (pp == "negativo").astype(int)[:, None]
        T_par = np.hstack([yt * pos_t, yt * neg_t])
        P_par = np.hstack([yp * pos_p, yp * neg_p])
        c0, p0, r0, f0 = _metr(yt, yp)              # so funcionalidade
        c1, p1, r1, f1 = _metr(T_par, P_par)        # par funcionalidade+sentim.
        print(f"\n  --- {titulo} ---")
        print(f"  so funcionalidade : concord {c0:6.1%} | P {p0:.3f} | "
              f"R {r0:.3f} | F1 {f0:.3f}")
        print(f"  par func+sentim.  : concord {c1:6.1%} | P {p1:.3f} | "
              f"R {r1:.3f} | F1 {f1:.3f}")

    print("\n===== QP3-PARES: funcionalidade+sentimento (subconjunto binario) =====")
    print(f"  avaliacoes (bin, c/ funcionalidade) : {len(idx_bin)}")
    print(f"  polaridade prevista por             : SVM linear balanceado (CV 5-fold)")
    _par_relatorio(y_dom, "Versao A: topico DOMINANTE")
    if tem_prob:
        _par_relatorio(
            y_multi, f"Versao B: MULTI-ROTULO (prob >= {config.LDA_MIN_PROB})")


def main() -> None:
    try:
        df = carregar_rotulada()
    except FileNotFoundError as e:
        print(e)
        return
    print(f"Amostra rotulada: {len(df)} avaliacoes validas")
    print("Distribuicao dos rotulos manuais:")
    print(df["sentimento"].value_counts().to_string())

    validar_lexico(df)
    validar_supervisionado(df)
    validar_funcionalidades(df)


if __name__ == "__main__":
    main()
