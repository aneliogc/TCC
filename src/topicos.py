"""
Etapa 3 - Modelagem de topicos (LDA) para descoberta de funcionalidades.

Usa sklearn.decomposition.LatentDirichletAllocation (mesmo modelo probabilistico
de Blei, Ng & Jordan, 2003). Como o gensim nao instala no Python 3.14, a metrica
de coerencia de topicos (UMass) e implementada aqui manualmente.

Fluxo:
  - vetoriza os textos em Bag-of-Words (CountVectorizer)
  - treina o LDA para cada k em LDA_TOPICOS_GRID
  - calcula coerencia UMass de cada modelo para apoiar a escolha de k
  - exporta os topicos (top termos) e a distribuicao topico-por-documento
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import config


def vetorizar(textos: list[str]) -> tuple[CountVectorizer, "np.ndarray"]:
    vect = CountVectorizer(
        min_df=config.BOW_MIN_DF,
        max_df=config.BOW_MAX_DF,
        max_features=config.BOW_MAX_FEATURES,
    )
    X = vect.fit_transform(textos)
    return vect, X


def treinar_lda(X, n_topicos: int) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(
        n_components=n_topicos,
        max_iter=config.LDA_MAX_ITER,
        learning_method="batch",
        random_state=config.LDA_RANDOM_STATE,
        evaluate_every=-1,
    )
    lda.fit(X)
    return lda


def top_termos_por_topico(lda, vect, n: int = config.LDA_TOP_TERMOS) -> list[list[str]]:
    termos = np.array(vect.get_feature_names_out())
    topicos = []
    for peso in lda.components_:
        idx = peso.argsort()[::-1][:n]
        topicos.append(list(termos[idx]))
    return topicos


def coerencia_umass(lda, vect, X, n_top: int = 10) -> float:
    """
    Coerencia UMass media entre os topicos.

    UMass(t) = soma_{i<j} log( (D(w_i, w_j) + 1) / D(w_j) )
    onde D(w) e o numero de documentos que contem w e D(w_i, w_j) o numero
    de documentos que contem ambos. Valores menos negativos = mais coerente.
    """
    Xb = (X > 0).astype(int)              # matriz binaria doc-termo
    doc_freq = np.asarray(Xb.sum(axis=0)).ravel()  # D(w)
    termos_idx = [peso.argsort()[::-1][:n_top] for peso in lda.components_]

    coerencias = []
    for idx in termos_idx:
        score = 0.0
        pares = 0
        for a in range(1, len(idx)):
            for b in range(a):
                wi, wj = idx[a], idx[b]
                # co-ocorrencia D(w_i, w_j)
                co = int(Xb[:, wi].multiply(Xb[:, wj]).sum())
                score += np.log((co + 1) / (doc_freq[wj] + 1e-12))
                pares += 1
        if pares:
            coerencias.append(score / pares)
    return float(np.mean(coerencias)) if coerencias else float("nan")


def buscar_melhor_k(textos: list[str], grid: list[int] = None) -> pd.DataFrame:
    """Treina o LDA para cada k do grid e retorna coerencia e perplexidade."""
    grid = grid or config.LDA_TOPICOS_GRID
    vect, X = vetorizar(textos)
    linhas = []
    for k in grid:
        print(f"  treinando LDA com k={k}...", flush=True)
        lda = treinar_lda(X, k)
        linhas.append({
            "n_topicos": k,
            "coerencia_umass": coerencia_umass(lda, vect, X),
            "perplexidade": lda.perplexity(X),
        })
    return pd.DataFrame(linhas)


def ajustar_final(textos: list[str], n_topicos: int = config.LDA_N_TOPICOS):
    """Treina o modelo final e devolve (lda, vect, X, distribuicao doc-topico)."""
    vect, X = vetorizar(textos)
    lda = treinar_lda(X, n_topicos)
    doc_topico = lda.transform(X)          # proporcao de cada topico por doc
    return lda, vect, X, doc_topico


def carregar_stem_map() -> dict[str, str]:
    """Le o mapa radical -> palavra completa (gerado no pre-processamento)."""
    if config.ARQ_STEM_MAP.exists():
        d = pd.read_csv(config.ARQ_STEM_MAP)
        return dict(zip(d["stem"].astype(str), d["palavra"].astype(str)))
    return {}


def exemplos_topico(n: int, quantas: int = 8) -> None:
    """Imprime as avaliacoes mais representativas de um topico, para ajudar a
    nomear a funcionalidade. Uso: py main.py exemplos <n>."""
    if not config.ARQ_TOPICOS_DOC.exists():
        print("Rode 'py main.py topicos' antes.")
        return
    df = pd.read_csv(config.ARQ_TOPICOS_DOC)
    sub = df[df["topico_dominante"] == n].sort_values("topico_prob", ascending=False)
    if sub.empty:
        print(f"Nenhuma avaliacao com topico dominante {n}.")
        return
    print(f"Topico {n}: {len(sub)} avaliacoes. As {quantas} mais representativas:\n")
    for _, r in sub.head(quantas).iterrows():
        print(f"- ({r['topico_prob']:.2f}) {str(r['texto'])[:130]}")


def main() -> None:
    df = pd.read_csv(config.ARQ_PROC)
    textos = df["texto_bow"].fillna("").tolist()

    print("== Busca pelo numero de topicos (k) ==")
    tabela = buscar_melhor_k(textos)
    print(tabela.to_string(index=False))
    tabela.to_csv(config.ARQ_LDA_BUSCA, index=False)

    print(f"\n== Modelo final com k={config.LDA_N_TOPICOS} ==")
    lda, vect, X, doc_topico = ajustar_final(textos)
    topicos = top_termos_por_topico(lda, vect)

    # traduz os radicais para a palavra completa mais frequente (so exibicao;
    # o LDA continua operando sobre os radicais)
    stem_map = carregar_stem_map()
    def traduz(termos: list[str]) -> list[str]:
        return [stem_map.get(t, t) for t in termos]

    topicos = [traduz(t) for t in topicos]
    for i, termos in enumerate(topicos):
        print(f"  Topico {i:2d}: {', '.join(termos)}")

    # exporta top termos (ja legiveis)
    pd.DataFrame(
        {"topico": range(len(topicos)),
         "termos": [", ".join(t) for t in topicos]}
    ).to_csv(config.ARQ_LDA_TOPICOS, index=False)

    # gera/atualiza o template de rotulagem manual dos topicos DESTA rodada,
    # com a coluna 'funcionalidade' em branco para voce preencher olhando os
    # termos. Salve como rotulos_topicos_<APP_ID>.csv para nomear a matriz.
    template = config.ARQ_ROTULOS_TEMPLATE
    pd.DataFrame(
        {"topico": range(len(topicos)),
         "termos_top": [", ".join(t[:6]) for t in topicos],
         "funcionalidade": [""] * len(topicos)}
    ).to_csv(template, index=False, encoding="utf-8")
    print(f"\nTemplate de rotulagem dos topicos: {template}")
    print(f"  -> preencha 'funcionalidade' e salve como {config.ARQ_ROTULOS.name}")

    # exporta topico dominante por avaliacao (para a etapa de associacao)
    df["topico_dominante"] = doc_topico.argmax(axis=1)
    df["topico_prob"] = doc_topico.max(axis=1)
    # salva tambem a distribuicao completa topico-documento (prob de cada topico
    # por avaliacao), usada na QP3 multi-rotulo (limiar LDA_MIN_PROB)
    for k in range(doc_topico.shape[1]):
        df[f"prob_topico_{k}"] = doc_topico[:, k]
    df.to_csv(config.ARQ_TOPICOS_DOC, index=False, encoding="utf-8")
    print(f"\nResultados salvos em {config.DATA_RESULTS}")


if __name__ == "__main__":
    main()
