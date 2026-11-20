"""
Etapa 5 - Associacao funcionalidade <-> sentimento.

Cruza o topico dominante de cada avaliacao (funcionalidade descoberta pelo LDA)
com o sentimento classificado, gerando a matriz estruturada que e a contribuicao
central do TCC: quais recursos sao elogiados e quais geram insatisfacao.
"""
from __future__ import annotations

import pandas as pd

import config
from src import preprocessamento, sentimentos


def gerar_matriz(df: pd.DataFrame, coluna_sent: str,
                 termos_topico: dict[int, str],
                 rotulos_topico: dict[int, str] | None = None,
                 classes: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Contagem de sentimentos por topico (funcionalidade).

    `termos_topico` (obrigatorio) traz os termos de maior peso de CADA topico do
    modelo atual, usados como rotulo padrao. `rotulos_topico` (opcional) sao os
    nomes manuais; quando ausentes, os proprios termos nomeiam o topico. Os
    termos aparecem sempre em uma coluna, para tornar visivel qualquer
    descasamento entre um nome manual e o conteudo real do topico.
    """
    cls = list(classes) if classes is not None else list(config.SENT_CLASSES)
    tab = pd.crosstab(df["topico_dominante"], df[coluna_sent])
    for classe in cls:
        if classe not in tab.columns:
            tab[classe] = 0
    tab = tab[cls]
    tab["total"] = tab.sum(axis=1)
    tab["%_positivo"] = (tab["positivo"] / tab["total"] * 100).round(1)
    tab["%_negativo"] = (tab["negativo"] / tab["total"] * 100).round(1)

    # coluna de termos (sempre) e de nome (manual, ou os proprios termos)
    tab.insert(0, "termos_top", [termos_topico.get(i, "?") for i in tab.index])
    if rotulos_topico:
        tab.insert(0, "funcionalidade",
                   [rotulos_topico.get(i, f"topico_{i}") for i in tab.index])
    return tab.sort_values("total", ascending=False)


def carregar_termos_topico(n: int = 4) -> dict[int, str]:
    """Le os termos de maior peso de cada topico do modelo ATUAL."""
    if not config.ARQ_LDA_TOPICOS.exists():
        return {}
    d = pd.read_csv(config.ARQ_LDA_TOPICOS)
    return {int(r["topico"]): ", ".join(str(r["termos"]).split(", ")[:n])
            for _, r in d.iterrows()}


def carregar_rotulos_topico() -> dict[int, str]:
    """
    Le rotulos de funcionalidade atribuidos manualmente aos topicos, se houver.
    Formato: resources/rotulos_topicos_<APP_ID>.csv (colunas topico,funcionalidade)

    ATENCAO: esses nomes valem apenas para a rodada de LDA em que foram criados.
    Se o corpus ou os parametros mudarem, renomeie os topicos novamente (o
    'topico N' de uma rodada nao corresponde ao 'topico N' de outra).
    """
    if config.ARQ_ROTULOS.exists():
        d = pd.read_csv(config.ARQ_ROTULOS)
        return dict(zip(d["topico"].astype(int), d["funcionalidade"].astype(str)))
    return {}


def main() -> None:
    df = pd.read_csv(config.ARQ_TOPICOS_DOC)
    df["texto_limpo"] = df["texto_limpo"].fillna("")

    # Sentimento do corpus para a matriz: SVM binario (pos/neg) treinado nas
    # anotacoes manuais -- o melhor classificador da QP2 -- aplicado a todo o
    # corpus. Cai no baseline lexico (3 classes) se nao houver amostra rotulada.
    if config.ARQ_ROTULADA.exists():
        rot = pd.read_csv(config.ARQ_ROTULADA, encoding="utf-8-sig")
        rot["sentimento"] = rot["sentimento"].astype(str).str.strip().str.lower()
        rot["texto_limpo"] = rot["texto"].map(preprocessamento.limpar)
        coluna_sent = "sent_svm"
        classes: tuple[str, ...] = ("positivo", "negativo")
        print("== Sentimento da matriz: SVM binario (pos/neg) ==")
        df[coluna_sent] = sentimentos.prever_corpus_binario(df, rot)
    else:
        coluna_sent = "sent_lexico"
        classes = tuple(config.SENT_CLASSES)
        print("== Sentimento da matriz: baseline lexico (sem amostra rotulada) ==")
        df[coluna_sent] = sentimentos.aplicar_baseline(df)

    termos = carregar_termos_topico()
    rotulos = carregar_rotulos_topico()
    if not rotulos:
        print(f"[info] sem {config.ARQ_ROTULOS.name} -> rotulando pelos termos "
              "do modelo atual. Preencha o template para dar nomes.")
    matriz = gerar_matriz(df, coluna_sent, termos, rotulos, classes)

    print("== Matriz funcionalidade x sentimento ==")
    print(matriz.to_string())

    saida = config.ARQ_MATRIZ
    matriz.to_csv(saida, encoding="utf-8")
    print(f"\nMatriz salva em {saida}")

    # destaques
    print("\nFuncionalidades com mais insatisfacao (%_negativo):")
    print(matriz.sort_values("%_negativo", ascending=False)
          [["%_negativo", "total"]].head(5).to_string())


if __name__ == "__main__":
    main()
