"""
Etapa extra - Geracao de graficos para o TCC.

Produz figuras (PNG, 300 dpi) a partir dos resultados ja gravados, prontas para
colar no texto. Uso:

  py main.py graficos

Gera, para o APP_ID atual (config.py):
  1. matriz_sentimento_<APP_ID>.png  - % de sentimento por funcionalidade (QP1)
  2. qp2_comparativo_<APP_ID>.png     - acuracia/F1 por abordagem (QP2)

As figuras ficam em data/results/figuras/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                  # backend sem janela (so salva arquivo)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config

# cores consistentes para os sentimentos
COR_NEG = "#d9534f"
COR_NEU = "#b8b8b8"
COR_POS = "#5cb85c"

FIG_DIR = config.DATA_RESULTS / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) Matriz funcionalidade x sentimento (QP1)
# ---------------------------------------------------------------------------
def grafico_matriz() -> None:
    if not config.ARQ_MATRIZ.exists():
        print(f"[graficos] {config.ARQ_MATRIZ.name} nao encontrado - rode 'associa'.")
        return
    m = pd.read_csv(config.ARQ_MATRIZ)
    # ordena por volume total (menor embaixo p/ leitura natural de cima p/ baixo)
    m = m.sort_values("total", ascending=True)

    nomes = m["funcionalidade"].astype(str).tolist()
    tot = m["total"].to_numpy()
    pneg = m["negativo"].to_numpy() / tot * 100
    # a matriz do SVM binario nao tem faixa neutra; o grafico se adapta
    tem_neu = "neutro" in m.columns
    pneu = (m["neutro"].to_numpy() / tot * 100) if tem_neu else tot * 0.0
    ppos = m["positivo"].to_numpy() / tot * 100

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(nomes) + 1.5))
    ax.barh(nomes, pneg, color=COR_NEG, label="negativo")
    if tem_neu:
        ax.barh(nomes, pneu, left=pneg, color=COR_NEU, label="neutro")
    ax.barh(nomes, ppos, left=pneg + pneu, color=COR_POS, label="positivo")

    for i, t in enumerate(tot):
        ax.text(101, i, f"n={int(t)}", va="center", fontsize=8, color="#555")

    ax.set_xlim(0, 100)
    ax.set_xlabel("% das avaliacoes da funcionalidade")
    ax.set_title(f"Sentimento por funcionalidade - {config.APP_NOME}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False)
    fig.tight_layout()
    saida = FIG_DIR / f"matriz_sentimento_{config.APP_ID}.png"
    fig.savefig(saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[graficos] salvo: {saida}")


# ---------------------------------------------------------------------------
# 2) Comparativo QP2 (lexico x supervisionado)
# ---------------------------------------------------------------------------
def _metricas_qp2() -> pd.DataFrame | None:
    """Recalcula acuracia e F1-macro de cada abordagem sobre a amostra rotulada."""
    from src import preprocessamento, sentimentos, validacao
    try:
        df = validacao.carregar_rotulada()
    except FileNotFoundError:
        print("[graficos] sem amostra rotulada - pulei o comparativo da QP2.")
        return None

    df = df.copy()
    df["texto_limpo"] = df["texto"].map(preprocessamento.limpar)

    linhas = []
    # Baseline lexico (avaliado sobre TODA a amostra rotulada)
    pred_lex = sentimentos.aplicar_baseline(df)
    linhas.append({
        "abordagem": "Lexico",
        "acuracia": accuracy_score(df["sentimento"], pred_lex),
        "f1_macro": f1_score(df["sentimento"], pred_lex,
                             labels=config.SENT_CLASSES, average="macro",
                             zero_division=0),
    })

    # Supervisionados (metricas no subconjunto de teste)
    _, resultados = sentimentos.treinar_supervisionado(df, "sentimento")
    nome_curto = {"Naive Bayes": "Naive Bayes", "SVM (Linear)": "SVM",
                  "Regressao Logistica": "Reg. Log."}
    for nome, res in resultados.items():
        rel = res["relatorio"]
        linhas.append({
            "abordagem": nome_curto.get(nome, nome),
            "acuracia": rel["accuracy"],
            "f1_macro": rel["macro avg"]["f1-score"],
        })
    return pd.DataFrame(linhas)


def grafico_qp2() -> None:
    dados = _metricas_qp2()
    if dados is None:
        return

    x = range(len(dados))
    larg = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar([i - larg / 2 for i in x], dados["acuracia"], larg,
                label="Acuracia", color="#337ab7")
    b2 = ax.bar([i + larg / 2 for i in x], dados["f1_macro"], larg,
                label="F1-macro", color="#f0ad4e")

    for barras in (b1, b2):
        for b in barras:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(dados["abordagem"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("desempenho")
    ax.set_title(f"QP2: lexico x supervisionado - {config.APP_NOME}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    saida = FIG_DIR / f"qp2_comparativo_{config.APP_ID}.png"
    fig.savefig(saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[graficos] salvo: {saida}")


def main() -> None:
    grafico_matriz()
    grafico_qp2()
    print(f"\nFiguras em: {FIG_DIR}")


if __name__ == "__main__":
    main()
