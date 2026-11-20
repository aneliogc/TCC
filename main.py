"""
Orquestrador do pipeline do TCC.

Uso (a partir da raiz do projeto, com o venv ativo):

  py main.py coleta          # 1. coleta avaliacoes da Google Play
  py main.py preprocessa     # 2. pre-processamento textual
  py main.py amostra         # (opcional) gera amostra para rotulagem manual
  py main.py topicos         # 3. LDA - descoberta de funcionalidades
  py main.py sentimentos     # 4. analise de sentimentos (lexico + supervisionado)
  py main.py associa         # 5. matriz funcionalidade x sentimento
  py main.py valida          # 6. validacao vs. amostra rotulada (truth set)
  py main.py graficos        # gera figuras (PNG) dos resultados para o TCC
  py main.py exemplos 6      # mostra avaliacoes reais de um topico (ajuda a nomear)
  py main.py tudo            # roda 1->5 em sequencia

Ou execute um modulo isolado: py -m src.coleta
"""
import sys

from src import (amostragem, associacao, coleta, graficos, preprocessamento,
                 sentimentos, topicos, validacao)

ETAPAS = {
    "coleta": coleta.main,
    "preprocessa": preprocessamento.main,
    "amostra": amostragem.main,
    "topicos": topicos.main,
    "sentimentos": sentimentos.main,
    "associa": associacao.main,
    "valida": validacao.main,
    "graficos": graficos.main,
}

SEQUENCIA = ["coleta", "preprocessa", "topicos", "sentimentos", "associa"]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    comando = sys.argv[1]
    if comando == "exemplos":
        if len(sys.argv) < 3:
            print("Uso: py main.py exemplos <numero_do_topico>")
            return
        topicos.exemplos_topico(int(sys.argv[2]))
    elif comando == "tudo":
        for etapa in SEQUENCIA:
            print(f"\n{'='*70}\n>>> {etapa.upper()}\n{'='*70}")
            ETAPAS[etapa]()
    elif comando in ETAPAS:
        ETAPAS[comando]()
    else:
        print(f"Comando desconhecido: {comando}\n")
        print(__doc__)


if __name__ == "__main__":
    main()
