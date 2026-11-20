"""
Configuracoes centrais do pipeline do TCC.

Identificacao de Funcionalidades e Sentimentos em Avaliacoes de Aplicativos
usando Tecnicas de Mineracao de Dados.

Altere APP_ID / APP_NOME para trocar o aplicativo analisado. O ID do app e o
que aparece na URL da Google Play Store, no parametro id=... Por exemplo:
  https://play.google.com/store/apps/details?id=com.nu.production  -> Nubank
"""
from pathlib import Path

# Aplicativo alvo (Google Play Store)

APP_ID = "br.com.brainweb.ifood"   # iFood  (Nubank: com.nu.production)
APP_NOME = "iFood"
LANG = "pt"
COUNTRY = "br"

# Quantidade de avaliacoes a coletar
MAX_AVALIACOES = 8000


# Diretorios do projeto

RAIZ = Path(__file__).resolve().parent
DATA = RAIZ / "data"
DATA_RAW = DATA / "raw"            # avaliacoes cruas coletadas
DATA_PROC = DATA / "processed"    # textos pre-processados
DATA_RESULTS = DATA / "results"   # topicos, matrizes, metricas
RESOURCES = RAIZ / "resources"    # lexicos (OpLexicon, SentiLex-PT)

for _d in (DATA_RAW, DATA_PROC, DATA_RESULTS, RESOURCES):
    _d.mkdir(parents=True, exist_ok=True)

# Arquivos padrao
ARQ_RAW = DATA_RAW / f"avaliacoes_{APP_ID}.csv"
ARQ_PROC = DATA_PROC / f"avaliacoes_proc_{APP_ID}.csv"
ARQ_AMOSTRA_ROTULAR = DATA_PROC / f"amostra_para_rotular_{APP_ID}.csv"
# mapa radical(stem) -> palavra completa mais frequente, para exibir os topicos
# de forma legivel (o LDA continua usando os radicais internamente)
ARQ_STEM_MAP = DATA_PROC / f"stem_map_{APP_ID}.csv"

# Todos os arquivos abaixo sao chaveados por APP_ID para permitir analisar varios
# aplicativos sem colisao (cada app tem seus proprios topicos, rotulos e matriz).
ARQ_TOPICOS_DOC = DATA_PROC / f"avaliacoes_topicos_{APP_ID}.csv"
ARQ_ROTULADA = DATA_PROC / f"amostra_rotulada_{APP_ID}.csv"
ARQ_LDA_BUSCA = DATA_RESULTS / f"lda_busca_k_{APP_ID}.csv"
ARQ_LDA_TOPICOS = DATA_RESULTS / f"lda_topicos_{APP_ID}.csv"
ARQ_MATRIZ = DATA_RESULTS / f"matriz_func_sentimento_{APP_ID}.csv"
ARQ_ROTULOS = RESOURCES / f"rotulos_topicos_{APP_ID}.csv"
ARQ_ROTULOS_TEMPLATE = RESOURCES / f"rotulos_topicos_template_{APP_ID}.csv"


# Pre-processamento

MIN_TOKENS = 3          # descarta avaliacoes com menos tokens que isso
USAR_STEMMER = True     # RSLP (portugues). Se False, mantem lemas simples.

# Stopwords de dominio (alem das do NLTK). Termos genericos de avaliacoes
# que nao ajudam a distinguir funcionalidades.
STOPWORDS_DOMINIO = {
    "app", "aplicativo", "aplicativos", "nubank", "nu", "conta", "ifood",
    "muito", "bom", "otimo", "excelente", "pessimo", "ruim", "top",
    "parabens", "gostei", "adorei", "recomendo", "nota", "estrela",
    "estrelas", "gente", "voces", "obrigado", "obrigada", "favor",
    "fazer", "so", "ai", "pra", "pro", "vc", "vcs", "ta", "tá",
}

# Palavras genericas/avaliativas removidas APOS o stemming (o codigo stemiza
# cada uma para pegar todas as flexoes de uma vez). Objetivo: o LDA focar em
# FUNCIONALIDADES, nao em verbos genericos nem em palavras de sentimento (o
# sentimento e tratado numa etapa separada).
STOPWORDS_EXTRA = {
    # verbos e preenchedores genericos
    "ficar", "poder", "ter", "fazer", "querer", "deixar", "dar", "ir", "vir",
    "ver", "achar", "passar", "colocar", "pegar", "usar", "mexer",
    "vez", "dia", "hora", "ano", "mes", "sempre", "agora", "hoje", "ainda",
    "nunca", "coisa", "tudo", "todo", "outro", "mesmo", "algum", "nenhum",
    "nada", "pois", "assim", "entao", "cada", "qualquer", "aqui", "ali",
    "gente", "mim", "voce", "porque", "final", "algo",
    # avaliativas / de sentimento (nao sao funcionalidades)
    "melhor", "pior", "bom", "otimo", "ruim", "pessimo", "excelente",
    "maravilha", "maravilhoso", "amar", "adorar", "gostar", "gosto", "top",
    "perfeito", "horrivel", "odiar", "parabens", "recomendar", "legal",
    "amo", "boa", "ajudar", "ajuda", "super", "alto", "baixo",
}


# Modelagem de topicos (LDA)

# Numero de topicos a testar na busca pela melhor configuracao.
LDA_TOPICOS_GRID = [5, 8, 10, 12, 15, 20]
LDA_N_TOPICOS = 8           # escolhido por interpretabilidade (ver comparacao de k)
LDA_MAX_ITER = 25
LDA_RANDOM_STATE = 42
LDA_TOP_TERMOS = 12         # termos exibidos por topico
# Limiar de probabilidade para associar MULTIPLAS funcionalidades a uma mesma
# avaliacao (QP3 multi-rotulo). Alem do topico dominante, todo topico com
# probabilidade >= este valor tambem e atribuido a avaliacao. Usado so na
# validacao; a matriz principal continua usando o topico dominante.
LDA_MIN_PROB = 0.20

# Vetorizacao (Bag-of-Words para o LDA)
BOW_MIN_DF = 5              # termo deve aparecer em ao menos 5 documentos
BOW_MAX_DF = 0.5           # ignora termos em mais de 50% dos documentos
BOW_MAX_FEATURES = 3000


# Analise de sentimentos

# Mapeamento nota -> rotulo fraco (usado so como apoio/heuristica, NAO como
# verdade).
SENT_CLASSES = ["negativo", "neutro", "positivo"]

# Vetorizacao TF-IDF para classificadores supervisionados
TFIDF_MIN_DF = 3
TFIDF_NGRAM = (1, 2)
TEST_SIZE = 0.25
RANDOM_STATE = 42
