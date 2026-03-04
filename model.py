"""
Análisis de Interacciones y Emociones en Antología Narrativa
============================================================
Instalación:
    pip install transformers torch networkx matplotlib pdfplumber pysentimiento

Uso:
    python anthology_analysis.py mi_antologia.pdf
    python anthology_analysis.py mi_antologia.pdf carpeta_salida/
"""

import sys
import re
import warnings
from collections import defaultdict
from pathlib import Path

import pdfplumber
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ── Modelos ───────────────────────────────────────────────────────────────────
from transformers import pipeline
from pysentimiento import create_analyzer

print("[1/3] Cargando modelos...")
ner = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-ner",
    aggregation_strategy="simple",
    device=0
)
emotion_analyzer = create_analyzer(task="emotion", lang="es")
print("      Modelos listos.\n")

# ── Paleta de emociones ───────────────────────────────────────────────────────
EMOTION_COLORS = {
    "joy":      "#f1c40f",
    "sadness":  "#3498db",
    "anger":    "#e74c3c",
    "fear":     "#9b59b6",
    "disgust":  "#27ae60",
    "surprise": "#e67e22",
    "others":   "#95a5a6",
}
EMOTION_ES = {
    "joy":      "Alegría",
    "sadness":  "Tristeza",
    "anger":    "Enojo",
    "fear":     "Miedo",
    "disgust":  "Asco",
    "surprise": "Sorpresa",
    "others":   "Neutro",
}

# Palabras que NER confunde con personas
FALSOS_POSITIVOS = {
    "Éste", "Ésta", "Este", "Esta", "Uno", "Una",
    "Aquel", "Aquella", "Alguien", "Nadie", "Ii",
    "Pero", "Cuando", "Como", "Porque", "Aunque",
}

# ── Patrón de títulos ─────────────────────────────────────────────────────────
TITLE_PATTERN = re.compile(
    r"^\s*(\d+\.\s+[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\[\]\d,]+)\s*$",
    re.MULTILINE
)

# ── Limpieza de texto ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    # Reunir palabras cortadas por guion al final de línea
    text = re.sub(r"-\n", "", text)
    # Eliminar marcadores de notas al pie: [1], [83]
    text = re.sub(r"\[\d+\]", "", text)
    # Eliminar números de página solos en una línea
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Eliminar encabezados repetidos del ePub
    text = re.sub(r"Franz Kafka.*?completos", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Titivillus.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ePub.*", "", text, flags=re.IGNORECASE)
    # Normalizar comillas tipográficas
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Normalizar guiones largos
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    # Eliminar caracteres de control y raros
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00C0-\u024F\u2000-\u206F]", "", text)
    # Normalizar espacios múltiples
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Normalizar líneas vacías múltiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ── Extracción de texto PDF ───────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return clean_text(text)

# ── Unir títulos partidos en dos líneas ──────────────────────────────────────
def join_split_titles(text: str) -> str:
    lines = text.split("\n")
    joined = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\d+\.\s+[A-ZÁÉÍÓÚÑ]", line):
            while (
                i + 1 < len(lines)
                and re.match(r"^[A-ZÁÉÍÓÚÑ\s\[\]\d,]+$", lines[i + 1].strip())
                and not re.match(r"^\s*\d+\.", lines[i + 1])
                and lines[i + 1].strip() != ""
            ):
                line = line.rstrip() + " " + lines[i + 1].strip()
                i += 1
        joined.append(line)
        i += 1
    return "\n".join(joined)

# ── División en cuentos ───────────────────────────────────────────────────────
def split_into_stories(text: str) -> list[dict]:
    text  = join_split_titles(text)
    parts = TITLE_PATTERN.split(text)
    stories = []
    for i in range(1, len(parts), 2):
        titulo = parts[i].strip()
        cuerpo = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if len(cuerpo.split()) >= 100:
            stories.append({"title": titulo, "text": cuerpo})
    return stories

# ── NER: limpieza de nombre extraído ─────────────────────────────────────────
def clean_entity_name(name: str) -> str | None:
    # Descartar sub-tokens BERT que escaparon la agregación
    if name.startswith("##"):
        return None
    # Eliminar puntuación pegada al nombre
    name = re.sub(r"[^\w\sáéíóúÁÉÍÓÚñÑüÜ]", "", name).strip()
    # Descartar si quedó muy corto (fragmento de token)
    if len(name) < 3:
        return None
    # Descartar si empieza con minúscula (no es nombre propio)
    if name[0].islower():
        return None
    # Descartar pronombres y palabras que NER confunde con personas
    if name in FALSOS_POSITIVOS:
        return None
    return name.title()

# ── NER: deduplicar variantes del mismo personaje ────────────────────────────
def deduplicate_characters(names: set[str]) -> set[str]:
    """
    Si 'Gregor' y 'Gregor Samsa' coexisten, conserva solo 'Gregor Samsa'.
    Ordena de más largo a más corto y descarta nombres contenidos en otros.
    """
    sorted_names = sorted(names, key=len, reverse=True)
    final = []
    for name in sorted_names:
        if not any(name in longer for longer in final):
            final.append(name)
    return set(final)

# ── NER: extraer personajes ───────────────────────────────────────────────────
def extract_characters(text: str) -> set[str]:
    words  = text.split()
    chunks = [" ".join(words[i:i + 200]) for i in range(0, len(words), 180)]
    raw_names = set()

    for chunk in chunks:
        try:
            entities = ner(chunk)
            for ent in entities:
                if ent["entity_group"] == "PER" and ent["score"] > 0.80:
                    cleaned = clean_entity_name(ent["word"])
                    if cleaned:
                        raw_names.add(cleaned)
        except Exception:
            pass

    return deduplicate_characters(raw_names)

# ── Análisis de emociones ─────────────────────────────────────────────────────
def analyze_emotion(text: str) -> dict:
    try:
        result   = emotion_analyzer.predict(text[:512])
        dominant = result.output
        score    = result.probas.get(dominant, 0.0)
        return {"emotion": dominant, "score": float(score), "all": dict(result.probas)}
    except Exception:
        return {"emotion": "others", "score": 0.5, "all": {}}

# ── Conteo de emociones por texto ─────────────────────────────────────────────
def count_emotions(text: str, max_sentences: int = 80) -> dict:
    counts = defaultdict(int)
    for sent in re.split(r"[.!?]", text)[:max_sentences]:
        sent = sent.strip()
        if len(sent) > 15:
            counts[analyze_emotion(sent)["emotion"]] += 1
    return dict(counts)

# ── Construcción del grafo de interacciones ───────────────────────────────────
def build_interaction_graph(story_text: str, characters: set[str]) -> nx.Graph:
    sentences     = re.split(r"[.!?]", story_text)
    G             = nx.Graph()
    edge_emotions = defaultdict(list)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        present = [c for c in characters if c.lower() in sent.lower()]
        if len(present) < 2:
            continue
        emo = analyze_emotion(sent)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair = tuple(sorted([present[i], present[j]]))
                edge_emotions[pair].append(emo["emotion"])

    for pair, emotions in edge_emotions.items():
        dominant = max(set(emotions), key=emotions.count)
        G.add_edge(
            pair[0], pair[1],
            emotion=dominant,
            count=len(emotions),
            color=EMOTION_COLORS.get(dominant, "#aaaaaa"),
        )

    if len(G.nodes) > 0:
        centrality = nx.degree_centrality(G)
        nx.set_node_attributes(G, centrality, "centrality")

    return G

# ── Grafo por cuento ──────────────────────────────────────────────────────────
def plot_story_graph(G: nx.Graph, title: str, output_path: str):
    if len(G.nodes) == 0:
        print("      [sin personajes detectados, omitiendo gráfico]")
        return

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    pos          = nx.spring_layout(G, seed=42, k=2.5)
    centralities = [G.nodes[n].get("centrality", 0.1) for n in G.nodes]
    node_sizes   = [800 + 3000 * c for c in centralities]
    edge_colors  = [G[u][v].get("color", "#aaaaaa") for u, v in G.edges()]
    edge_widths  = [1.5 + G[u][v].get("count", 1) * 0.8 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color="#1f6feb", alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white",
                            font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_colors, alpha=0.78)

    edge_labels = {(u, v): EMOTION_ES.get(G[u][v]["emotion"], "?")
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_color="white", font_size=7,
                                 bbox=dict(boxstyle="round,pad=0.15",
                                           fc="#161b22", alpha=0.75))

    emotions_in_graph = set(G[u][v]["emotion"] for u, v in G.edges())
    legend = [mpatches.Patch(color=EMOTION_COLORS[e], label=EMOTION_ES.get(e, e))
              for e in emotions_in_graph]
    ax.legend(handles=legend, loc="upper left", facecolor="#161b22",
              labelcolor="white", fontsize=9, framealpha=0.85)

    ax.set_title(title, color="white", fontsize=13, pad=10, wrap=True)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()

# ── Resumen general ───────────────────────────────────────────────────────────
def plot_summary(all_stories_data: list[dict], output_path: str):
    titles     = [d["title"][:25] for d in all_stories_data]
    emo_counts = [d["emotion_counts"] for d in all_stories_data]
    emotions   = list(EMOTION_COLORS.keys())
    x, width   = np.arange(len(titles)), 0.1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for i, emo in enumerate(emotions):
        vals = [ec.get(emo, 0) for ec in emo_counts]
        ax.bar(x + i * width, vals, width,
               label=EMOTION_ES[emo], color=EMOTION_COLORS[emo], alpha=0.85)

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels(titles, rotation=30, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Frecuencia de emoción", color="white")
    ax.set_title("Emociones por Cuento — Resumen de la Antología",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()

# ── Pipeline principal ────────────────────────────────────────────────────────
def run(pdf_path: str, output_dir: str = "."):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print(f"[2/3] Leyendo PDF: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    print("      Dividiendo en cuentos...")
    stories = split_into_stories(raw_text)
    print(f"      {len(stories)} cuento(s) detectado(s).\n")

    all_stories_data = []

    for idx, story in enumerate(stories, 1):
        title = story["title"]
        text  = story["text"]

        print(f"[Cuento {idx}/{len(stories)}] {title}")
        print(f"      {len(text.split())} palabras")

        print("      → Extrayendo personajes...")
        characters = extract_characters(text)
        print(f"      → {len(characters)} personaje(s): {', '.join(sorted(characters)) or 'ninguno'}")

        # Siempre analizar emociones, haya o no personajes
        print("      → Analizando emociones...")
        emotion_counts = count_emotions(text)

        if len(characters) < 2:
            print("      → Sin interacciones entre personajes, registrando solo emociones narrativas.\n")
            all_stories_data.append({
                "title": title,
                "characters": list(characters),
                "emotion_counts": emotion_counts,
            })
            continue

        print("      → Construyendo grafo de interacciones...")
        G = build_interaction_graph(text, characters)

        graph_path = str(
            out / f"cuento_{idx:02d}_{re.sub(r'[^a-z0-9]', '_', title.lower()[:30])}.png"
        )
        print(f"      → Guardando grafo: {graph_path}")
        plot_story_graph(G, f"Cuento {idx}: {title}", graph_path)

        all_stories_data.append({
            "title": title,
            "characters": list(characters),
            "emotion_counts": emotion_counts,
            "graph": graph_path,
        })
        print()

    # Resumen general
    if len(all_stories_data) > 1:
        summary_path = str(out / "resumen_antologia.png")
        print("[3/3] Generando resumen general...")
        plot_summary(all_stories_data, summary_path)
        print(f"      Guardado en: {summary_path}")

    # Reporte en consola
    print("\n" + "=" * 65)
    print("REPORTE FINAL")
    print("=" * 65)
    for d in all_stories_data:
        print(f"\nCuento: {d['title']}")
        chars_str = ', '.join(d['characters']) if d['characters'] else 'ninguno (monólogo/narración)'
        print(f"  Personajes: {chars_str}")
        if d["emotion_counts"]:
            dominant = max(d["emotion_counts"], key=d["emotion_counts"].get)
            print(f"  Emocion dominante: {EMOTION_ES.get(dominant, dominant)}")
            for emo, cnt in sorted(d["emotion_counts"].items(), key=lambda x: -x[1]):
                print(f"    {EMOTION_ES.get(emo, '?'):10s} {'#' * cnt} ({cnt})")

    print("\nListo.")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:     python anthology_analysis.py ruta/antologia.pdf")
        print("Ejemplo: python anthology_analysis.py resources/kafka.pdf resultados/")
        sys.exit(1)

    pdf_file   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "resultados_antologia"
    run(pdf_file, output_dir)