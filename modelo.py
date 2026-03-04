"""
Análisis de Interacciones y Emociones en Antología Narrativa
============================================================
Requiere Ollama corriendo localmente: https://ollama.com
    ollama pull llama3.2
    ollama serve

Instalación de dependencias Python:
    pip install pdfplumber networkx matplotlib numpy requests pyvis

Uso:
    python anthology_analysis.py mi_antologia.pdf
    python anthology_analysis.py mi_antologia.pdf carpeta_salida/
    python anthology_analysis.py mi_antologia.pdf carpeta_salida/ "LA CONDENA"
"""

import sys
import re
import json
import warnings
import requests
import traceback
from collections import defaultdict
from pathlib import Path

import pdfplumber
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pyvis.network import Network

warnings.filterwarnings("ignore")

# ── Configuración de Ollama ───────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"  
# Máximo de palabras que se envían al LLM para el análisis de interacciones.
# llama3.2 (3B) se satura con textos muy largos. Subir si usas un modelo mayor.
MAX_WORDS_FOR_INTERACTIONS = None  # sin limite para modelos grandes

def ollama(prompt: str) -> str:
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
            "num_ctx":     128000,
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] No se puede conectar a Ollama.")
        print("  Asegurate de que Ollama esta corriendo: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR Ollama] {e}")
        return ""

def ollama_json(prompt: str) -> dict | list | None:
    response = ollama(prompt)
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", response)
    if match:
        response = match.group(1).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    print(f"      [WARN] No se pudo parsear JSON. Respuesta cruda:\n{response[:400]}")
    return None

# ── Paleta de emociones ───────────────────────────────────────────────────────
EMOTION_COLORS = {
    "alegria":  "#f1c40f",
    "tristeza": "#3498db",
    "enojo":    "#e74c3c",
    "miedo":    "#9b59b6",
    "asco":     "#27ae60",
    "sorpresa": "#e67e22",
    "neutro":   "#95a5a6",
    "sarcasmo": "#f1c40f",
}
EMOTIONS_VALID = set(EMOTION_COLORS.keys())

def normalize_emotion(raw: str) -> str:
    clean = raw.lower().strip()
    clean = clean.replace("á","a").replace("é","e") \
                 .replace("í","i").replace("ó","o").replace("ú","u")
    return clean if clean in EMOTIONS_VALID else "neutro"

# ── Patrón de títulos ─────────────────────────────────────────────────────────
TITLE_PATTERN = re.compile(
    r"^\s*(\d+\.\s+[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\[\]\d,]+)\s*$",
    re.MULTILINE
)

# ── Limpieza de texto ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Franz Kafka.*?completos", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Titivillus.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ePub.*", "", text, flags=re.IGNORECASE)
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00C0-\u024F\u2000-\u206F]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
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

# ── LLM: extraer personajes ───────────────────────────────────────────────────
def extract_characters(text: str) -> set[str]:
    # Para extracción de personajes usamos solo las primeras 3000 palabras
    truncated = text
    prompt = f"""Eres un asistente literario. Lee el siguiente cuento de un cuento en español
e identifica ÚNICAMENTE los nombres propios de personajes humanos o animales con nombre.

Reglas:
- Pueden incluirse personajes como "El amigo de" "El padre de" etc.
- Pueden incluirse nombres de tipo abreviatura "A" "F" etc.
- No incluyas pronombres, títulos sin nombre (Tu, ti, etc.)
- En caso de ser un monologo, el personaje que narra debería ser considerado
- Si un personaje aparece con nombre y apellido, incluye ambos como una sola entrada
- Responde SOLO con un array JSON de strings, sin texto adicional

Fragmento:
\"\"\"{truncated}\"\"\"

Respuesta (solo JSON):"""

    result = ollama_json(prompt)
    if isinstance(result, list):
        names = {n.strip().title() for n in result
                 if isinstance(n, str) and len(n.strip()) > 2}
        return deduplicate_characters(names)
    return set()

# ── Deduplicar variantes del mismo personaje ──────────────────────────────────
def deduplicate_characters(names: set[str]) -> set[str]:
    sorted_names = sorted(names, key=len, reverse=True)
    final = []
    for name in sorted_names:
        if not any(name in longer for longer in final):
            final.append(name)
    return set(final)

# ── LLM: contar emociones ─────────────────────────────────────────────────────
def count_emotions(text: str) -> dict:
    truncated = text
    prompt = f"""Analiza las emociones presentes en el siguiente cuento de un cuento en español.

Devuelve un JSON con cuántas veces aproximadamente domina cada emoción
en los distintos momentos y escenas del texto.

Formato exacto (sin texto adicional):
{{"alegria": 0, "tristeza": 0, "enojo": 0, "miedo": 0, "asco": 0, "sorpresa": 0, "neutro": 0, "sarcasmo": 0}}

Fragmento:
\"\"\"{truncated}\"\"\"

Respuesta (solo JSON):"""

    result = ollama_json(prompt)
    if isinstance(result, dict):
        return {normalize_emotion(k): int(v)
                for k, v in result.items()
                if normalize_emotion(k) in EMOTIONS_VALID}
    return {"neutro": 1}

# ── LLM: construir grafo de interacciones ────────────────────────────────────
def build_interaction_graph(story_text: str, characters: set[str]) -> nx.Graph:
    chars_list = ", ".join(sorted(characters))

    # Truncar el texto para que el modelo no se sature
    words     = story_text.split()
    if MAX_WORDS_FOR_INTERACTIONS and len(words) > MAX_WORDS_FOR_INTERACTIONS:
        truncated = " ".join(words[:MAX_WORDS_FOR_INTERACTIONS])
        print(f"      [INFO] Texto truncado a {MAX_WORDS_FOR_INTERACTIONS} palabras para el LLM")
    else:
        truncated = story_text

    prompt = f"""Analiza el siguiente fragmento de un cuento en español y describe las interacciones
directas entre estos personajes: {chars_list}

Para cada par de personajes que interactúe de forma directa (se hablen,
se vean, se mencionen mutuamente), indica:
- La emoción dominante de esa relación
- El número aproximado de veces que interactúan

Emociones válidas: alegria, tristeza, enojo, miedo, asco, sorpresa, neutro, sarcasmo

Responde SOLO con un array JSON con este formato exacto, sin texto adicional:
[
  {{"personaje_a": "Nombre1", "personaje_b": "Nombre2", "emocion": "enojo", "interacciones": 3}}
]

Si no hay interacciones directas entre ningún par, responde: []

Fragmento:
\"\"\"{truncated}\"\"\"

Respuesta (solo JSON):"""

    G      = nx.Graph()
    result = ollama_json(prompt)

    print(f"      [DEBUG] Tipo respuesta: {type(result).__name__} | Valor: {str(result)[:200]}")

    if isinstance(result, list):
        print(f"      [DEBUG] {len(result)} interaccion(es) recibida(s)")
        for item in result:
            if not isinstance(item, dict):
                continue
            a   = item.get("personaje_a", "").strip().title()
            b   = item.get("personaje_b", "").strip().title()
            emo = normalize_emotion(item.get("emocion", "neutro"))
            cnt = int(item.get("interacciones", 1))
            if a and b and a != b:
                G.add_edge(a, b,
                           emotion=emo,
                           count=cnt,
                           color=EMOTION_COLORS.get(emo, "#aaaaaa"))
    else:
        print(f"      [DEBUG] El LLM no devolvio una lista valida")

    if len(G.nodes) > 0:
        centrality = nx.degree_centrality(G)
        nx.set_node_attributes(G, centrality, "centrality")

    print(f"      [DEBUG] Grafo final — Nodos: {len(G.nodes)} | Aristas: {len(G.edges)}")
    return G

# ── Grafo estático PNG ────────────────────────────────────────────────────────
def plot_story_graph_static(G: nx.Graph, title: str, output_path: str):
    if len(G.nodes) == 0:
        print("      [sin nodos, omitiendo PNG]")
        return

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    pos          = nx.kamada_kawai_layout(G)
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

    edge_labels = {(u, v): G[u][v]["emotion"].capitalize() for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_color="white", font_size=7,
                                 bbox=dict(boxstyle="round,pad=0.15",
                                           fc="#161b22", alpha=0.75))

    emotions_in_graph = set(G[u][v]["emotion"] for u, v in G.edges())
    legend = [mpatches.Patch(color=EMOTION_COLORS[e], label=e.capitalize())
              for e in emotions_in_graph if e in EMOTION_COLORS]
    ax.legend(handles=legend, loc="upper left", facecolor="#161b22",
              labelcolor="white", fontsize=9, framealpha=0.85)

    ax.set_title(title, color="white", fontsize=13, pad=10, wrap=True)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"      PNG guardado: {output_path}")

# ── Grafo interactivo HTML (pyvis) ────────────────────────────────────────────
def plot_story_graph_interactive(G: nx.Graph, title: str, output_path: str):
    """Siempre genera el HTML, con o sin nodos."""
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#0d1117",
        font_color="white",
        heading=title,
    )

    if len(G.nodes) == 0:
        # Grafo vacío — agregar nodo informativo
        net.add_node(
            "Sin interacciones detectadas",
            size=20,
            color="#95a5a6",
            font={"size": 14, "color": "white"},
        )
    else:
        for node in G.nodes:
            centrality = G.nodes[node].get("centrality", 0.1)
            net.add_node(
                node,
                label=node,
                size=15 + 40 * centrality,
                color="#1f6feb",
                font={"size": 14, "color": "white"},
                title=f"{node} — centralidad: {centrality:.2f}",
            )

        for u, v, data in G.edges(data=True):
            emo   = data.get("emotion", "neutro")
            count = data.get("count", 1)
            net.add_edge(
                u, v,
                color=EMOTION_COLORS.get(emo, "#aaaaaa"),
                width=1 + count * 0.8,
                label=emo.capitalize(),
                title=f"{emo.capitalize()} — {count} interaccion(es)",
                font={"size": 11, "color": "white", "align": "middle"},
            )

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.3,
          "springLength": 220,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      },
      "edges": {
        "smooth": { "type": "dynamic" }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true
      }
    }
    """)

    try:
        html_content = net.generate_html()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        size = Path(output_path).stat().st_size
        print(f"      HTML guardado: {output_path} ({size} bytes)")
    except Exception:
        print("      [ERROR guardando HTML]")
        traceback.print_exc()

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
               label=emo.capitalize(), color=EMOTION_COLORS[emo], alpha=0.85)

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels(titles, rotation=30, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Frecuencia de emocion", color="white")
    ax.set_title("Emociones por Cuento - Resumen de la Antologia",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()

# ── Pipeline principal ────────────────────────────────────────────────────────
def run(pdf_path: str, output_dir: str = ".", story_filter: str = None):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print(f"[0/3] Verificando conexion con Ollama ({OLLAMA_MODEL})...")
    test = ollama("Responde solo con la palabra: ok")
    if not test:
        print("[ERROR] Ollama no responde. Ejecuta: ollama serve")
        sys.exit(1)
    print("      Conexion exitosa.\n")

    print(f"[1/3] Leyendo PDF: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    print("      Dividiendo en cuentos...")
    stories = split_into_stories(raw_text)
    print(f"      {len(stories)} cuento(s) detectado(s).")

    if story_filter:
        matches = [s for s in stories
                   if story_filter.lower() in s["title"].lower()]
        if not matches:
            print(f"\n[ERROR] No se encontro ningun cuento con '{story_filter}'.")
            print("Titulos disponibles:")
            for s in stories:
                print(f"  - {s['title']}")
            sys.exit(1)
        stories = matches
        print(f"      Filtrando por '{story_filter}' -> {len(stories)} cuento(s).\n")
    else:
        print()

    all_stories_data = []

    for idx, story in enumerate(stories, 1):
        title = story["title"]
        text  = story["text"]

        print(f"[Cuento {idx}/{len(stories)}] {title}")
        print(f"      {len(text.split())} palabras")

        print("      -> Extrayendo personajes...")
        characters = extract_characters(text)
        print(f"      -> {len(characters)} personaje(s): {', '.join(sorted(characters)) or 'ninguno'}")

        print("      -> Analizando emociones...")
        emotion_counts = count_emotions(text)

        base_path   = str(out / f"cuento_{idx:02d}_{re.sub(r'[^a-z0-9]', '_', title.lower()[:30])}")
        static_path = base_path + ".png"
        html_path   = base_path + ".html"

        if len(characters) < 2:
            print("      -> Sin suficientes personajes para grafo de interacciones.")
            # Generar HTML vacío igualmente
            G = nx.Graph()
        else:
            print("      -> Construyendo grafo de interacciones...")
            G = build_interaction_graph(text, characters)
            plot_story_graph_static(G, f"Cuento {idx}: {title}", static_path)

        # Siempre generar HTML
        plot_story_graph_interactive(G, f"Cuento {idx}: {title}", html_path)

        story_entry = {
            "title":         title,
            "characters":    list(characters),
            "emotion_counts": emotion_counts,
            "graph_html":    html_path,
        }
        if len(G.nodes) > 0:
            story_entry["graph_png"] = static_path

        all_stories_data.append(story_entry)
        print()

    if len(all_stories_data) > 1:
        summary_path = str(out / "resumen_antologia.png")
        print("[2/3] Generando resumen general...")
        plot_summary(all_stories_data, summary_path)
        print(f"      Guardado en: {summary_path}")

    print("\n" + "=" * 65)
    print("REPORTE FINAL")
    print("=" * 65)
    for d in all_stories_data:
        print(f"\nCuento: {d['title']}")
        chars_str = ', '.join(d['characters']) if d['characters'] else 'ninguno (monologo/narracion)'
        print(f"  Personajes:        {chars_str}")
        print(f"  Grafo interactivo: {d['graph_html']}")
        if d["emotion_counts"]:
            dominant = max(d["emotion_counts"], key=d["emotion_counts"].get)
            print(f"  Emocion dominante: {dominant.capitalize()}")
            for emo, cnt in sorted(d["emotion_counts"].items(), key=lambda x: -x[1]):
                print(f"    {emo.capitalize():10s} {'#' * cnt} ({cnt})")

    print("\nListo.")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:     python anthology_analysis.py ruta/antologia.pdf [carpeta] [titulo]")
        print()
        print("Ejemplos:")
        print("  Toda la antologia:")
        print("    python anthology_analysis.py resources/kafka.pdf resultados/")
        print("  Un solo cuento:")
        print('    python anthology_analysis.py resources/kafka.pdf resultados/ "LA CONDENA"')
        print()
        print(f"Modelo activo: {OLLAMA_MODEL}")
        print("Para cambiarlo edita la variable OLLAMA_MODEL al inicio del script.")
        sys.exit(1)

    pdf_file     = sys.argv[1]
    output_dir   = sys.argv[2] if len(sys.argv) > 2 else "resultados_antologia"
    story_filter = sys.argv[3] if len(sys.argv) > 3 else None
    run(pdf_file, output_dir, story_filter)