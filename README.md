# Análisis de Interacciones y Emociones en Antologías Narrativas

Herramienta de procesamiento de lenguaje natural (NLP) que analiza un PDF narrativo, detecta automáticamente los cuentos que lo componen, identifica personajes, clasifica las emociones presentes y genera grafos de interacción entre personajes — tanto estáticos (PNG) como interactivos (HTML).

Desarrollado y probado con *Cuentos Completos* de Franz Kafka (traducción de José Rafael Hernández Arias).

---

## Tabla de Contenidos

- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Uso](#uso)
- [Qué produce el script](#qué-produce-el-script)
- [Descripción de comandos](#descripción-de-comandos)
- [Modelos disponibles en Ollama](#modelos-disponibles-en-ollama)
- [Parámetros configurables](#parámetros-configurables)
- [Cómo funciona internamente](#cómo-funciona-internamente)
- [Decisiones de diseño](#decisiones-de-diseño)

---

## Requisitos

- Python 3.10 o superior
- Ollama instalado: https://ollama.com
- GPU recomendada — guía de modelos según VRAM:

| VRAM disponible | Modelo recomendado |
|---|---|
| 6 GB | `llama3.1:8b` o `mistral:7b` |
| 10 GB | `qwen2.5:14b` |
| 16 GB | `qwen2.5:14b` (óptimo) |
| 40 GB+ | `llama3.1:70b` |

---

## Instalación

### 1. Crear entorno virtual

```bash
python -m venv .venv
```

Crea un entorno aislado en `.venv`. Evita conflictos entre versiones de librerías de distintos proyectos.

### 2. Activar el entorno virtual

```bash
# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Redirige `python` y `pip` al entorno aislado. El prompt mostrará `(.venv)` cuando esté activo.

### 3. Instalar dependencias

```bash
pip install pdfplumber networkx matplotlib numpy requests pyvis
```

### 4. Configurar Ollama

```bash
# Descargar el modelo
ollama pull qwen2.5:14b

# Verificar que está disponible
ollama list
```

Ollama inicia automáticamente como servicio en Windows. En Mac/Linux puede ser necesario ejecutar `ollama serve` en una terminal separada.

---

## Estructura del proyecto

```
proyecto/
│
├── .venv/                              # Entorno virtual (no subir a git)
├── resources/
│   └── kafka_cuentos.pdf              # PDF de entrada
├── resultados/                         # Carpeta de salida (se crea automáticamente)
│   ├── cuento_01_21__la_condena.png   # Grafo estático por cuento
│   ├── cuento_01_21__la_condena.html  # Grafo interactivo por cuento
│   └── resumen_antologia.png          # Gráfico comparativo general
├── anthology_analysis.py              # Script principal
└── README.md
```

---

## Uso

### Analizar toda la antología
```bash
python anthology_analysis.py resources/kafka_cuentos.pdf resultados/
```

### Analizar un solo cuento
```bash
python anthology_analysis.py resources/kafka_cuentos.pdf resultados/ "LA CONDENA"
```

La búsqueda del título es parcial e insensible a mayúsculas — basta con escribir una parte del título. Si no encuentra coincidencia, el script lista todos los títulos disponibles.

### Descargar el PDF directamente desde URL
```python
import requests
from pathlib import Path

url = "https://web.seducoahuila.gob.mx/biblioweb/upload/13%20Cuentos%20Comp%20Kafka.pdf"
Path("resources/kafka_cuentos.pdf").write_bytes(requests.get(url, timeout=30).content)
```

---

## Qué produce el script

### Por cada cuento detectado
- **PNG** — grafo estático con layout `kamada_kawai` que minimiza cruces de aristas
- **HTML** — grafo interactivo con pyvis donde puedes arrastrar nodos, hacer zoom y ver detalles en hover

### Al finalizar todos los cuentos
- **`resumen_antologia.png`** — gráfico de barras con la distribución emocional comparada entre todos los cuentos

### En consola
Reporte con personajes identificados, emoción dominante y distribución completa por cuento.

---

## Descripción de comandos

### Entorno virtual

| Comando | Propósito |
|---|---|
| `python -m venv .venv` | Crea el entorno virtual aislado en `.venv` |
| `source .venv/bin/activate` | Activa el entorno en Mac/Linux |
| `.venv\Scripts\activate` | Activa el entorno en Windows |
| `deactivate` | Desactiva el entorno y restaura el PATH del sistema |
| `which python` (Mac/Linux) | Verifica que Python apunta al entorno activo |
| `where python` (Windows) | Equivalente Windows de `which python` |
| `python -m pip list` | Lista paquetes instalados en el entorno activo |

### Instalación de dependencias

| Comando | Propósito |
|---|---|
| `pip install pdfplumber` | Extracción de texto de PDFs conservando layout |
| `pip install networkx` | Creación y análisis de grafos de red |
| `pip install matplotlib` | Generación de gráficos e imágenes PNG |
| `pip install numpy` | Operaciones numéricas para el gráfico de resumen |
| `pip install requests` | Comunicación HTTP con la API local de Ollama |
| `pip install pyvis` | Generación de grafos interactivos en HTML |

### Comandos de Ollama

| Comando | Propósito |
|---|---|
| `ollama pull qwen2.5:14b` | Descarga el modelo (~9 GB). Solo necesario una vez |
| `ollama list` | Muestra los modelos descargados y disponibles localmente |
| `ollama run qwen2.5:14b "hola"` | Prueba el modelo directamente en terminal |
| `ollama serve` | Inicia el servidor de Ollama (automático en Windows) |
| `ollama ps` | Muestra modelos activos y si usan GPU o CPU |

### Diagnóstico y depuración

| Comando | Propósito |
|---|---|
| `nvidia-smi` | Muestra VRAM disponible y uso actual de la GPU |
| `print(repr(full_text[:2000]))` | Muestra caracteres ocultos del texto extraído del PDF para diagnosticar ruido |
| `print(repr(linea))` | Muestra una línea con todos sus caracteres visibles para depurar el regex de títulos |

### Ejecución del script

| Comando | Propósito |
|---|---|
| `python anthology_analysis.py archivo.pdf` | Analiza toda la antología con carpeta de salida por defecto |
| `python anthology_analysis.py archivo.pdf carpeta/` | Analiza toda la antología y guarda resultados en la carpeta indicada |
| `python anthology_analysis.py archivo.pdf carpeta/ "TITULO"` | Analiza solo el cuento cuyo título contenga el texto indicado |

---

## Modelos disponibles en Ollama

| Modelo | Tamaño | VRAM | Calidad JSON | Comando |
|---|---|---|---|---|
| `llama3.2` | 3B | ~3 GB | Regular | `ollama pull llama3.2` |
| `llama3.1:8b` | 8B | ~6 GB | Buena | `ollama pull llama3.1:8b` |
| `mistral:7b` | 7B | ~5 GB | Buena | `ollama pull mistral:7b` |
| `gemma3:9b` | 9B | ~7 GB | Buena | `ollama pull gemma3:9b` |
| `qwen2.5:14b` | 14B | ~10 GB | Muy buena ✅ | `ollama pull qwen2.5:14b` |
| `llama3.1:70b` | 70B | ~40 GB | Excelente | `ollama pull llama3.1:70b` |

Para cambiar el modelo activo, edita la variable al inicio del script:
```python
OLLAMA_MODEL = "qwen2.5:14b"
```

---

## Parámetros configurables

Todos al inicio de `anthology_analysis.py`:

```python
# Modelo de Ollama a utilizar
OLLAMA_MODEL = "qwen2.5:14b"

# URL del servidor Ollama (cambiar si corre en otro equipo o puerto)
OLLAMA_URL = "http://localhost:11434/api/generate"

# Límite de palabras enviadas al LLM para interacciones
# None = sin límite (recomendado para modelos 14B+)
MAX_WORDS_FOR_INTERACTIONS = None

# Dentro de la función ollama():
"num_predict": 2048,   # tokens máximos en la respuesta del LLM
"num_ctx":     32768,  # ventana de contexto (tokens de entrada)
```

---

## Cómo funciona internamente

### Pipeline completo

```
PDF
 │
 ▼
extract_text_from_pdf()      → Extrae texto con pdfplumber
 │
 ▼
clean_text()                 → Limpia ruido del PDF (números de página,
 │                             encabezados, caracteres especiales)
 ▼
join_split_titles()          → Une títulos que quedaron partidos en dos líneas
 │
 ▼
split_into_stories()         → Divide en cuentos usando TITLE_PATTERN (regex)
 │
 ├─ Por cada cuento:
 │   │
 │   ├── extract_characters()           → LLM identifica nombres propios
 │   ├── count_emotions()               → LLM clasifica distribución emocional
 │   ├── build_interaction_graph()      → LLM detecta interacciones entre personajes
 │   ├── plot_story_graph_static()      → Genera PNG con kamada_kawai_layout
 │   └── plot_story_graph_interactive() → Genera HTML interactivo con pyvis
 │
 └── plot_summary()                     → Gráfico comparativo de toda la antología
```

### Detección de títulos

El script detecta títulos con el patrón:
```
"83. JOSEFINA, LA CANTORA, O EL PUEBLO DE LOS RATONES"
 ↑   ↑
número  TODO EN MAYÚSCULAS
```

Si un título queda partido en dos líneas por el PDF, `join_split_titles()` las une antes de aplicar el regex.

### Comunicación con Ollama

Python se comunica con Ollama a través de su API REST local. Cada tarea (personajes, emociones, interacciones) es una llamada separada con un prompt estructurado que instruye al modelo a responder únicamente en JSON:

```python
requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5:14b",
    "prompt": "...",
    "options": {"temperature": 0.0}
})
```

### Grafo de interacciones

El LLM analiza el cuento completo y devuelve un JSON con los pares de personajes que interactúan, la emoción dominante de cada relación y el número aproximado de interacciones:

```json
[
  {"personaje_a": "Georg", "personaje_b": "El Padre", "emocion": "enojo", "interacciones": 5},
  {"personaje_a": "Georg", "personaje_b": "Frieda",   "emocion": "alegria", "interacciones": 2}
]
```

Cada arista del grafo hereda el color de la emoción y el grosor según el número de interacciones.

### Emociones detectadas

| Clave | Español | Color |
|---|---|---|
| `alegria` | Alegría | Amarillo |
| `tristeza` | Tristeza | Azul |
| `enojo` | Enojo | Rojo |
| `miedo` | Miedo | Violeta |
| `asco` | Asco | Verde oscuro |
| `sorpresa` | Sorpresa | Naranja |
| `neutro` | Neutro | Gris |

---

## Decisiones de diseño

**¿Por qué Ollama?**
Un LLM local puede leer el cuento completo de una vez y razonar sobre las relaciones entre personajes con todo el contexto disponible, sin necesidad de dividir el texto en fragmentos.

**¿Por qué `kamada_kawai_layout` en el PNG?**
El layout `spring_layout` (por defecto en networkx) produce grafos saturados cuando hay muchos nodos. `kamada_kawai` minimiza los cruces de aristas y distribuye los nodos de forma más equilibrada.

**¿Por qué pyvis para el HTML?**
pyvis genera un HTML autocontenido que se abre en cualquier navegador sin servidor. Permite reorganizar el grafo arrastrando nodos, hacer zoom y ver detalles en hover — algo imposible con una imagen estática.

**¿Por qué `temperature: 0.0`?**
Las tareas de extracción de entidades y generación de JSON estructurado requieren respuestas deterministas. Con temperatura 0 el modelo siempre elige el token más probable, lo que reduce variabilidad y mejora la consistencia del JSON generado.

**¿Por qué `MAX_WORDS_FOR_INTERACTIONS = None` con modelos grandes?**
Modelos pequeños (3B) se saturan con textos largos y devuelven JSON malformado. Con 14B+ el modelo maneja sin problema cuentos de 15k palabras, por lo que no hay razón para truncar y perder contexto narrativo.
