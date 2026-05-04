"""
============================================================
  PROYECTO DAPPIA — AVANCE 2: PRUEBA DE CONCEPTO RAG
  Autor: Equipo Dappia
  Descripción: Sistema RAG con ChromaDB + Gemini 2.5 Flash
============================================================
"""

import os
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============================================================
# 1. CONFIGURACIÓN INICIAL
# ============================================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

print("=" * 60)
print("  AVANCE 2: PRUEBA DE CONCEPTO RAG")
print("=" * 60)

# ============================================================
# 2. BASE DE CONOCIMIENTO DEL ASISTENTE
#    Esta sección define los documentos que el asistente
#    usará EXCLUSIVAMENTE para responder. El LLM NO puede
#    usar conocimiento externo a esta base.
# ============================================================

BASE_DE_CONOCIMIENTO = """
El sistema RAG (Generación Aumentada por Recuperación) mejora las respuestas de un LLM al proporcionar contexto específico y actualizado en cada consulta.

RAG funciona en tres fases: primero recupera fragmentos relevantes de una base de datos de conocimiento, luego los inyecta en el prompt del LLM como contexto, y finalmente el LLM genera una respuesta basada únicamente en ese contexto.

ChromaDB es una base de datos vectorial de código abierto diseñada para aplicaciones de IA. Permite guardar, indexar y buscar texto localmente usando embeddings semánticos.

Los embeddings son representaciones numéricas (vectores) del significado de un texto. Textos con significados similares tendrán vectores cercanos en el espacio vectorial, lo que permite búsquedas semánticas eficientes.

Los chunks son fragmentos pequeños en los que se divide un documento grande. Dividir el texto en chunks mejora la precisión de la recuperación, ya que se puede recuperar solo el fragmento más relevante para cada consulta.

El modelo all-MiniLM-L6-v2 es el modelo de embeddings que usa ChromaDB por defecto. Es un modelo liviano y eficiente que convierte texto a vectores de 384 dimensiones.

Gemini 2.5 Flash es el modelo de lenguaje de Google usado en este proyecto para generar las respuestas finales a partir del contexto recuperado por RAG.

La temperatura en un LLM controla la creatividad de las respuestas. Un valor cercano a 0 produce respuestas más deterministas y fieles al contexto, lo que es ideal para sistemas RAG donde se busca precisión.
"""

# Dividimos la base de conocimiento en chunks (uno por línea no vacía)
chunks = [linea.strip() for linea in BASE_DE_CONOCIMIENTO.strip().split('\n') if linea.strip()]

print(f"\n📚 BASE DE CONOCIMIENTO:")
print(f"   Total de chunks indexados: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"   [{i}] {chunk[:80]}...")

# ============================================================
# 3. VECTORIZACIÓN Y BASE DE DATOS VECTORIAL (ChromaDB)
# ============================================================
print(f"\n{'='*60}")
print("  PASO 1: Inicializando ChromaDB y vectorizando chunks")
print("=" * 60)

chroma_client = chromadb.Client()
coleccion_rag = chroma_client.get_or_create_collection(
    name="base_conocimiento_dappia"
)

coleccion_rag.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"✅ ChromaDB inicializado en memoria.")
print(f"✅ {len(chunks)} chunks vectorizados con modelo all-MiniLM-L6-v2.")
print(f"✅ Colección creada: 'base_conocimiento_dappia'")

# ============================================================
# 4. CONSULTAS AL SISTEMA RAG
# ============================================================

preguntas = [
    "¿Qué es ChromaDB y para qué sirve?",
    "¿Cómo funciona el sistema RAG paso a paso?",
    "¿Qué es un embedding y qué modelo se usa aquí?"
]

for pregunta in preguntas:
    print(f"\n{'='*60}")
    print(f"  CONSULTA RAG")
    print(f"{'='*60}")
    print(f"\n❓ PREGUNTA DEL USUARIO:\n   {pregunta}")

    # ----------------------------------------------------------
    # PASO 2: Búsqueda vectorial semántica en ChromaDB
    # ----------------------------------------------------------
    resultados = coleccion_rag.query(
        query_texts=[pregunta],
        n_results=2  # Recuperamos los 2 chunks más relevantes
    )

    # Ensamblamos el contexto recuperado
    contextos_recuperados = resultados['documents'][0]
    contexto_combinado = "\n".join(
        [f"  - {ctx}" for ctx in contextos_recuperados]
    )

    print(f"\n🔍 CONTEXTO RECUPERADO POR RAG (top-2 chunks):")
    print(contexto_combinado)

    # ----------------------------------------------------------
    # PASO 3: ESTRUCTURACIÓN DEL PROMPT
    # ----------------------------------------------------------
    # El prompt tiene 3 secciones claramente definidas:
    #   A) Rol del asistente
    #   B) Base de conocimiento inyectada (contexto RAG)
    #   C) Instrucciones de comportamiento y restricciones

    SYSTEM_CONFIGURATION = f"""
<rol>
  Eres DAPPIA, un asistente experto en inteligencia artificial y sistemas RAG.
  Respondes preguntas técnicas de manera clara, precisa y estructurada.
  Tu conocimiento está limitado exclusivamente a la base de conocimiento proporcionada.
</rol>

<base_conocimiento>
{contexto_combinado}
</base_conocimiento>

<instrucciones>
  1. Responde ÚNICAMENTE usando la información dentro de <base_conocimiento>.
  2. Si la respuesta no está en la base de conocimiento, indica: "Esta información no está en mi base de conocimiento actual."
  3. No inventes datos, modelos, porcentajes ni afirmaciones no presentes en el contexto.
  4. Sé conciso pero completo. Máximo 3 oraciones para la respuesta.
</instrucciones>

<formato_salida>
  Estructura SIEMPRE tu respuesta exactamente así:
  - **Respuesta:** [Tu respuesta directa a la pregunta, máximo 3 oraciones]
  - **Fuente:** [Cita textual del fragmento de la base de conocimiento que usaste]
  - **Confianza:** [Alta / Media / Baja — según qué tan directamente el contexto responde la pregunta]
</formato_salida>
"""

    # Configuration del modelo Gemini
    CONFIGURACION_MODELO = types.GenerateContentConfig(
        system_instruction=SYSTEM_CONFIGURATION,
        temperature=0.1,   # Temperatura baja = respuestas fieles al contexto
        max_output_tokens=512
    )

    print(f"\n⚙️  CONFIGURACIÓN DEL SISTEMA:")
    print(f"   Modelo       : gemini-2.5-flash")
    print(f"   Temperature  : 0.1 (modo preciso/determinístico)")
    print(f"   Max tokens   : 512")
    print(f"   Chunks usados: {len(contextos_recuperados)}")

    # ----------------------------------------------------------
    # PASO 4: Generación de respuesta con el LLM
    # ----------------------------------------------------------
    print(f"\n🤖 RESPUESTA DEL ASISTENTE DAPPIA:")
    print("-" * 60)

    respuesta_llm = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=pregunta,
        config=CONFIGURACION_MODELO
    )

    print(respuesta_llm.text)
    print("-" * 60)

print(f"\n{'='*60}")
print("  FIN DE LA PRUEBA DE CONCEPTO RAG — AVANCE 2")
print("=" * 60)
