# DAPPIA — Avance 2: Prueba de Concepto RAG

## 📌 Descripción del Proyecto

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** que combina una base de datos vectorial local (**ChromaDB**) con el modelo de lenguaje **Gemini 2.5 Flash** de Google para responder preguntas técnicas de forma precisa, basándose únicamente en una base de conocimiento predefinida.

---

## 🔄 Flujo del Sistema RAG Implementado

```
┌─────────────────────────────────────────────────────────────────┐
│                      FLUJO RAG — DAPPIA                        │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────────┐                   │
│  │  BASE DE     │    │   VECTORIZACIÓN     │                   │
│  │ CONOCIMIENTO │───▶│   (ChromaDB +       │                   │
│  │  (Texto)     │    │  all-MiniLM-L6-v2)  │                   │
│  └──────────────┘    └─────────┬───────────┘                   │
│                                │  Embeddings almacenados        │
│                                ▼                               │
│  ┌──────────────┐    ┌─────────────────────┐                   │
│  │  PREGUNTA    │───▶│  BÚSQUEDA SEMÁNTICA │                   │
│  │  DEL USUARIO │    │  (top-2 chunks más  │                   │
│  └──────────────┘    │   relevantes)       │                   │
│                      └─────────┬───────────┘                   │
│                                │  Contexto recuperado           │
│                                ▼                               │
│                      ┌─────────────────────┐                   │
│                      │  ESTRUCTURACIÓN     │                   │
│                      │  DEL PROMPT         │                   │
│                      │  - Rol del asistente│                   │
│                      │  - Contexto RAG     │                   │
│                      │  - Instrucciones    │                   │
│                      │  - Formato salida   │                   │
│                      └─────────┬───────────┘                   │
│                                │                               │
│                                ▼                               │
│                      ┌─────────────────────┐                   │
│                      │   GEMINI 2.5 FLASH  │                   │
│                      │   (temp=0.1)        │                   │
│                      └─────────┬───────────┘                   │
│                                │                               │
│                                ▼                               │
│                      ┌─────────────────────┐                   │
│                      │  RESPUESTA          │                   │
│                      │  ESTRUCTURADA       │                   │
│                      │  - Respuesta        │                   │
│                      │  - Fuente           │                   │
│                      │  - Confianza        │                   │
│                      └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Estructura del Proyecto

```
gemini-api/
├── proyecto_dappia.py     # Código fuente principal
├── .env                   # Variables de entorno (API key)
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Este archivo
```

---

## ⚙️ Componentes del Sistema

### 1. Base de Conocimiento (`BASE_DE_CONOCIMIENTO`)

La base de conocimiento es un texto estructurado definido directamente en el código. Contiene 8 fragmentos (chunks) sobre los conceptos clave del sistema RAG:

| Chunk | Tema |
|-------|------|
| 0 | Definición del sistema RAG |
| 1 | Fases de funcionamiento de RAG |
| 2 | ChromaDB como base de datos vectorial |
| 3 | Definición de embeddings |
| 4 | Definición y propósito de los chunks |
| 5 | Modelo all-MiniLM-L6-v2 |
| 6 | Modelo Gemini 2.5 Flash |
| 7 | Temperatura en LLMs |

### 2. Vectorización (ChromaDB)

- **Motor:** ChromaDB en memoria (`chromadb.Client()`)
- **Modelo de embeddings:** `all-MiniLM-L6-v2` (descargado automáticamente la primera vez)
- **Dimensiones del vector:** 384
- **Colección:** `base_conocimiento_dappia`

### 3. Búsqueda Semántica

Para cada pregunta del usuario, ChromaDB realiza una búsqueda por similitud coseno en el espacio vectorial y retorna los **top-2 chunks** más relevantes. Estos se combinan como contexto para el LLM.

### 4. Estructuración del Prompt (System Configuration)

El prompt enviado al LLM está organizado en 4 secciones con etiquetas XML:

```xml
<rol>
  Define la identidad del asistente (DAPPIA) y su ámbito de conocimiento.
</rol>

<base_conocimiento>
  Contiene los chunks recuperados por ChromaDB para esa consulta específica.
</base_conocimiento>

<instrucciones>
  Reglas de comportamiento: usar solo el contexto, no inventar datos,
  qué hacer si la información no está disponible.
</instrucciones>

<formato_salida>
  Define la estructura obligatoria de la respuesta:
  - **Respuesta:** respuesta directa
  - **Fuente:** cita del contexto usado
  - **Confianza:** Alta / Media / Baja
</formato_salida>
```

### 5. Configuración del Modelo Gemini

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `model` | `gemini-2.5-flash` | Modelo rápido y eficiente |
| `temperature` | `0.1` | Respuestas determinísticas, fieles al contexto |
| `max_output_tokens` | `512` | Respuestas concisas y controladas |

### 6. Formato de Salida

Cada respuesta del asistente sigue este formato estructurado:

```
- **Respuesta:** [Respuesta directa, máximo 3 oraciones]
- **Fuente:** [Cita del fragmento de la base de conocimiento usado]
- **Confianza:** [Alta / Media / Baja]
```

---

## 🚀 Instalación y Ejecución

### Requisitos previos

- Python 3.10+
- Cuenta en Google AI Studio con API key de Gemini

### Pasos

```bash
# 1. Clonar/descargar el proyecto
cd "C:\PROYECTOS DESARROLLO CON IA\gemini-api"

# 2. Activar el entorno virtual
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\env\Scripts\Activate.ps1"

# 3. Instalar dependencias
pip install chromadb python-dotenv google-genai

# 4. Configurar API key en archivo .env
# Crear archivo .env con contenido:
# GEMINI_API_KEY=tu_api_key_aqui

# 5. Ejecutar el proyecto
python .\proyecto_dappia.py
```

---

## 📊 Diferencia entre RAG y LLM tradicional

| Aspecto | LLM Tradicional | Sistema RAG |
|---------|----------------|-------------|
| Fuente de conocimiento | Entrenamiento previo | Base de conocimiento controlada |
| Actualización | Requiere reentrenamiento | Actualizar colección en ChromaDB |
| Alucinaciones | Alta probabilidad | Reducidas (temperatura baja + contexto forzado) |
| Transparencia | Sin fuentes | Cita el fragmento usado |
| Dominio específico | Conocimiento general | Conocimiento especializado |

---

## 🔮 Próximos Pasos (Avance 3)

- [ ] Cargar documentos PDF reales como base de conocimiento
- [ ] Implementar ChromaDB persistente (en disco)
- [ ] Agregar interfaz de usuario con Streamlit
- [ ] Evaluar métricas de precisión del RAG (Recall@K, MRR)
- [ ] Implementar re-ranking de chunks recuperados

---

## DESARROLLO

Estudiante Salim Ferez Mendez - Programa de Ingeniería de Sistemas.
Proyecto DAPPIA — Desarrollo con IA  
Avance 2: Prueba de Concepto RAG  
Tecnologías: Python · ChromaDB · Google Gemini 2.5 Flash 
