from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FILE = "notes.json"


# ---------- keyword extraction ----------

def extract_keywords(text: str):

    try:

        prompt = f"""
Extract 3 to 5 important keywords from this note.
Return only comma separated words.

Text:
{text}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        result = response.choices[0].message.content

        keywords = [
            k.strip().lower()
            for k in result.split(",")
        ]

        return keywords[:5]

    except Exception as e:

        print("keyword error:", e)

        words = text.lower().split()
        return list(set(words))[:5]


# ---------- embeddings ----------

def get_embedding(text: str):

    vec = model.encode(text)

    return vec.tolist()


def cosine_sim(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ---------- storage ----------

def read_notes():
    if not os.path.exists(FILE):
        return []

    with open(FILE, "r") as f:
        return json.load(f)


def write_notes(notes):
    with open(FILE, "w") as f:
        json.dump(notes, f, indent=2)


def create_note(text, keywords=None):

    if keywords is None:
        keywords = []

    embedding = get_embedding(text)

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "keywords": keywords,
        "embedding": embedding
    }


# ---------- graph ----------

def build_graph(notes):

    nodes = []
    edges = []

    for note in notes:
        nodes.append({
            "id": note["id"],
            "label": note["text"][:25]
        })

    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):

            e1 = notes[i].get("embedding")
            e2 = notes[j].get("embedding")

            if not e1 or not e2:
                continue

            sim = cosine_sim(e1, e2)

            if sim > 0.3:
                edges.append({
                    "from": notes[i]["id"],
                    "to": notes[j]["id"]
                })

    return {
        "nodes": nodes,
        "edges": edges
    }


# ---------- semantic search ----------

def find_similar_notes(question, notes, k=5):

    q_emb = get_embedding(question)

    scored = []

    for n in notes:

        emb = n.get("embedding")

        if not emb:
            continue

        sim = cosine_sim(q_emb, emb)

        scored.append((sim, n))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [n for _, n in scored[:k]]


# ---------- RAG chat ----------

def rag_chat(question, notes):

    top_notes = find_similar_notes(question, notes, k=5)

    context = ""

    for n in top_notes:
        context += n["text"] + "\n"

    prompt = f"""
You are a Second Brain AI assistant.

Answer using the notes below.

Notes:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "used_notes": top_notes
    }


# ---------- routes ----------

@app.get("/")
def root():
    return {"message": "Second Brain API running"}


@app.get("/notes")
def get_notes():
    return read_notes()


@app.post("/add_note")
def add_note(data: dict):

    text = data.get("text")

    notes = read_notes()

    keywords = extract_keywords(text)

    note = create_note(text, keywords)

    notes.append(note)

    write_notes(notes)

    return note


@app.post("/upload_md")
async def upload_md(file: UploadFile = File(...)):

    content = await file.read()

    text = content.decode("utf-8")

    notes = read_notes()

    keywords = extract_keywords(text)

    note = create_note(text, keywords)

    notes.append(note)

    write_notes(notes)

    return {
        "filename": file.filename,
        "note": note
    }


@app.get("/graph")
def get_graph():

    notes = read_notes()

    return build_graph(notes)


@app.post("/chat")
def chat(data: dict):

    question = data.get("question")

    notes = read_notes()

    return rag_chat(question, notes)