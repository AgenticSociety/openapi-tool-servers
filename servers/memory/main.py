from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from pathlib import Path
from datetime import datetime
import json
import os

app = FastAPI(
    title="Knowledge Graph Server",
    version="1.0.0",
    description="A structured knowledge graph memory system that supports entity and relation storage, observation tracking, and manipulation.",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Persistence Setup -----
MEMORY_FILE_PATH_ENV = os.getenv("MEMORY_FILE_PATH", "memory.json")
MEMORY_FILE_PATH = Path(
    MEMORY_FILE_PATH_ENV
    if Path(MEMORY_FILE_PATH_ENV).is_absolute()
    else Path(__file__).parent / MEMORY_FILE_PATH_ENV
)

# ----- Data Models -----
class Entity(BaseModel):
    name: str = Field(..., description="The name of the entity")
    entityType: str = Field(..., description="The type of the entity")
    observations: List[str] = Field(..., description="An array of observation contents associated with the entity")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class Relation(BaseModel):
    from_: str = Field(..., alias="from", description="The name of the entity where the relation starts")
    to: str = Field(..., description="The name of the entity where the relation ends")
    relationType: str = Field(..., description="The type of the relation")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

class EntityWrapper(BaseModel):
    type: Literal["entity"]
    name: str
    entityType: str
    observations: List[str]
    created_at: str

class RelationWrapper(BaseModel):
    type: Literal["relation"]
    from_: str = Field(..., alias="from")
    to: str
    relationType: str
    created_at: str

# ----- I/O Handlers -----
def read_graph_file() -> KnowledgeGraph:
    if not MEMORY_FILE_PATH.exists():
        return KnowledgeGraph(entities=[], relations=[])
    with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
        entities = []
        relations = []
        for line in lines:
            item = json.loads(line)
            if item["type"] == "entity":
                entities.append(Entity(**item))
            elif item["type"] == "relation":
                relations.append(Relation(**item))
        return KnowledgeGraph(entities=entities, relations=relations)

def save_graph(graph: KnowledgeGraph):
    lines = [json.dumps({"type": "entity", **e.dict()}) for e in graph.entities] + [
        json.dumps({"type": "relation", **r.dict(by_alias=True)}) for r in graph.relations
    ]
    with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
