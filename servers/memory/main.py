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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEMORY_FILE_PATH = Path(os.getenv("MEMORY_FILE_PATH", "memory.json"))

class Entity(BaseModel):
    name: str
    entityType: str
    observations: List[str]
    created_at: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    source: Optional[str] = None
    user_id: Optional[str] = None
    tags: Optional[List[str]] = []

class Relation(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    relationType: str
    created_at: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    source: Optional[str] = None
    user_id: Optional[str] = None
    tags: Optional[List[str]] = []

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

class SearchNodesRequest(BaseModel):
    query: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    user_id: Optional[str] = None

@app.post("/search_nodes", response_model=KnowledgeGraph)
def search_nodes(req: SearchNodesRequest):
    graph = read_graph_file()
    entities = graph.entities

    if req.query:
        entities = [
            e for e in entities
            if req.query.lower() in e.name.lower()
            or req.query.lower() in e.entityType.lower()
            or any(req.query.lower() in obs.lower() for obs in e.observations)
        ]

    if req.tags:
        entities = [e for e in entities if set(req.tags).intersection(set(e.tags or []))]

    if req.source:
        entities = [e for e in entities if e.source == req.source]

    if req.user_id:
        entities = [e for e in entities if e.user_id == req.user_id]

    names = {e.name for e in entities}
    relations = [r for r in graph.relations if r.from_ in names and r.to in names]
    return KnowledgeGraph(entities=entities, relations=relations)

@app.post("/add_observations")
def add_observations(req: AddObservationsRequest):
    graph = read_graph_file()
    result = []
    for obs in req.observations:
        entity = next((e for e in graph.entities if e.name == obs.entityName), None)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{obs.entityName}' not found")
        added = [c for c in obs.contents if c not in entity.observations]
        if added:
            entity.observations.extend(added)
            entity.updated_at = datetime.utcnow().isoformat()
        result.append({"entity": entity.name, "added": added})
    save_graph(graph)
    return result

# Other endpoints remain unchanged
