from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from pathlib import Path
import json
import os

# --- App Setup ---
app = FastAPI(
    title="Knowledge Graph Server",
    version="1.0.0",
    description="A structured knowledge graph memory system that supports entity and relation storage, observation tracking, and manipulation."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Persistence Path ---
MEMORY_FILE_PATH_ENV = os.getenv("MEMORY_FILE_PATH", "memory.json")
MEMORY_FILE_PATH = Path(
    MEMORY_FILE_PATH_ENV if Path(MEMORY_FILE_PATH_ENV).is_absolute() else Path(__file__).parent / MEMORY_FILE_PATH_ENV
)

# --- Models ---
class Entity(BaseModel):
    name: str
    entityType: str
    observations: List[str]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source: Optional[str] = None
    user_id: Optional[str] = None
    tags: Optional[List[str]] = None

class Relation(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    relationType: str

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

class ObservationItem(BaseModel):
    entityName: str
    contents: List[str]

class AddObservationsRequest(BaseModel):
    observations: List[ObservationItem]

class CreateEntitiesRequest(BaseModel):
    entities: List[Entity]

class CreateRelationsRequest(BaseModel):
    relations: List[Relation]

class DeleteEntitiesRequest(BaseModel):
    entityNames: List[str]

class DeleteRelationsRequest(BaseModel):
    relations: List[Relation]

class DeletionItem(BaseModel):
    entityName: str
    observations: List[str]

class DeleteObservationsRequest(BaseModel):
    deletions: List[DeletionItem]

class SearchNodesRequest(BaseModel):
    query: str

class OpenNodesRequest(BaseModel):
    names: List[str]

# --- Helpers ---
def read_graph_file() -> KnowledgeGraph:
    if not MEMORY_FILE_PATH.exists():
        return KnowledgeGraph(entities=[], relations=[])
    with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
        entities, relations = [], []
        for line in lines:
            item = json.loads(line)
            if item["type"] == "entity":
                entities.append(Entity(**{k: v for k, v in item.items() if k != "type"}))
            elif item["type"] == "relation":
                relations.append(Relation(**{k: v for k, v in item.items() if k != "type"}))
        return KnowledgeGraph(entities=entities, relations=relations)

def save_graph(graph: KnowledgeGraph):
    lines = [json.dumps({"type": "entity", **e.dict()}) for e in graph.entities] + \
            [json.dumps({"type": "relation", **r.dict(by_alias=True)}) for r in graph.relations]
    with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --- Endpoints ---
@app.post("/create_entities")
def create_entities(req: CreateEntitiesRequest):
    graph = read_graph_file()
    existing = {e.name for e in graph.entities}
    now = datetime.utcnow().isoformat()
    new_entities = [Entity(**e.dict(), created_at=now) for e in req.entities if e.name not in existing]
    graph.entities.extend(new_entities)
    save_graph(graph)
    return new_entities

@app.post("/create_relations")
def create_relations(req: CreateRelationsRequest):
    graph = read_graph_file()
    existing = {(r.from_, r.to, r.relationType) for r in graph.relations}
    new = [r for r in req.relations if (r.from_, r.to, r.relationType) not in existing]
    graph.relations.extend(new)
    save_graph(graph)
    return new

@app.post("/add_observations")
def add_observations(req: AddObservationsRequest):
    graph = read_graph_file()
    now = datetime.utcnow().isoformat()
    results = []
    for obs in req.observations:
        entity = next((e for e in graph.entities if e.name == obs.entityName), None)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity {obs.entityName} not found")
        added = [c for c in obs.contents if c not in entity.observations]
        entity.observations.extend(added)
        entity.updated_at = now
        results.append({"entityName": obs.entityName, "addedObservations": added})
    save_graph(graph)
    return results

@app.post("/delete_entities")
def delete_entities(req: DeleteEntitiesRequest):
    graph = read_graph_file()
    graph.entities = [e for e in graph.entities if e.name not in req.entityNames]
    graph.relations = [r for r in graph.relations if r.from_ not in req.entityNames and r.to not in req.entityNames]
    save_graph(graph)
    return {"message": "Entities deleted"}

@app.post("/delete_observations")
def delete_observations(req: DeleteObservationsRequest):
    graph = read_graph_file()
    for d in req.deletions:
        entity = next((e for e in graph.entities if e.name == d.entityName), None)
        if entity:
            entity.observations = [o for o in entity.observations if o not in d.observations]
            entity.updated_at = datetime.utcnow().isoformat()
    save_graph(graph)
    return {"message": "Observations deleted"}

@app.post("/delete_relations")
def delete_relations(req: DeleteRelationsRequest):
    graph = read_graph_file()
    del_set = {(r.from_, r.to, r.relationType) for r in req.relations}
    graph.relations = [r for r in graph.relations if (r.from_, r.to, r.relationType) not in del_set]
    save_graph(graph)
    return {"message": "Relations deleted"}

@app.get("/read_graph", response_model=KnowledgeGraph)
def read_graph():
    return read_graph_file()

@app.post("/search_nodes", response_model=KnowledgeGraph)
def search_nodes(req: SearchNodesRequest):
    graph = read_graph_file()
    entities = [
        e for e in graph.entities if req.query.lower() in e.name.lower()
        or req.query.lower() in (e.entityType or "").lower()
        or any(req.query.lower() in o.lower() for o in e.observations)
    ]
    names = {e.name for e in entities}
    relations = [r for r in graph.relations if r.from_ in names and r.to in names]
    return KnowledgeGraph(entities=entities, relations=relations)

@app.post("/open_nodes", response_model=KnowledgeGraph)
def open_nodes(req: OpenNodesRequest):
    graph = read_graph_file()
    entities = [e for e in graph.entities if e.name in req.names]
    names = {e.name for e in entities}
    relations = [r for r in graph.relations if r.from_ in names and r.to in names]
    return KnowledgeGraph(entities=entities, relations=relations)
