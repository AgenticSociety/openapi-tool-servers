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
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class Relation(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    relationType: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

class CreateEntitiesRequest(BaseModel):
    entities: List[Entity]

class CreateRelationsRequest(BaseModel):
    relations: List[Relation]

class ObservationItem(BaseModel):
    entityName: str
    contents: List[str]

class AddObservationsRequest(BaseModel):
    observations: List[ObservationItem]

class DeletionItem(BaseModel):
    entityName: str
    observations: List[str]

class DeleteObservationsRequest(BaseModel):
    deletions: List[DeletionItem]

class DeleteEntitiesRequest(BaseModel):
    entityNames: List[str]

class DeleteRelationsRequest(BaseModel):
    relations: List[Relation]

class SearchNodesRequest(BaseModel):
    query: str

class OpenNodesRequest(BaseModel):
    names: List[str]

def read_graph_file() -> KnowledgeGraph:
    if not MEMORY_FILE_PATH.exists():
        return KnowledgeGraph(entities=[], relations=[])
    with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
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
    lines = [json.dumps({"type": "entity", **e.dict()}) for e in graph.entities] + \
            [json.dumps({"type": "relation", **r.dict(by_alias=True)}) for r in graph.relations]
    with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

@app.post("/create_entities")
def create_entities(req: CreateEntitiesRequest):
    graph = read_graph_file()
    existing = {e.name for e in graph.entities}
    new_entities = [e for e in req.entities if e.name not in existing]
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
    result = []
    for obs in req.observations:
        entity = next((e for e in graph.entities if e.name == obs.entityName), None)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{obs.entityName}' not found")
        added = [c for c in obs.contents if c not in entity.observations]
        entity.observations.extend(added)
        result.append({"entity": entity.name, "added": added})
    save_graph(graph)
    return result

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
    for deletion in req.deletions:
        entity = next((e for e in graph.entities if e.name == deletion.entityName), None)
        if entity:
            entity.observations = [o for o in entity.observations if o not in deletion.observations]
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
        e for e in graph.entities
        if req.query.lower() in e.name.lower()
        or req.query.lower() in e.entityType.lower()
        or any(req.query.lower() in obs.lower() for obs in e.observations)
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
