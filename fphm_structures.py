# fphm_structures.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Link:
    target_id: str
    relationship_type: str

@dataclass
class TurnNote:
    id: str
    speaker: str
    content: str
    timestamp: str
    keywords: List[str] = field(default_factory=list)
    context: str = ""
    tags: List[str] = field(default_factory=list)

    links: List[Link] = field(default_factory=list)

    parent_event_ids: List[str] = field(default_factory=list)
    embedding: Optional[Any] = None

@dataclass
class FactSheet:
    timeline: List[Dict[str, str]] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)

@dataclass
class EventSummary:
    id: str
    title: str
    summary_content: str
    fact_sheet: FactSheet = field(default_factory=FactSheet)
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    turn_note_ids: List[str] = field(default_factory=list)

    links: List[Link] = field(default_factory=list)
    embedding: Optional[Any] = None
    specific_entities: List[str] = field(default_factory=list)

@dataclass
class CharacterProfile:
    character_name: str
    profile_summary: str = ""

    attributes: Dict[str, List[str]] = field(default_factory=dict)

    event_summary_ids: List[str] = field(default_factory=list)
    embedding: Optional[Any] = None
