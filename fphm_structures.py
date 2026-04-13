# fphm_structures.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Link:
    """定义一个带类型的链接"""
    target_id: str
    relationship_type: str

@dataclass
class TurnNote:
    """
    最底层的记忆单元，代表一个对话轮次。
    我们复用A-Mem中MemoryNote的核心思想，但结构更清晰。
    """
    id: str
    speaker: str
    content: str
    timestamp: str
    keywords: List[str] = field(default_factory=list)
    context: str = ""
    tags: List[str] = field(default_factory=list)
    # 增加了带类型的链接
    links: List[Link] = field(default_factory=list)
    # 增加了对父事件的引用，支持多重归属
    parent_event_ids: List[str] = field(default_factory=list)
    embedding: Optional[Any] = None

@dataclass
class FactSheet:
    """
    结构化的事实列表，用于保证摘要的事实性。
    """
    timeline: List[Dict[str, str]] = field(default_factory=list)  # e.g., [{"timestamp": "...", "fact": "...", "evidence_turn_id": "..."}]
    key_entities: List[str] = field(default_factory=list)

@dataclass
class EventSummary:
    """
    中间层记忆单元，代表一个事件。
    """
    id: str
    title: str
    summary_content: str
    fact_sheet: FactSheet = field(default_factory=FactSheet)
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    turn_note_ids: List[str] = field(default_factory=list)
    # 事件之间的链接
    links: List[Link] = field(default_factory=list)
    embedding: Optional[Any] = None
    specific_entities: List[str] = field(default_factory=list)

@dataclass
class CharacterProfile:
    """
    顶层记忆单元，代表一个人物。
    """
    character_name: str
    profile_summary: str = ""
    # 结构化属性，e.g., {"hobbies": ["baking", "skiing"]}
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    # 证据链，指向用于构建此简介的事件
    event_summary_ids: List[str] = field(default_factory=list)
    embedding: Optional[Any] = None
