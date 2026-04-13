# prompts.py

# -----------------------------------------------------------------------------
# 1. TurnNote级别 (底层)
# -----------------------------------------------------------------------------

# prompts.py

CREATE_AND_LINK_TURN_PROMPT = """
You are a dialogue analyst and knowledge graph architect. Your task is to process a "Current Turn" by analyzing its content, identifying key entities, and linking it to the "Recent History".
## Core Instructions:
1.  **Analyze Content**: Generate `keywords`, a `context` summary, and `tags` for the "Current Turn".
2.  **Identify Specific Entities for Profile Retrieval**: From the "Current Turn", extract all **specific, identifiable entities** whose profiles would be relevant for understanding this turn's context.
    - An entity MUST be a **Named Entity** (e.g., 'Anna', 'Melanie', 'Aspen', 'Fido') OR a **Specifically Owned Entity** (e.g., 'Caroline's children', 'Melanie's husband').
    - Do NOT extract generic common nouns like 'friends', 'kids', 'family', or abstract concepts like 'mental health'.
3.  **Identify Conversational Links**: Determine if the "Current Turn" has a direct conversational relationship (e.g., question-answer) with any turn in the "Recent History".
## Recent History (previous {window_size} turns):
{recent_history}
## Current Turn to Process:
- Speaker: {speaker}
- Timestamp: {timestamp}
- Text: {content}
## Output (JSON format):
{{
  "metadata": {{
    "keywords": ["list of specific, distinct keywords"],
    "context": "A one-sentence summary of the turn's context.",
    "tags": ["list of broad categories/themes"]
  }},
  "profile_retrieval_keys": ["list of specific entity names to look up, e.g., 'Caroline', 'Melanie', 'Caroline's children'"],
  "direct_links": [
    {{
      "target_turn_id": "ID of a turn from recent history",
      "relationship_type": "e.g., 'answer_to_question', 'continuation_of'"
    }}
  ]
}}
// If no direct link is found, "direct_links" should be an empty list.
"""


# -----------------------------------------------------------------------------
# 2. EventSummary级别 (中层)
# -----------------------------------------------------------------------------

UPDATE_EVENT_PROMPT = """
You are a meticulous archivist. Your task is to update an event summary and its fact sheet by integrating a new dialogue turn.
## Core Instructions:
1.  **Analyze Context**: Understand the "New Turn" in the context of the "Recent Dialogue Turns" that preceded it.
2.  **Identify Specific Entities**: From the "New Turn" and its context, list all key entities involved for whom a personal profile might be created. An entity MUST be a **Named Entity** (e.g., 'Anna', 'Melanie', 'Aspen', 'Fido') OR a **Specifically Owned Entity** (e.g., "Caroline's children", "Melanie's husband"). 
    - **DO NOT** extract abstract concepts ('mental health', 'counseling'), activities ('painting'), or generic groups ('friends', 'kids', 'family').
3.  **Fact Extraction**: Extract all objective facts from the "New Turn".
4.  **Update Summary**: Rewrite the "Current Summary" to chronologically incorporate the new facts. When you add information from the "New Turn", you MUST append its ID as evidence, like this: [turn_id].
5.  **Update Fact Sheet**: Add new facts to the structured fact sheet. Each fact MUST also cite its evidence ID.
## Input:
### Current Event Summary:
{current_summary_text}
### Current Event Fact Sheet:
{current_fact_sheet_json}
### Recent Dialogue Turns (for context):
{recent_dialogue_context}
### New Turn to Integrate:
{new_turn_text}
## Output (JSON format):
{{
  "specific_entities": ["list of all unique specific entity names found, e.g., 'Anna', 'Caroline's children'. This list should be empty if no valid entities are found."],
  "updated_summary_text": "The new, complete narrative of the event, with evidence IDs. For example: 'Anna made a gluten-free cake [D8:8] for her dog, Fido [D8:8].'",
  "updated_fact_sheet": {{
    "timeline": [
      {{"timestamp": "...", "fact": "Anna made a cake.", "evidence_turn_id": "D8:8"}},
      {{"timestamp": "...", "fact": "The cake was for Fido.", "evidence_turn_id": "D8:8"}}
    ],
    "key_entities": ["Anna", "cake", "Fido"]
  }}
}}
"""

# -----------------------------------------------------------------------------
# 3. CharacterProfile级别 (顶层)
# -----------------------------------------------------------------------------

PROFILE_UPDATE_DECISION_PROMPT = """
You are a lead biographer deciding which entity profiles to update based on a concluded event.
## Concluded Event:
- **ID**: {event_id}
- **Identified Entities**: {specific_entities}
- **Summary**: {event_summary}
## Instructions:
Analyze the event summary. For each identified entity, decide if this event contains significant new information about them that warrants an update to their personal profile. An update is warranted if the event reveals new personality traits, hobbies, life goals, relationships, or significant life events for that entity.
**An entity MUST be a Named Entity (e.g., 'Anna', 'Fido') or a Specifically Owned Entity (e.g., 'Caroline's children'). Do not suggest updates for abstract concepts or generic groups.**
## Output (JSON format):
{{
  "update_decisions": [
    {{
      "entity_name": "A specific entity name from the list, e.g., 'Anna' or 'Fido the dog'",
      "should_update": boolean,
      "reasoning": "Briefly explain why this event is (or is not) significant for this entity's profile."
    }}
  ]
}}
"""

UPDATE_PROFILE_PROMPT = """
You are a character analyst and biographer. Update the profile for an entity (person, pet, etc.) based on a recently concluded event, ensuring every piece of information is traceable to its source.
## Core Instructions:
1.  **Extract Insights**: Analyze the "Concluded Event" to find new information about the entity's personality, attributes, or life story.
2.  **Update Profile Summary**: Weave the new insights into the "Current Profile Summary". Every new statement MUST cite the event ID as evidence, like this: [event_id].
3.  **Update Attributes**: Add or update the structured "Attributes". Each attribute value MUST also cite its evidence event ID.
## Input:
### Entity Name: {entity_name}
### Current Profile Summary:
{current_profile_summary}
### Current Attributes:
{current_attributes_json}
### Concluded Event to Incorporate:
- **Event ID**: {event_id}
- **Event Context**: {event_context}
- **Event Fact Sheet**: {event_fact_sheet_json}
## Output (JSON format):
{{
  "updated_profile_summary": "A comprehensive narrative of the entity, with evidence IDs. For example: 'Fido is a beloved dog who enjoys special treats like gluten-free cake [event_cake_baking].'",
  "updated_attributes": {{
    "diet": ["gluten-free cake [event_cake_baking]"],
    "owner": ["Anna [event_cake_baking]"]
  }}
}}
"""

# -----------------------------------------------------------------------------
# 4. QA检索相关
# -----------------------------------------------------------------------------

# 【保持不变】这个Prompt用于事件归属，其逻辑是正确的。
EVENT_AFFILIATION_PROMPT = """
You are a memory organization agent. Your task is to determine the event affiliation for a "New Turn". A turn can belong to one or more existing events, or be the start of a new one.

## Context for Decision:
- Character Profile (if available): {profile_summary}
- Last Updated Event Summary: {last_event_summary}

## Candidate Events (Top-{k} most similar events):
{candidate_events}

## New Turn:
{new_turn}

## Instructions:
1.  Analyze the "New Turn" in the context of the character's history and recent conversations.
2.  Compare its content and topic with each "Candidate Event".
3.  Decide which event(s) it belongs to. It's possible for a turn to bridge two topics, thus belonging to multiple events.
4.  If it doesn't fit any existing event, it's a "NEW_EVENT".

## Output (JSON format):
{{
  "reasoning": "A brief explanation of your decision process.",
  "affiliations": ["list of relevant event IDs from candidates", "or 'NEW_EVENT' if it starts a new one"]
}}
"""

# 【修改】增强RELEVANCE_JUDGMENT_PROMPT，使其在关注主语的同时，具有一定的灵活性。
RELEVANCE_JUDGMENT_PROMPT = """
You are a helpful research assistant. Your task is to judge if the following memory items are helpful for answering the question.

## Core Instructions:
1.  **Analyze Question**: Identify the main subject(s) (person, pet, entity), the core action/topic, and any temporal constraints (e.g., 'after the road trip', 'in June 2023') from the "Question".
2.  **Primary Filter (Subject Check)**: A memory item is **highly relevant** if its main subject matches the question's subject.
3.  **Secondary Filter (Context Check)**: If the subject doesn't directly match, consider if the item provides crucial context to the question's topic. For example, if the question is about 'Anna's feelings', a memory of 'Bob comforting Anna' is relevant even if Bob is the speaker.
4.  **Be Inclusive, Not Strict**: Your goal is to provide all potentially useful clues. When in doubt, lean towards including the memory item. Do not discard a memory just because one aspect (e.g., time) doesn't perfectly match, if other aspects are highly relevant.

## Question:
"{original_query}"

## Memory Items:
{memory_items_formatted_string}

## Output (JSON format):
{{
  "relevant_ids": ["list of IDs of all helpful memory items"]
}}
"""

# 【修改】增强QUERY_REWRITING_PROMPT，使其同时提取关键词和用于Profile检索的实体键
QUERY_REWRITING_PROMPT = """
You are a search query optimization expert. Your task is to analyze a user's question and extract two types of information:
1.  **Keyword Query**: A concise set of keywords for vector-based semantic search. Extract important entities, concepts, and actions. Omit conversational filler or question words.
    - Do not include conversational filler or question words (like 'what', 'how', 'did').
    - The output should be a single string of space-separated keywords.
2.  **Profile Retrieval Keys**: A list of all **specific, identifiable entities** mentioned in the question whose profiles would be relevant.
    - An entity MUST be a **Named Entity** (e.g., 'Evan', 'Sam', 'Fido') OR a **Specifically Owned Entity** (e.g., 'Evan's son').
    - Do NOT extract generic nouns ('car', 'friend') or abstract concepts ('advice', 'growth').
## User Question:
"{original_query}"
## Output (JSON format):
{{
  "keyword_query": "keyword1 keyword2 entity3...",
  "profile_retrieval_keys": ["list of specific entity names to look up, e.g., 'Evan', 'Sam'"]
}}
"""

PREDICT_EVENTS_FROM_PROFILE_PROMPT = """
You are a detective reviewing a suspect's file to find relevant case files.

## Question:
"{original_query}"

## Character Profile:
- **Name**: {character_name}
- **Summary**: {profile_summary}
- **Attributes**: {attributes_json}
- **Known Events (ID and Summary)**:
{list_of_known_events}

## Instructions:
Based on the question and the character's complete profile, which of their "Known Events" are most likely to contain the answer?

## Output (JSON format):
{{
  "reasoning": "Briefly explain why you chose these events.",
  "predicted_event_ids": ["list of event IDs from the profile that you predict are relevant"]
}}
"""

FINAL_ANSWER_GENERATION_PROMPT = """
You are a Question Answering assistant. Your task is to answer the question based *only* on the provided context. You must be concise and factual.
## Provided Context Structure:
The context may contain up to three types of information:
1.  **Relevant Character Profiles**: High-level summaries about people/entities.
2.  **Relevant Event Summaries**: Narratives of specific events.
3.  **Relevant Dialogue Turns**: Raw, verbatim excerpts from the conversation, marked with IDs like [D1:2].
## Core Answering Rules:
- **Prioritize Dialogue Turns**: If a direct, factual answer is present in the 'Relevant Dialogue Turns' section, you MUST prioritize extracting it.
- **Use Summaries for Context**: Use 'Event Summaries' and 'Character Profiles' for broader context or when a direct answer is not in the dialogue turns.
- **Adhere to Constraints**: You must strictly follow the specific answering constraints provided below.
## Context:
{context}
## Answering Constraints:
{constraints}
## Question:
{question}
## Output (JSON format):
{{
  "answer": "Your generated answer here."
}}
"""

UPDATE_EVENT_TITLE_MODE_SMALL_EVENT_PROMPT = """
You are a meticulous archivist. Your task is to update a small event's metadata (title, keywords, tags) and its fact sheet by integrating a new dialogue turn.
## Core Instructions:
1.  **Analyze Context**: Understand the "New Turn" in the context of the "Recent Dialogue Turns".
2.  **Update Title**: Based on all information so far, create or refine a concise, descriptive title for the event (max 10 words).
3.  **Update Metadata**: Generate relevant `keywords` and `tags` for the event.
4.  **Extract New Facts**: Extract all new, objective facts from the "New Turn" and add them to the fact sheet. Each fact MUST cite the new turn's ID as evidence.
## Input:
### Current Event Title:
{current_title}
### Current Event Keywords:
{current_keywords}
### Current Event Tags:
{current_tags}
### Current Event Fact Sheet:
{current_fact_sheet_json}
### Recent Dialogue Turns (for context):
{recent_dialogue_context}
### New Turn to Integrate:
{new_turn_text}
## Output (JSON format):
{{
  "updated_title": "The new or refined concise title for the event.",
  "updated_keywords": ["list", "of", "updated", "keywords"],
  "updated_tags": ["list", "of", "updated", "tags"],
  "new_facts": [
    {{"timestamp": "...", "fact": "A new fact extracted from the turn.", "evidence_turn_id": "..."}}
  ]
}}
"""
UPDATE_ADAPTIVE_SUMMARY_PROMPT = """
You are a meticulous archivist. Your task is to update a small event's metadata by integrating a new dialogue turn.
## Core Instructions:
1.  **Analyze Context**: Understand the "New Turn" in the context of the "Recent Dialogue Turns".
2.  **Update Summary**: Based on all information so far, create or refine a concise, descriptive summary for the event (2-3 sentences). This summary will be used to represent the event.
3.  **Update Metadata**: Generate relevant `keywords` and `tags` for the event.
4.  **Extract New Facts**: Extract all new, objective facts from the "New Turn" and add them to the fact sheet. Each fact MUST cite the new turn's ID as evidence.
## Input:
### Current Event Summary (previously generated):
{current_summary}
### Current Event Keywords:
{current_keywords}
### Current Event Tags:
{current_tags}
### Current Event Fact Sheet:
{current_fact_sheet_json}
### Recent Dialogue Turns (for context):
{recent_dialogue_context}
### New Turn to Integrate:
{new_turn_text}
## Output (JSON format):
{{
  "updated_summary": "The new or refined concise summary for the event (2-3 sentences).",
  "updated_keywords": ["list", "of", "updated", "keywords"],
  "updated_tags": ["list", "of", "updated", "tags"],
  "new_facts": [
    {{"timestamp": "...", "fact": "A new fact extracted from the turn.", "evidence_turn_id": "..."}}
  ]
}}
"""
# --- MODIFICATION END ---
# 这个Prompt用于处理包含10个或更多Turn的大事件。
# 它只要求LLM从新Turn中提取事实，而不需要更新标题等元数据，以节省成本和保持稳定性。
UPDATE_EVENT_TITLE_MODE_LARGE_EVENT_PROMPT = """
You are a factual data entry assistant. Your task is to extract ONLY the new, objective facts from a "New Turn" and format them for addition to an existing event's fact sheet. Do not summarize or analyze.
## Core Instructions:
1.  **Extract New Facts**: Read the "New Turn" and identify all new, objective pieces of information.
2.  **Format Facts**: For each fact, record its timestamp and cite the "New Turn" ID as evidence.
## Input:
### Current Event Fact Sheet (for context, to avoid duplication):
{current_fact_sheet_json}
### New Turn to Integrate:
{new_turn_text}
## Output (JSON format):
{{
  "new_facts": [
    {{"timestamp": "...", "fact": "A new fact extracted from the turn.", "evidence_turn_id": "..."}}
  ]
}}
"""

UPDATE_PROFILE_ATTRIBUTE_FOCUSED_PROMPT = """
You are a character analyst creating a structured fact sheet for an entity. Your goal is to extract specific attributes, not to write a narrative.
## Core Instructions:
1.  **Generate a Minimal Summary**: Create a very concise, one-line descriptive summary for the entity (e.g., "A passionate baker and dog lover"). This summary should act like a title.
2.  **Extract Structured Attributes**: Analyze the "Concluded Event" to find new, specific attributes.
3.  **Cite Evidence Precisely**: For each attribute value, you MUST cite its evidence.
    - The base citation is the event ID: `[event:event_id]`.
    - **Crucially**, if the attribute comes from a specific fact in the event's "Fact Sheet", you MUST also add the `evidence_turn_id` from that fact. The final format should be: `value [event:event_id, turn:turn_id_from_fact]`.
## Input:
### Entity Name: {entity_name}
### Current Profile Summary:
{current_profile_summary}
### Current Attributes:
{current_attributes_json}
### Concluded Event to Incorporate:
- **Event ID**: {event_id}
- **Event Context**: {event_context}
- **Event Fact Sheet**: {event_fact_sheet_json}
## Output (JSON format):
{{
  "updated_profile_summary": "A very concise, one-line summary of the entity.",
  "updated_attributes": {{
    "hobbies": ["baking [event:event_cake_baking, turn:D8:8]", "skiing [event:ski_trip]"],
    "pets": ["dog named Fido [event:event_cake_baking, turn:D8:8]"]
  }}
}}
"""

EXTRACT_FACTS_DIRECTLY_PROMPT = """
You are a data entry assistant. Your task is to extract new, objective facts from a "New Turn" that are relevant to the provided "Event Context".
## Core Instructions:
1.  Read the "Event Context" to understand the event's main topic.
2.  Read the "New Turn" and identify any new, objective pieces of information that logically belong to this event.
3.  If you find relevant facts, format them for addition to a fact sheet. Each fact MUST cite the "New Turn" ID as evidence.
4.  **If the "New Turn" is completely unrelated to the "Event Context" or contains no new objective facts, return an empty list for "new_facts".**
## Input:
### Event Context (Title/Summary):
{event_context}
### New Turn to Integrate:
{new_turn_text}
## Output (JSON format):
{{
  "new_facts": [
    {{"timestamp": "...", "fact": "A new fact extracted from the turn.", "evidence_turn_id": "..."}}
  ]
}}
"""
# 【新增】为消融实验3 (no-link) 设计的Prompt
# 这是 CREATE_AND_LINK_TURN_PROMPT 的简化版，移除了历史上下文和链接生成的指令。
CREATE_TURN_NOTE_ISOLATED_PROMPT = """
You are a dialogue analyst. Your task is to process a "Current Turn" in isolation by analyzing its content and identifying key entities.
## Core Instructions:
1.  **Analyze Content**: Generate `keywords`, a `context` summary, and `tags` for the "Current Turn".
2.  **Identify Specific Entities for Profile Retrieval**: From the "Current Turn", extract all **specific, identifiable entities** whose profiles would be relevant for understanding this turn's context.
    - An entity MUST be a **Named Entity** (e.g., 'Anna', 'Melanie', 'Aspen', 'Fido') OR a **Specifically Owned Entity** (e.g., 'Caroline's children', 'Melanie's husband').
    - Do NOT extract generic common nouns like 'friends', 'kids', 'family', or abstract concepts like 'mental health'.
## Current Turn to Process:
- Speaker: {speaker}
- Timestamp: {timestamp}
- Text: {content}
## Output (JSON format):
{{
  "metadata": {{
    "keywords": ["list of specific, distinct keywords"],
    "context": "A one-sentence summary of the turn's context.",
    "tags": ["list of broad categories/themes"]
  }},
  "profile_retrieval_keys": ["list of specific entity names to look up, e.g., 'Caroline', 'Melanie', 'Caroline's children'"]
}}
"""

UPDATE_EVENT_METADATA_LARGE_EVENT_PROMPT = """
You are a data efficiency expert and memory curator. For a large, established event, you must first determine if a "New Turn" is relevant. If it is, you must then concisely update the event's `keywords` and `tags`.
## Core Instructions:
1.  **Relevance Check**: First, decide if the "New Turn" is topically relevant to the "Current Event Title".
2.  **Metadata Update (if relevant)**:
    - Analyze the "New Turn" for new, important concepts.
    - **Consolidate, DO NOT just append**: Review the "Current Keywords/Tags". Merge new information intelligently. If a new keyword is just a synonym of an existing one, keep the existing one. If it's a genuinely new concept, add it.
    - Your goal is to keep the keyword and tag lists concise and highly descriptive, preventing them from growing too long.
3.  **Output**: Return your decision and the updated metadata. If the turn is not relevant, `is_relevant` must be `false`, and the keyword/tag lists should be returned unchanged.
## Input:
### Current Event Title:
{current_title}
### Current Keywords:
{current_keywords}
### Current Tags:
{current_tags}
### New Turn to Integrate:
{new_turn_text}
## Output (JSON format):
{{
  "is_relevant": boolean,
  "updated_keywords": ["list", "of", "consolidated", "keywords"],
  "updated_tags": ["list", "of", "consolidated", "tags"]
}}
"""