# fphm_core.py

import json
import random
import uuid
import time
from functools import wraps
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from memory_layer import LLMController, SimpleEmbeddingRetriever
from fphm_structures import TurnNote, EventSummary, CharacterProfile, Link, FactSheet
from fphm_logger import FPHMLogger
import prompts


def retry(max_tries=10, initial_delay=5, backoff_factor=1.2, max_delay=120, jitter=True):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            tries = 0
            current_delay = initial_delay
            while tries < max_tries:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries >= max_tries:
                        final_error_message = f"LLM call failed after {max_tries} attempts. Final error: {e}"
                        print(final_error_message)
                        if hasattr(self, 'logger') and self.logger:
                            self.logger.log("llm_retry_failed",
                                            {"function": func.__name__, "max_tries": max_tries, "error": str(e)})
                        break
                    sleep_time = current_delay
                    if jitter:
                        sleep_time += random.uniform(0, current_delay * 0.25)
                    error_message = (f"LLM call failed (attempt {tries}/{max_tries}): {e}. "
                                     f"Retrying in {sleep_time:.2f}s...")
                    print(error_message)
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.log("llm_retry_attempt",
                                        {"function": func.__name__, "attempt": tries, "error": str(e),
                                         "delay": sleep_time})
                    time.sleep(sleep_time)
                    current_delay = min(current_delay * backoff_factor, max_delay)
            return None
        return wrapper
    return decorator


class FPHMSystem:
    def __init__(self, llm_controller: LLMController, run_name: str, use_character_profile: bool = True,
                 immediate_link_window: int = 5, use_event_title_mode: bool = False,
                 use_event_metadata_mode: bool = False,
                 use_attribute_focused_profile: bool = False, k_event_affiliation: int = 5,
                 ablation_no_fact_judgment: bool = False,
                 ablation_no_filter: bool = False,
                 ablation_no_link: bool = False,
                 ablation_no_event: bool = False,
                 ablation_mpnet_retrieval: bool = False,
                 log_dir: str = "fphm_logs"):
        self.k_event_affiliation = k_event_affiliation
        self.llm = llm_controller
        self.logger = FPHMLogger(log_dir=log_dir, run_name=run_name)
        self.use_character_profile = use_character_profile
        self.immediate_link_window = immediate_link_window
        self.use_event_title_mode = use_event_title_mode
        self.use_event_metadata_mode = use_event_metadata_mode
        self.use_attribute_focused_profile = use_attribute_focused_profile
        self.ablation_no_fact_judgment = ablation_no_fact_judgment
        self.ablation_no_filter = ablation_no_filter
        self.ablation_no_link = ablation_no_link
        self.ablation_no_event = ablation_no_event
        self.ablation_mpnet_retrieval = ablation_mpnet_retrieval
        self.logger.log("init", {
            "use_character_profile": self.use_character_profile,
            "immediate_link_window": self.immediate_link_window,
            "use_event_title_mode": self.use_event_title_mode,
            "use_event_metadata_mode": self.use_event_metadata_mode,
            "use_attribute_focused_profile": self.use_attribute_focused_profile,
            "k_event_affiliation": self.k_event_affiliation,
            "ablation_no_fact_judgment": self.ablation_no_fact_judgment,
            "ablation_no_filter": self.ablation_no_filter,
            "ablation_no_link": self.ablation_no_link,
            "ablation_no_event": self.ablation_no_event,
            "ablation_mpnet_retrieval": self.ablation_mpnet_retrieval,
            "mode": "synchronous_incremental_indexing_v8_ablations"
        })

        self.turn_notes: Dict[str, TurnNote] = {}
        self.events: Dict[str, EventSummary] = {}
        self.profiles: Dict[str, CharacterProfile] = {}

        if self.ablation_mpnet_retrieval:
            retriever_model_name = 'all-mpnet-base-v2'
            self.logger.log("retriever_model_selected", {"model": retriever_model_name})
        else:
            retriever_model_name = 'all-MiniLM-L6-v2'

        self.turn_retriever = SimpleEmbeddingRetriever(model_name=retriever_model_name)
        self.event_retriever = SimpleEmbeddingRetriever(model_name=retriever_model_name)
        if self.use_character_profile:
            self.profile_retriever = SimpleEmbeddingRetriever(model_name=retriever_model_name)
        self.last_updated_event_id: Optional[str] = None
        self.recent_turns_window: List[TurnNote] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
    def spawn_qa_view(
        self,
        llm_controller: Optional[LLMController] = None,
        *,
        executor_workers: int = 10,
    ) -> "FPHMSystem":
        """
        Create a read-only QA view that shares the current memory snapshot but owns an
        independent LLM controller / executor.

        This is used for question-level parallelism at a fixed checkpoint:
        the underlying memory state is not mutated during QA, while token accounting and
        LLM calls remain isolated per worker.
        """
        qa_view = object.__new__(FPHMSystem)
        qa_view.k_event_affiliation = self.k_event_affiliation
        qa_view.llm = llm_controller or self.llm
        qa_view.logger = self.logger
        qa_view.use_character_profile = self.use_character_profile
        qa_view.immediate_link_window = self.immediate_link_window
        qa_view.use_event_title_mode = self.use_event_title_mode
        qa_view.use_event_metadata_mode = self.use_event_metadata_mode
        qa_view.use_attribute_focused_profile = self.use_attribute_focused_profile
        qa_view.ablation_no_fact_judgment = self.ablation_no_fact_judgment
        qa_view.ablation_no_filter = self.ablation_no_filter
        qa_view.ablation_no_link = self.ablation_no_link
        qa_view.ablation_no_event = self.ablation_no_event
        qa_view.ablation_mpnet_retrieval = self.ablation_mpnet_retrieval
        qa_view.turn_notes = self.turn_notes
        qa_view.events = self.events
        qa_view.profiles = self.profiles
        qa_view.turn_retriever = self.turn_retriever
        qa_view.event_retriever = self.event_retriever
        if self.use_character_profile:
            qa_view.profile_retriever = self.profile_retriever
        qa_view.last_updated_event_id = self.last_updated_event_id
        qa_view.recent_turns_window = self.recent_turns_window
        qa_view.executor = ThreadPoolExecutor(max_workers=max(1, int(executor_workers)))
        return qa_view

    @retry()
    def _get_llm_json_response(self, prompt: str, schema: dict, caller: str, temperature: float = 0.0) -> Any:
        response_str = None
        parsed_response = None
        try:
            response_str = self.llm.llm.get_completion(prompt,
                                                       response_format={"type": "json_schema", "json_schema": schema},
                                                       temperature=temperature)
            try:
                parsed_response = json.loads(response_str)
                # Some OpenAI-compatible servers (or degraded runs after retries) may return a valid JSON
                # primitive (e.g., a JSON string) instead of the requested object. In that case, attempt a
                # schema-aware wrap for the common single-field schemas (keeps runs from crashing).
                if isinstance(parsed_response, dict):
                    return parsed_response
                expected = schema.get("schema", {}) if isinstance(schema, dict) else {}
                if expected.get("type") == "object":
                    props = expected.get("properties", {}) or {}
                    required = expected.get("required", []) or []
                    if len(required) == 1:
                        key = required[0]
                        prop_type = (props.get(key) or {}).get("type")
                        if prop_type == "string" and isinstance(parsed_response, str):
                            parsed_response = {key: parsed_response}
                            return parsed_response
                        if prop_type == "array" and isinstance(parsed_response, list):
                            parsed_response = {key: parsed_response}
                            return parsed_response
                # Fall through to the JSONDecodeError handler below to generate an empty response.
                raise json.JSONDecodeError("JSON response is not an object", str(response_str), 0)
            except json.JSONDecodeError:
                # Best-effort salvage for OpenAI-compatible servers / open models that may wrap JSON
                # in extra text. This does not affect normal runs where the response is already valid JSON.
                cleaned = str(response_str or "").strip()
                expected = schema.get("schema", {}) if isinstance(schema, dict) else {}
                if expected.get("type") == "object":
                    props = expected.get("properties", {}) or {}
                    required = expected.get("required", []) or []
                    if len(required) == 1:
                        key = required[0]
                        prop_type = (props.get(key) or {}).get("type")
                        if prop_type == "string" and cleaned:
                            # Open-source / OpenAI-compatible servers sometimes ignore json_schema
                            # and directly return the final option text. Preserve that answer instead
                            # of degrading to an empty string.
                            parsed_response = {key: cleaned}
                            return parsed_response
                # Try object-shaped extraction.
                if "{" in cleaned and "}" in cleaned:
                    start = cleaned.find("{")
                    end = cleaned.rfind("}")
                    if 0 <= start < end:
                        try:
                            parsed_response = json.loads(cleaned[start : end + 1])
                            return parsed_response
                        except Exception:
                            pass
                # Try array-shaped extraction (rare).
                if "[" in cleaned and "]" in cleaned:
                    start = cleaned.find("[")
                    end = cleaned.rfind("]")
                    if 0 <= start < end:
                        try:
                            parsed_response = json.loads(cleaned[start : end + 1])
                            return parsed_response
                        except Exception:
                            pass
                raise
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            self.logger.log("llm_json_error", {"prompt": prompt, "error": str(e),
                                               "raw_response": response_str if response_str is not None else 'N/A'})
            if schema.get("name") == "response" and schema.get("schema", {}).get("type") == "object":
                properties = schema["schema"].get("properties", {})
                empty_response = {}
                for key, prop_schema in properties.items():
                    prop_type = prop_schema.get("type")
                    if prop_type == "array":
                        empty_response[key] = []
                    elif prop_type == "string":
                        empty_response[key] = ""
                    elif prop_type == "object":
                        empty_response[key] = {}
                    else:
                        empty_response[key] = None
                parsed_response = empty_response
                return parsed_response
            return None
        finally:
            if hasattr(self, 'logger') and self.logger:
                self.logger.log("llm_call", {
                    "caller_function": caller,
                    "prompt": prompt,
                    "schema": schema,
                    "temperature": temperature,
                    "raw_response": response_str,
                    "parsed_response": parsed_response
                })


    @retry()
    def generate_query_llm(self, question: str) -> str:
        prompt = f"""You are an expert Search Query Optimizer for a Semantic Vector Retrieval system (using MPNet).
    
            Your goal is to translate a User Question into a **Generalized Declarative Memory Statement**.
            The embedding model retrieves best when the query looks like a vague but accurate summary of the answer found in a chat log.
            ### CRITICAL GUIDELINES:
            1. **Identify the Semantic Target:** Detailedly analyze what specific category of information is missing (e.g., a specific *Brand*, a *Timestamp*, a *Location Name*).
    
            2. **Analyze the True Answer Need (Beyond Surface Keywords):** - **Do not limit yourself to the literal keywords in the question.** Instead, deduce what the *actual answer* looks like in the real world.
               - *Example:* If the user asks "What does he drive?", the surface keyword is "drive", but the **true answer need** is a **Vehicle/Object** (like 'Toyota', 'Prius'). Focus the query on the *Object*, not the *Action*.
    
            3. **Use "Cluster Vocabulary":** Instead of guessing a specific answer, use a string of 3-4 high-level synonyms to cast a wide net.
               - Example: Instead of just "car", use "car, vehicle, brand, or model".
               - Example: Instead of just "time", use "timestamp, date, day, or moment".
    
            4. **Declarative Format:** Phrase it as a factual statement that *contains* the answer, but use placeholders.
            5. **NO Hallucinations:** Do NOT invent random dates (e.g., "Monday"), random names, or random brands. Use abstract terms only.
            ### Examples:
            User: "When did Calvin first travel to Tokyo?"
            Target: Time/Date
            Rewrite: "Calvin traveled to Tokyo at a specific time, date, day, or timestamp recorded in the conversation."
            User: "What kind of car does Evan drive?"
            Target: Entity (Vehicle) - *Answer Need: The Object (Vehicle), not the driving action.*
            Rewrite: "Evan owns or uses a specific car, vehicle, brand, automobile, or model."
            User: "Why was she upset?"
            Target: Reason/Context - *Answer Need: The Cause of emotion.*
            Rewrite: "She felt upset, angry, or sad because of a specific reason, cause, event, or incident mentioned in the chat."
            Format your response as a valid JSON object with a single key "search_query".
            User Question: {question}
            """

        response = self.llm.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                    }
                },
                "required": ["search_query"],
                "additionalProperties": False
            },
            "strict": True
        }})

        try:
            response = json.loads(response)["search_query"]
        except:
            if isinstance(response, str):
                clean_response = response.replace("```json", "").replace("```", "").strip()
                try:
                    response = json.loads(clean_response)["search_query"]
                except:
                    response = response.strip()
        final_query = f"{question} {response}"

        return final_query

    def _create_and_link_turn_note(self, turn_content: str, speaker: str, timestamp: str) -> TurnNote:
        if self.ablation_no_link:
            prompt = prompts.CREATE_TURN_NOTE_ISOLATED_PROMPT.format(
                speaker=speaker,
                timestamp=timestamp,
                content=turn_content
            )
            schema = {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "keywords": {"type": "array", "items": {"type": "string"}},
                                "context": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["keywords", "context", "tags"]
                        },
                        "profile_retrieval_keys": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["metadata", "profile_retrieval_keys"]
                }
            }
            response = self._get_llm_json_response(prompt, schema, caller='_create_turn_note_isolated')
            links_data = []
        else:
            history_str = json.dumps(
                [{"id": t.id, "speaker": t.speaker, "content": t.content} for t in self.recent_turns_window], indent=2)
            prompt = prompts.CREATE_AND_LINK_TURN_PROMPT.format(
                window_size=len(self.recent_turns_window),
                recent_history=history_str,
                speaker=speaker,
                timestamp=timestamp,
                content=turn_content
            )
            schema = {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "keywords": {"type": "array", "items": {"type": "string"}},
                                "context": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["keywords", "context", "tags"]
                        },
                        "profile_retrieval_keys": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "direct_links": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_turn_id": {"type": "string"},
                                    "relationship_type": {"type": "string"}
                                },
                                "required": ["target_turn_id", "relationship_type"]
                            }
                        }
                    },
                    "required": ["metadata", "profile_retrieval_keys", "direct_links"]
                }
            }
            response = self._get_llm_json_response(prompt, schema, caller='_create_and_link_turn_note')
            links_data = response.get('direct_links', []) if response else []

        metadata = response.get('metadata', {}) if response else {}
        profile_keys = response.get('profile_retrieval_keys', []) if response else []
        note_id = f"turn_{uuid.uuid4()}"
        turn_note = TurnNote(
            id=note_id, speaker=speaker, content=turn_content, timestamp=timestamp,
            keywords=metadata.get('keywords', []), context=metadata.get('context', ''), tags=metadata.get('tags', [])
        )
        setattr(turn_note, 'profile_retrieval_keys', profile_keys)
        self.logger.log("create_and_link_turn_note", {
            "turn_id": note_id,
            "llm_input_history": json.loads(history_str) if not self.ablation_no_link else "N/A (no-link ablation)",
            "llm_response": response
        })
        for link_data in links_data:
            target_id = link_data.get('target_turn_id')
            rel_type = link_data.get('relationship_type')
            if target_id in self.turn_notes:
                turn_note.links.append(Link(target_id=target_id, relationship_type=rel_type))
                self.turn_notes[target_id].links.append(
                    Link(target_id=turn_note.id, relationship_type=f"inverse_{rel_type}"))
                self.logger.log("immediate_linking", {"source": turn_note.id, "target": target_id, "type": rel_type})
        return turn_note

    def _get_facts_as_string(self, event: EventSummary) -> str:
        if not event.fact_sheet.timeline:
            return ""
        return " ".join([fact_item.get('fact', '') for fact_item in event.fact_sheet.timeline])

    def _decide_event_affiliation(self, turn: TurnNote, speaker: str) -> List[str]:
        candidate_event_indices = self.event_retriever.search(turn.content, k=self.k_event_affiliation)
        candidate_event_ids = [self.event_retriever.document_ids[i] for i in candidate_event_indices]
        candidate_events_list = [self.events[eid] for eid in candidate_event_ids if eid in self.events]
        if self.use_event_title_mode or self.use_event_metadata_mode:
            candidate_events_str = json.dumps(
                [{"id": e.id, "title": e.title} for e in candidate_events_list],
                indent=2
            )
            last_event_summary = self.events[
                self.last_updated_event_id].title if self.last_updated_event_id and self.last_updated_event_id in self.events else "None."
        else:
            candidate_events_str = json.dumps(
                [{"id": e.id, "summary": e.summary_content} for e in candidate_events_list],
                indent=2
            )
            last_event_summary = self.events[
                self.last_updated_event_id].summary_content if self.last_updated_event_id and self.last_updated_event_id in self.events else "None."
        profile_summary_for_context = "No relevant profiles found for context."
        context_profile_ids = []
        context_profiles_for_log = []
        if self.use_character_profile:
            profile_keys = getattr(turn, 'profile_retrieval_keys', [])
            for key in profile_keys:
                if key in self.profiles and key not in context_profile_ids:
                    context_profile_ids.append(key)
            self.logger.log("profile_retrieval_exact_match", {"keys": profile_keys, "found": context_profile_ids})
            num_needed = 3 - len(context_profile_ids)
            if num_needed > 0:
                profile_indices = self.profile_retriever.search(turn.content, k=num_needed + len(context_profile_ids))
                retrieved_ids = [self.profile_retriever.document_ids[i] for i in profile_indices]
                for pid in retrieved_ids:
                    if pid not in context_profile_ids:
                        context_profile_ids.append(pid)
                    if len(context_profile_ids) >= 3:
                        break
                self.logger.log("profile_retrieval_vector_supplement", {"needed": num_needed,
                                                                        "found_and_added": [pid for pid in retrieved_ids
                                                                                            if
                                                                                            pid not in profile_keys]})
            if context_profile_ids:
                profiles_for_context = []
                for pid in context_profile_ids:
                    if pid in self.profiles:
                        profile = self.profiles[pid]
                        profile_text_for_context = self._get_profile_doc_content(profile)
                        profiles_for_context.append(f"- Profile for {pid}: {profile_text_for_context}")
                        context_profiles_for_log.append({"id": pid, "summary_or_attributes": profile_text_for_context})
                profile_summary_for_context = "\n".join(profiles_for_context)

        new_turn_str = json.dumps({"id": turn.id, "speaker": turn.speaker, "content": turn.content}, indent=2)
        prompt = prompts.EVENT_AFFILIATION_PROMPT.format(
            k=len(candidate_events_list),
            profile_summary=profile_summary_for_context,
            last_event_summary=last_event_summary,
            candidate_events=candidate_events_str,
            new_turn=new_turn_str
        )
        schema = {"name": "response", "schema": {"type": "object", "properties": {"reasoning": {"type": "string"},
                                                                                  "affiliations": {"type": "array",
                                                                                                   "items": {
                                                                                                       "type": "string"}}},
                                                 "required": ["reasoning", "affiliations"]}}
        response = self._get_llm_json_response(prompt, schema, caller='_decide_event_affiliation')
        self.logger.log("decide_event_affiliation", {
            "turn_id": turn.id,
            "llm_input_candidates": json.loads(candidate_events_str),
            "llm_input_context_profiles": context_profiles_for_log,
            "llm_input_last_event_summary": last_event_summary,
            "llm_response": response
        })

        affiliations = response.get('affiliations', []) if response else []
        final_affiliations = []
        if "NEW_EVENT" in affiliations:
            new_event_id = f"event_{uuid.uuid4()}"
            if self.use_event_title_mode or self.use_event_metadata_mode:
                new_event = EventSummary(id=new_event_id, title=f"New event by {turn.speaker}", summary_content="")
            else:
                new_event = EventSummary(id=new_event_id,
                                         title=f"Event about: {turn.context}",
                                         summary_content=f"New event started by {turn.speaker} about: {turn.context}")
            self.events[new_event_id] = new_event
            final_affiliations.append(new_event_id)
            if self.use_event_title_mode or self.use_event_metadata_mode:
                self.event_retriever.add_document(doc_id=new_event_id, doc_content=new_event.title)
            else:
                self.event_retriever.add_document(doc_id=new_event_id, doc_content=new_event.summary_content)

        for aff in affiliations:
            if aff != "NEW_EVENT" and aff in self.events:
                final_affiliations.append(aff)

        if not final_affiliations:
            new_event_id = f"event_{uuid.uuid4()}"
            if self.use_event_title_mode or self.use_event_metadata_mode:
                new_event = EventSummary(id=new_event_id, title=f"New event by {turn.speaker}", summary_content="")
            else:
                new_event = EventSummary(id=new_event_id,
                                         title=f"Event about: {turn.context}",
                                         summary_content=f"New event started by {turn.speaker} about: {turn.context}")
            self.events[new_event_id] = new_event
            final_affiliations.append(new_event_id)
            self.logger.log("decide_event_affiliation_fallback", {"turn_id": turn.id, "created_event": new_event_id})
            if self.use_event_title_mode or self.use_event_metadata_mode:
                self.event_retriever.add_document(doc_id=new_event_id, doc_content=new_event.title)
            else:
                self.event_retriever.add_document(doc_id=new_event_id, doc_content=new_event.summary_content)

        return final_affiliations

    def _update_event(self, event_id: str, turn: TurnNote):
        if event_id not in self.events: return
        event = self.events[event_id]
        recent_dialogue_context = json.dumps(
            [{"id": t.id, "speaker": t.speaker, "content": t.content} for t in self.recent_turns_window], indent=2,
            ensure_ascii=False)
        new_turn_text_input = json.dumps(
            {"id": turn.id, "timestamp": turn.timestamp, "speaker": turn.speaker, "content": turn.content},
            ensure_ascii=False
        )
        if self.use_event_title_mode:
            current_fact_sheet_json_input = json.dumps(event.fact_sheet.__dict__, default=str, ensure_ascii=False)
            if self.ablation_no_fact_judgment:
                prompt = prompts.EXTRACT_FACTS_DIRECTLY_PROMPT.format(
                    event_context=event.title,
                    new_turn_text=new_turn_text_input
                )
                schema = {"name": "response", "schema": {"type": "object", "properties": {
                    "new_facts": {"type": "array", "items": {"type": "object", "properties": {
                        "timestamp": {"type": "string"}, "fact": {"type": "string"},
                        "evidence_turn_id": {"type": "string"}
                    }}}
                }, "required": ["new_facts"]}}
                response = self._get_llm_json_response(prompt, schema, caller='_update_event_extract_facts_only')
                self.logger.log("update_event_no_judgment", {"event_id": event_id, "llm_response": response})
                if response:
                    new_facts = response.get('new_facts', [])
                    event.fact_sheet.timeline.extend(new_facts)
            else:
                if len(event.turn_note_ids) < 10:
                    prompt = prompts.UPDATE_ADAPTIVE_SUMMARY_PROMPT.format(
                        current_summary=event.title,
                        current_keywords=json.dumps(event.keywords),
                        current_tags=json.dumps(event.tags),
                        current_fact_sheet_json=current_fact_sheet_json_input,
                        recent_dialogue_context=recent_dialogue_context,
                        new_turn_text=new_turn_text_input
                    )
                    schema = {"name": "response", "schema": {"type": "object", "properties": {
                        "updated_summary": {"type": "string"},
                        "updated_keywords": {"type": "array", "items": {"type": "string"}},
                        "updated_tags": {"type": "array", "items": {"type": "string"}},
                        "new_facts": {"type": "array", "items": {"type": "object", "properties": {
                            "timestamp": {"type": "string"}, "fact": {"type": "string"},
                            "evidence_turn_id": {"type": "string"}
                        }}}
                    }, "required": ["updated_summary", "updated_keywords", "updated_tags", "new_facts"]}}
                    response = self._get_llm_json_response(prompt, schema, caller='_update_event_adaptive_small')
                    self.logger.log("update_event_adaptive_summary_small",
                                    {"event_id": event_id, "llm_response": response})
                    if response:
                        event.title = response.get('updated_summary', event.title)
                        event.keywords = response.get('updated_keywords', event.keywords)
                        event.tags = response.get('updated_tags', event.tags)
                        new_facts = response.get('new_facts', [])
                        event.fact_sheet.timeline.extend(new_facts)
                else:
                    prompt = prompts.UPDATE_EVENT_TITLE_MODE_LARGE_EVENT_PROMPT.format(
                        current_fact_sheet_json=current_fact_sheet_json_input,
                        new_turn_text=new_turn_text_input
                    )
                    schema = {"name": "response", "schema": {"type": "object", "properties": {
                        "new_facts": {"type": "array", "items": {"type": "object", "properties": {
                            "timestamp": {"type": "string"}, "fact": {"type": "string"},
                            "evidence_turn_id": {"type": "string"}
                        }}}
                    }, "required": ["new_facts"]}}
                    response = self._get_llm_json_response(prompt, schema, caller='_update_event_adaptive_large')
                    self.logger.log("update_event_adaptive_summary_large",
                                    {"event_id": event_id, "llm_response": response})
                    if response:
                        new_facts = response.get('new_facts', [])
                        event.fact_sheet.timeline.extend(new_facts)
            event.summary_content = ""
            doc_content = event.title
            self.event_retriever.add_document(doc_id=event_id, doc_content=doc_content)
        elif self.use_event_metadata_mode:
            if len(event.turn_note_ids) < 10:
                current_fact_sheet_json_input = json.dumps(event.fact_sheet.__dict__, default=str, ensure_ascii=False)
                prompt = prompts.UPDATE_ADAPTIVE_SUMMARY_PROMPT.format(
                    current_summary=event.title,
                    current_keywords=json.dumps(event.keywords),
                    current_tags=json.dumps(event.tags),
                    current_fact_sheet_json=current_fact_sheet_json_input,
                    recent_dialogue_context=recent_dialogue_context,
                    new_turn_text=new_turn_text_input
                )
                schema = {"name": "response", "schema": {"type": "object", "properties": {
                    "updated_summary": {"type": "string"},
                    "updated_keywords": {"type": "array", "items": {"type": "string"}},
                    "updated_tags": {"type": "array", "items": {"type": "string"}},
                    "new_facts": {"type": "array", "items": {"type": "object", "properties": {
                        "timestamp": {"type": "string"}, "fact": {"type": "string"},
                        "evidence_turn_id": {"type": "string"}
                    }}}
                }, "required": ["updated_summary", "updated_keywords", "updated_tags", "new_facts"]}}
                response = self._get_llm_json_response(prompt, schema, caller='_update_event_metadata_small')
                self.logger.log("update_event_metadata_small", {"event_id": event_id, "llm_response": response})
                if response:
                    event.title = response.get('updated_summary', event.title)
                    event.keywords = response.get('updated_keywords', event.keywords)
                    event.tags = response.get('updated_tags', event.tags)
                    new_facts = response.get('new_facts', [])
                    event.fact_sheet.timeline.extend(new_facts)
            else:
                prompt = prompts.UPDATE_EVENT_METADATA_LARGE_EVENT_PROMPT.format(
                    current_title=event.title,
                    current_keywords=json.dumps(event.keywords),
                    current_tags=json.dumps(event.tags),
                    new_turn_text=new_turn_text_input
                )
                schema = {"name": "response", "schema": {"type": "object", "properties": {
                    "is_relevant": {"type": "boolean"},
                    "updated_keywords": {"type": "array", "items": {"type": "string"}},
                    "updated_tags": {"type": "array", "items": {"type": "string"}}
                }, "required": ["is_relevant", "updated_keywords", "updated_tags"]}}
                response = self._get_llm_json_response(prompt, schema, caller='_update_event_metadata_large')
                self.logger.log("update_event_metadata_large", {"event_id": event_id, "llm_response": response})
                if response and response.get('is_relevant'):
                    event.keywords = response.get('updated_keywords', event.keywords)
                    event.tags = response.get('updated_tags', event.tags)

            event.summary_content = ""
            doc_content = f"{' '.join(event.keywords)} {' '.join(event.tags)} {event.title}"
            self.event_retriever.add_document(doc_id=event_id, doc_content=doc_content)
        else:
            current_summary_text_input = event.summary_content
            current_fact_sheet_json_input = json.dumps(event.fact_sheet.__dict__, default=str, ensure_ascii=False)
            prompt = prompts.UPDATE_EVENT_PROMPT.format(current_summary_text=current_summary_text_input,
                                                        current_fact_sheet_json=current_fact_sheet_json_input,
                                                        recent_dialogue_context=recent_dialogue_context,
                                                        new_turn_text=new_turn_text_input)
            schema = {"name": "response", "schema": {"type": "object", "properties": {
                "specific_entities": {"type": "array", "items": {"type": "string"}},
                "updated_summary_text": {"type": "string"}, "updated_fact_sheet": {"type": "object", "properties": {
                    "timeline": {"type": "array", "items": {"type": "object",
                                                            "properties": {"timestamp": {"type": "string"},
                                                                           "fact": {"type": "string"},
                                                                           "evidence_turn_id": {"type": "string"}},
                                                            "required": ["timestamp", "fact", "evidence_turn_id"]}},
                    "key_entities": {"type": "array", "items": {"type": "string"}}}, "required": ["timeline",
                                                                                                  "key_entities"]}},
                                                     "required": ["specific_entities", "updated_summary_text",
                                                                  "updated_fact_sheet"]}}
            response = self._get_llm_json_response(prompt, schema, caller='_update_event_default')
            self.logger.log("update_event_default_mode", {
                "event_id": event_id, "trigger_turn_id": turn.id, "llm_response": response
            })
            if response:
                event.specific_entities = list(set(event.specific_entities + response.get('specific_entities', [])))
                event.summary_content = response.get('updated_summary_text', event.summary_content)
                fact_sheet_data = response.get('updated_fact_sheet', {})
                event.fact_sheet = FactSheet(timeline=fact_sheet_data.get('timeline', []),
                                             key_entities=fact_sheet_data.get('key_entities', []))
            doc_content = f"Event Summary: {event.summary_content}"
            self.event_retriever.add_document(doc_id=event_id, doc_content=doc_content)
        if turn.speaker not in event.specific_entities: event.specific_entities.append(turn.speaker)
        if turn.id not in event.turn_note_ids: event.turn_note_ids.append(turn.id)
        if event_id not in turn.parent_event_ids: turn.parent_event_ids.append(event_id)
        self.events[event_id] = event
        self.logger.log("incremental_index_update", {"type": "event", "updated_id": event_id,
                                                     "mode": "adaptive_summary" if self.use_event_title_mode else "default"})

    def _get_profile_doc_content(self, profile: CharacterProfile) -> str:
        if self.use_attribute_focused_profile:
            attributes_str = json.dumps(profile.attributes, ensure_ascii=False)
            return f"Character: {profile.character_name}. Profile: {profile.profile_summary}. Attributes: {attributes_str}"
        else:
            return f"Character: {profile.character_name}. Profile: {profile.profile_summary}"

    def _update_profile(self, entity_name: str, event_source: EventSummary):
        if not self.use_character_profile: return
        if entity_name not in self.profiles:
            self.logger.log("profile_update_skipped",
                            {"reason": "Profile does not exist for this entity.", "entity": entity_name})
            return
        profile = self.profiles[entity_name]

        event_fact_sheet_json_input = json.dumps(event_source.fact_sheet.__dict__, default=str)

        if self.use_event_title_mode or self.use_event_metadata_mode:
            event_context_for_profile_update = (
                f"Event Title: {event_source.title}\n"
                f"Event Facts: {event_fact_sheet_json_input}"
            )
            event_summary_text_input_for_log = f"Title: {event_source.title}"
        else:
            event_context_for_profile_update = event_source.summary_content
            event_summary_text_input_for_log = event_source.summary_content
        current_profile_summary_input = profile.profile_summary
        current_attributes_json_input = json.dumps(profile.attributes)

        if self.use_attribute_focused_profile:
            prompt = prompts.UPDATE_PROFILE_ATTRIBUTE_FOCUSED_PROMPT.format(
                entity_name=entity_name,
                current_profile_summary=current_profile_summary_input,
                current_attributes_json=current_attributes_json_input,
                event_id=event_source.id,
                event_context=event_context_for_profile_update,
                event_fact_sheet_json=event_fact_sheet_json_input
            )
            log_mode = "attribute_focused"
        else:
            prompt = prompts.UPDATE_PROFILE_PROMPT.format(
                entity_name=entity_name,
                current_profile_summary=current_profile_summary_input,
                current_attributes_json=current_attributes_json_input,
                event_id=event_source.id,
                event_context=event_context_for_profile_update,
                event_fact_sheet_json=event_fact_sheet_json_input
            )
            log_mode = "narrative_summary"

        schema = {"name": "response", "schema": {"type": "object",
                                                 "properties": {"updated_profile_summary": {"type": "string"},
                                                                "updated_attributes": {"type": "object"}},
                                                 "required": ["updated_profile_summary", "updated_attributes"]}}
        response = self._get_llm_json_response(prompt, schema, caller=f'_update_profile_{log_mode}')

        self.logger.log("update_profile", {
            "character": entity_name,
            "trigger_event_id": event_source.id,
            "mode": log_mode,
            "llm_input": {
                "current_profile_summary": current_profile_summary_input,
                "current_attributes": json.loads(current_attributes_json_input),
                "event_summary_or_context": event_summary_text_input_for_log,
                "event_fact_sheet": json.loads(event_fact_sheet_json_input)
            },
            "llm_response": response
        })

        if response:
            profile.profile_summary = response.get('updated_profile_summary', profile.profile_summary)
            profile.attributes = response.get('updated_attributes', profile.attributes)
        if event_source.id not in profile.event_summary_ids: profile.event_summary_ids.append(event_source.id)
        self.profiles[entity_name] = profile

        doc_content = self._get_profile_doc_content(profile)
        self.profile_retriever.add_document(doc_id=profile.character_name, doc_content=doc_content)
        self.logger.log("incremental_index_update", {"type": "profile", "updated_id": entity_name, "mode": log_mode})

    def add_turn(self, turn_id: str, turn_content: str, speaker: str, timestamp: str):
        start_time = time.time()

        turn_note = self._create_and_link_turn_note(turn_content, speaker, timestamp)
        turn_note.id = turn_id
        self.turn_notes[turn_note.id] = turn_note
        self.recent_turns_window.append(turn_note)
        if len(self.recent_turns_window) > self.immediate_link_window: self.recent_turns_window.pop(0)
        doc_content = f"Speaker: {turn_note.speaker}. Time: {turn_note.timestamp}. Content: {turn_note.content}"
        self.turn_retriever.add_document(doc_id=turn_note.id, doc_content=doc_content)

        if self.ablation_no_event:
            self.logger.log("add_turn_skipped_event_profile_logic", {"turn_id": turn_id, "reason": "no-event ablation"})
            end_time = time.time()
            duration = end_time - start_time
            self.logger.log("timing_add_turn", {"duration_seconds": duration, "turn_id": turn_id})
            return

        affiliated_event_ids = self._decide_event_affiliation(turn_note, speaker)
        current_main_event_id = affiliated_event_ids[0] if affiliated_event_ids else None
        if self.use_character_profile and self.last_updated_event_id and current_main_event_id != self.last_updated_event_id:
            old_event = self.events.get(self.last_updated_event_id)
            if old_event and old_event.specific_entities:
                if self.use_event_title_mode or self.use_event_metadata_mode:
                    event_context_for_decision = f"Event Title: {old_event.title}"
                else:
                    event_context_for_decision = old_event.summary_content
                decision_prompt = prompts.PROFILE_UPDATE_DECISION_PROMPT.format(event_id=old_event.id,
                                                                                specific_entities=json.dumps(
                                                                                    old_event.specific_entities),
                                                                                event_summary=event_context_for_decision)
                schema = {"name": "response", "schema": {"type": "object", "properties": {
                    "update_decisions": {"type": "array", "items": {"type": "object",
                                                                    "properties": {"entity_name": {"type": "string"},
                                                                                   "should_update": {"type": "boolean"},
                                                                                   "reasoning": {"type": "string"}},
                                                                    "required": ["entity_name", "should_update"]}}},
                                                         "required": ["update_decisions"]}}
                decision_response = self._get_llm_json_response(decision_prompt, schema,
                                                                caller='profile_update_decision')
                self.logger.log("profile_update_decision",
                                {"trigger_event_id": old_event.id, "decisions": decision_response})
                if decision_response and 'update_decisions' in decision_response:
                    for decision in decision_response['update_decisions']:
                        if decision.get('should_update'):
                            entity_to_update = decision.get('entity_name')
                            if entity_to_update not in self.profiles:
                                self.profiles[entity_to_update] = CharacterProfile(character_name=entity_to_update)
                                self.logger.log("create_profile_on_demand", {"character": entity_to_update})
                                doc_content = self._get_profile_doc_content(self.profiles[entity_to_update])
                                self.profile_retriever.add_document(doc_id=entity_to_update, doc_content=doc_content)
                            self._update_profile(entity_to_update, old_event)
        for event_id in affiliated_event_ids: self._update_event(event_id, turn_note)
        if current_main_event_id: self.last_updated_event_id = current_main_event_id

        end_time = time.time()
        duration = end_time - start_time
        self.logger.log("timing_add_turn", {"duration_seconds": duration, "turn_id": turn_id})

    def finalize_memory_build(self):
        self.logger.log("finalize_memory_build_started", {"last_event_id": self.last_updated_event_id})
        if self.use_character_profile and self.last_updated_event_id:
            last_event = self.events.get(self.last_updated_event_id)
            if last_event and last_event.specific_entities:
                self.logger.log("final_profile_update_triggered", {"event_id": last_event.id})

                if self.use_event_title_mode or self.use_event_metadata_mode:
                    event_context_for_decision = f"Event Title: {last_event.title}"
                else:
                    event_context_for_decision = last_event.summary_content
                decision_prompt = prompts.PROFILE_UPDATE_DECISION_PROMPT.format(
                    event_id=last_event.id,
                    specific_entities=json.dumps(last_event.specific_entities),
                    event_summary=event_context_for_decision
                )
                schema = {"name": "response", "schema": {"type": "object", "properties": {
                    "update_decisions": {"type": "array", "items": {"type": "object",
                                                                    "properties": {"entity_name": {"type": "string"},
                                                                                   "should_update": {"type": "boolean"},
                                                                                   "reasoning": {"type": "string"}},
                                                                    "required": ["entity_name", "should_update"]}}},
                                                         "required": ["update_decisions"]}}
                decision_response = self._get_llm_json_response(decision_prompt, schema,
                                                                caller='final_profile_update_decision')
                self.logger.log("final_profile_update_decision",
                                {"trigger_event_id": last_event.id, "decisions": decision_response})
                if decision_response and 'update_decisions' in decision_response:
                    for decision in decision_response['update_decisions']:
                        if decision.get('should_update'):
                            entity_to_update = decision.get('entity_name')
                            if entity_to_update not in self.profiles:
                                self.profiles[entity_to_update] = CharacterProfile(character_name=entity_to_update)
                                self.logger.log("create_profile_on_demand_final", {"character": entity_to_update})
                                doc_content = self._get_profile_doc_content(self.profiles[entity_to_update])
                                self.profile_retriever.add_document(doc_id=entity_to_update, doc_content=doc_content)
                            self._update_profile(entity_to_update, last_event)
        self.logger.log("finalize_memory_build_finished", {})

    def build_indices(self):
        self.logger.log("final_index_check", {"message": "Running a final check on all indices."})
        turn_docs = {tid: f"Speaker: {t.speaker}. Time: {t.timestamp}. Content: {t.content}" for tid, t in
                     self.turn_notes.items()}
        self.turn_retriever.add_documents(turn_docs)

        if self.use_event_title_mode:
            event_docs = {eid: self._get_facts_as_string(e) for eid, e in self.events.items()}
            self.logger.log("build_indices_event_mode", {"mode": "title_mode (facts indexed)"})
        elif self.use_event_metadata_mode:
            event_docs = {eid: f"{' '.join(e.keywords)} {' '.join(e.tags)} {e.title}" for eid, e in self.events.items()}
            self.logger.log("build_indices_event_mode", {"mode": "metadata_mode (keywords+tags+title indexed)"})
        else:
            event_docs = {eid: f"Event Summary: {e.summary_content}" for eid, e in self.events.items()}
            self.logger.log("build_indices_event_mode", {"mode": "default_mode (summaries indexed)"})
        self.event_retriever.add_documents(event_docs)

        if self.use_character_profile:
            profile_docs = {pid: self._get_profile_doc_content(p) for pid, p in self.profiles.items()}
            self.profile_retriever.add_documents(profile_docs)
        self.logger.log("final_index_built", {"turn_notes": len(turn_docs), "events": len(event_docs),
                                              "profiles": len(profile_docs) if self.use_character_profile else 0})

    def _judge_relevance_parallel(self, query: str, items: Dict[str, str], item_type: str, chunk_size: int = 1) -> \
    List[
        str]:
        if not items: return []
        item_chunks = [dict(list(items.items())[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
        futures = []
        for chunk in item_chunks:
            formatted_items = "\n".join([f"- ID: {id}\n  Content: {content}" for id, content in chunk.items()])
            prompt = prompts.RELEVANCE_JUDGMENT_PROMPT.format(original_query=query,
                                                              memory_items_formatted_string=formatted_items)
            schema = {"name": "response", "schema": {"type": "object", "properties": {
                "relevant_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["relevant_ids"]}}
            futures.append(self.executor.submit(self._get_llm_json_response, prompt, schema,
                                                caller=f'_judge_relevance_parallel_{item_type}'))
        relevant_ids = []
        for future in as_completed(futures):
            response = future.result()
            if response and 'relevant_ids' in response:
                relevant_ids.extend(response['relevant_ids'])
        self.logger.log(f"judge_relevance_{item_type}",
                        {"query": query, "candidates": list(items.keys()), "selected": relevant_ids,
                         "chunk_size": chunk_size})
        return relevant_ids

    def _judge_relevance_sequential(self, query: str, items: Dict[str, str], item_type: str) -> List[str]:
        if not items:
            return []
        formatted_items = "\n".join([f"- ID: {id}\n  Content: {content}" for id, content in items.items()])
        prompt = prompts.RELEVANCE_JUDGMENT_PROMPT.format(
            original_query=query,
            memory_items_formatted_string=formatted_items
        )
        schema = {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "relevant_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["relevant_ids"]
            }
        }
        response = self._get_llm_json_response(prompt, schema, caller=f'_judge_relevance_sequential_{item_type}')
        relevant_ids = []
        if response and 'relevant_ids' in response:
            relevant_ids.extend(response['relevant_ids'])
        self.logger.log(f"judge_relevance_sequential_{item_type}",
                        {"query": query, "candidates": list(items.keys()), "selected": relevant_ids})
        return relevant_ids

    def retrieve_for_query(
        self,
        original_query: str,
        keyword_query: str,
        profile_retrieval_keys: List[str],
        k_profile: int,
        k_event: int,
        k_turn: int,
        return_trace: bool = False,
    ):
        if self.ablation_no_event:
            turn_indices = self.turn_retriever.search(keyword_query, k=k_turn)
            candidate_turn_ids = [self.turn_retriever.document_ids[i] for i in turn_indices]
            self.logger.log("initial_recall_no_event",
                            {"query": original_query, "recalled_turns": candidate_turn_ids})
            relevant_turn_ids = candidate_turn_ids
            if not self.ablation_no_filter:
                candidate_turns = {tid: self.turn_notes[tid].content for tid in candidate_turn_ids if
                                   tid in self.turn_notes}
                relevant_turn_ids = self._judge_relevance_parallel(original_query, candidate_turns, "turn",
                                                                   chunk_size=10)
            else:
                self.logger.log("retrieve_for_query_skipped_filter", {"reason": "no-filter ablation"})
            final_context_parts = []
            try:
                sorted_turns = sorted([self.turn_notes[tid] for tid in relevant_turn_ids if tid in self.turn_notes],
                                      key=lambda t: t.timestamp)
            except (TypeError, ValueError):
                sorted_turns = sorted([self.turn_notes[tid] for tid in relevant_turn_ids if tid in self.turn_notes],
                                      key=lambda t: t.id)
            for t in sorted_turns:
                turn_string = (
                    f"Timestamp: {t.timestamp}"
                    f" Speaker: {t.speaker}"
                    f" Content: {t.content}"
                    f" Context Summary: {t.context}\n"
                )
                final_context_parts.append(turn_string)
            final_context = "\n\n".join(final_context_parts)
            self.logger.log("final_context_construction_no_event",
                            {"query": original_query, "final_context": final_context})
            if return_trace:
                trace = {
                    "mode": "no_event",
                    "keyword_query": keyword_query,
                    "candidate_turn_ids": candidate_turn_ids,
                    "relevant_turn_ids": relevant_turn_ids,
                }
                return final_context, trace
            return final_context
        turn_indices = self.turn_retriever.search(keyword_query, k=k_turn)
        event_indices = self.event_retriever.search(keyword_query, k=k_event)
        candidate_turn_ids = [self.turn_retriever.document_ids[i] for i in turn_indices]
        candidate_event_ids = [self.event_retriever.document_ids[i] for i in event_indices]
        candidate_profile_ids = []
        candidate_profiles = {}
        if self.use_character_profile:
            for key in profile_retrieval_keys:
                if key in self.profiles and key not in candidate_profile_ids:
                    candidate_profile_ids.append(key)
            num_needed = k_profile - len(candidate_profile_ids)
            if num_needed > 0:
                profile_indices = self.profile_retriever.search(keyword_query, k=num_needed)
                retrieved_ids_by_vector = [self.profile_retriever.document_ids[i] for i in profile_indices]
                for pid in retrieved_ids_by_vector:
                    if pid not in candidate_profile_ids:
                        candidate_profile_ids.append(pid)
            candidate_profiles = {pid: self._get_profile_doc_content(self.profiles[pid]) for pid in
                                  candidate_profile_ids if
                                  pid in self.profiles}
        self.logger.log("initial_recall", {"query": original_query, "keyword_query": keyword_query,
                                           "recalled_profiles": list(candidate_profiles.keys()),
                                           "recalled_events": candidate_event_ids,
                                           "recalled_turns": candidate_turn_ids})
        relevant_profile_ids, predicted_event_ids_from_profiles = [], []
        if self.use_character_profile and candidate_profiles:
            if self.ablation_no_filter:
                relevant_profile_ids = list(candidate_profiles.keys())
            else:
                relevant_profile_ids = self._judge_relevance_parallel(original_query, candidate_profiles, "profile",
                                                                      chunk_size=1)
            futures = []
            for pid in relevant_profile_ids:
                if pid not in self.profiles: continue
                profile = self.profiles[pid]
                profile_content_for_prompt = self._get_profile_doc_content(profile)
                known_events_str = json.dumps(
                    [{"id": eid, "summary": self.events[eid].summary_content[:150] + "..."} for eid in
                     profile.event_summary_ids if eid in self.events], indent=2, ensure_ascii=False)
                prompt = prompts.PREDICT_EVENTS_FROM_PROFILE_PROMPT.format(original_query=original_query,
                                                                           character_name=profile.character_name,
                                                                           profile_summary=profile_content_for_prompt,
                                                                           attributes_json="{}",
                                                                           list_of_known_events=known_events_str)
                schema = {"name": "response", "schema": {"type": "object",
                                                         "properties": {"reasoning": {"type": "string"},
                                                                        "predicted_event_ids": {"type": "array",
                                                                                                "items": {
                                                                                                    "type": "string"}}},
                                                         "required": ["predicted_event_ids"]}}
                futures.append(self.executor.submit(self._get_llm_json_response, prompt, schema,
                                                    caller='predict_events_from_profile'))
            for future in as_completed(futures):
                response = future.result()
                if response and 'predicted_event_ids' in response: predicted_event_ids_from_profiles.extend(
                    response['predicted_event_ids'])
            self.logger.log("predict_events_from_profile",
                            {"query": original_query, "relevant_profiles": relevant_profile_ids,
                             "predicted_events": predicted_event_ids_from_profiles})
        fused_candidate_event_ids = list(set(candidate_event_ids + predicted_event_ids_from_profiles))
        if self.use_event_title_mode or self.use_event_metadata_mode:
            candidate_events = {eid: self.events[eid].title for eid in fused_candidate_event_ids if
                                eid in self.events}
        else:
            candidate_events = {eid: self.events[eid].summary_content for eid in fused_candidate_event_ids if
                                eid in self.events}
        relevant_event_ids = list(candidate_events.keys())
        predicted_turn_ids_from_events = []
        futures = []
        for eid in relevant_event_ids:
            if eid not in self.events: continue
            event = self.events[eid]
            event_context = event.title if self.use_event_title_mode or self.use_event_metadata_mode else event.summary_content
            internal_turns_formatted = "\n".join(
                [f"- ID: {tid}\n  Content: {self.turn_notes[tid].content}" for tid in event.turn_note_ids if
                 tid in self.turn_notes])
            prompt = f"""You are a research assistant. Your task is to select relevant dialogue turns from within a single event that help answer the question.
## Question:
"{original_query}"
## Event Context:
- Event ID: {eid}
- Event Summary/Title: {event_context}
## Dialogue Turns within this Event:
{internal_turns_formatted}
## Instructions:
Based on the question and the event's context, identify which of the dialogue turns listed above are most likely to contain the answer.
## Output (JSON format):
{{
  "relevant_turn_ids_from_event": ["list of relevant turn IDs from this specific event"]
}}
"""
            schema = {"name": "response", "schema": {"type": "object", "properties": {
                "relevant_turn_ids_from_event": {"type": "array", "items": {"type": "string"}}
            }, "required": ["relevant_turn_ids_from_event"]}}
            futures.append(
                self.executor.submit(self._get_llm_json_response, prompt, schema,
                                     caller='predict_turns_from_event'))
        for future in as_completed(futures):
            response = future.result()
            if response and 'relevant_turn_ids_from_event' in response:
                predicted_turn_ids_from_events.extend(response['relevant_turn_ids_from_event'])
        self.logger.log("predict_turns_from_event", {
            "query": original_query,
            "relevant_events": relevant_event_ids,
            "predicted_turns": predicted_turn_ids_from_events
        })
        initial_fused_turn_ids = list(set(candidate_turn_ids + predicted_turn_ids_from_events))

        expanded_turn_ids = list(initial_fused_turn_ids)
        added_by_links = {}
        if not self.ablation_no_link:
            for turn_id in initial_fused_turn_ids:
                if turn_id in self.turn_notes:
                    note = self.turn_notes[turn_id]
                    linked_ids = [link.target_id for link in note.links]
                    if linked_ids:
                        added_by_links[turn_id] = linked_ids
                        expanded_turn_ids.extend(linked_ids)

        final_fused_candidate_turn_ids = list(set(expanded_turn_ids))

        self.logger.log("expand_turns_by_links", {
            "query": original_query,
            "initial_candidates_count": len(initial_fused_turn_ids),
            "initial_candidates": initial_fused_turn_ids,
            "links_found_and_added": added_by_links,
            "expanded_candidates_count": len(final_fused_candidate_turn_ids),
            "expanded_candidates": final_fused_candidate_turn_ids
        })

        candidate_turns = {tid: self.turn_notes[tid].content for tid in final_fused_candidate_turn_ids if
                           tid in self.turn_notes}
        if self.ablation_no_filter:
            relevant_turn_ids = list(candidate_turns.keys())
            self.logger.log("retrieve_for_query_skipped_filter", {"reason": "no-filter ablation"})
        else:
            relevant_turn_ids = self._judge_relevance_parallel(original_query, candidate_turns, "turn",
                                                               chunk_size=10)
        final_context_parts = []
        try:
            sorted_turns = sorted([self.turn_notes[tid] for tid in relevant_turn_ids if tid in self.turn_notes],
                                  key=lambda t: t.timestamp)
        except (TypeError, ValueError):
            sorted_turns = sorted([self.turn_notes[tid] for tid in relevant_turn_ids if tid in self.turn_notes],
                                  key=lambda t: t.id)
        for t in sorted_turns:
            turn_string = (
                f"--- Turn Start ---\n"
                f"Timestamp: {t.timestamp}\n"
                f"Speaker: {t.speaker}\n"
                f"Content: {t.content}\n"
                f"Context Summary: {t.context}\n"
                f"--- Turn End ---"
            )
            final_context_parts.append(turn_string)
        final_context = "\n\n".join(final_context_parts)
        self.logger.log("final_context_construction", {"query": original_query, "final_context": final_context})
        if return_trace:
            trace = {
                "mode": "full",
                "keyword_query": keyword_query,
                "profile_retrieval_keys": profile_retrieval_keys,
                "candidate_profile_ids": candidate_profile_ids,
                "candidate_event_ids": candidate_event_ids,
                "candidate_turn_ids": candidate_turn_ids,
                "relevant_event_ids": relevant_event_ids,
                "predicted_turn_ids_from_events": predicted_turn_ids_from_events,
                "final_fused_candidate_turn_ids": final_fused_candidate_turn_ids,
                "relevant_turn_ids": relevant_turn_ids,
            }
            return final_context, trace
        return final_context
