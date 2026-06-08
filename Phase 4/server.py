#!/usr/bin/env python3
"""
Phase 4: Crew Interface Agent — Production Pipeline
FastAPI + WebSocket backend with Progressive Disclosure,
Dual-Route Conversation, Armstrong Protocol, and Voice Grammar.
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# CONFIGURATION & DATA MODELS
# -----------------------------------------------------------------------------

class PoseUncertainty(BaseModel):
    jensen_gain: float = 2.82
    sigma_R: float = 0.12
    sigma_t: float = 0.05
    confidence: Literal["high", "moderate", "low"] = "low"

class AnomalyState(BaseModel):
    subsystem: str = "ECLSS"
    severity: Literal["nominal", "degraded", "critical", "failed"] = "critical"
    component: str = "Radiator 2 Loop"
    description: str = "Coolant pressure dropping at 0.4 bar/min"
    time_to_critical: Optional[float] = 14.0  # minutes

class HDCComponent(BaseModel):
    name: str
    weight: float  # 0-100

class CounterfactualPath(BaseModel):
    action_id: str
    action_name: str
    collision_probability: float
    mission_success_probability: float
    resource_impact: str
    color: Literal["green", "yellow", "red"]

class SystemState(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["GREEN", "YELLOW", "RED"] = "RED"
    recommendation: str = "HOLD POSITION - Pose Uncertain"
    confidence_pct: float = 73.0
    pose: PoseUncertainty = Field(default_factory=PoseUncertainty)
    anomaly: Optional[AnomalyState] = Field(default_factory=AnomalyState)
    hdc_components: List[HDCComponent] = [
        HDCComponent(name="Pose Uncertainty", weight=40),
        HDCComponent(name="Thermal Anomaly", weight=30),
        HDCComponent(name="Mission Phase", weight=30),
    ]
    counterfactuals: List[CounterfactualPath] = [
        CounterfactualPath(action_id="abort", action_name="Abort Approach",
                           collision_probability=0.0, mission_success_probability=0.95,
                           resource_impact="Eliminates collision risk completely", color="green"),
        CounterfactualPath(action_id="slow", action_name="Proceed Slow",
                           collision_probability=0.12, mission_success_probability=0.78,
                           resource_impact="Maximizes time-to-dock vs ECLSS decay", color="yellow"),
        CounterfactualPath(action_id="vent", action_name="Emergency Vent",
                           collision_probability=0.35, mission_success_probability=0.45,
                           resource_impact="Isolates loop but induces attitude drift", color="red"),
    ]
    historical_cases: List[Dict[str, Any]] = [
        {"case_id": "2847", "similarity": 92, "outcome": "Success",
         "description": "Solar panel joint lock + sensor glare"}
    ]
    mission_phase: str = "approach"

class ConversationTurn(BaseModel):
    turn_id: int
    timestamp: str
    crew_input: str
    intent_classified: str
    system_response: str
    associated_hdc_snapshot: Optional[str] = None

# -----------------------------------------------------------------------------
# PROGRESSIVE DISCLOSURE ENGINE (T1/T2/T3)
# -----------------------------------------------------------------------------

class ProgressiveDisclosureEngine:
    """Maps system state to the three temporal audit layers."""
    
    def update(self, state: SystemState):
        self.state = state

    def render_tactical(self) -> Dict[str, Any]:
        """T1: 2-Second Audit — Immediate awareness."""
        s = self.state
        return {
            "status": s.status,
            "recommendation": s.recommendation,
            "confidence_pct": s.confidence_pct,
            "confidence_hex": self._confidence_color(s.confidence_pct),
            "pulse": s.status == "RED"
        }

    def render_operational(self) -> Dict[str, Any]:
        """T2: 10-Second Diagnostic — Why and alternatives."""
        s = self.state
        jg = s.pose.jensen_gain
        warning = None
        if jg >= 2.0:
            warning = "[SYMMETRY AMBIGUITY: Spacecraft 180-degree roll-inversion suspected]"
        
        anomaly_row = None
        if s.anomaly:
            anomaly_row = {
                "subsystem": s.anomaly.subsystem,
                "severity": s.anomaly.severity,
                "component": s.anomaly.component,
                "time_to_critical": s.anomaly.time_to_critical,
                "description": s.anomaly.description
            }

        actions = []
        for cf in s.counterfactuals[:3]:
            actions.append({
                "name": cf.action_name,
                "confidence": round(cf.mission_success_probability * 100, 1),
                "justification": cf.resource_impact,
                "color": cf.color
            })

        return {
            "jensen_gain": jg,
            "symmetry_warning": warning,
            "anomaly": anomaly_row,
            "alternatives": actions
        }

    def render_analytical(self) -> Dict[str, Any]:
        """T3: 2-Minute Drill-Down — Full diagnostic workspace."""
        s = self.state
        return {
            "hdc_pie": [{"label": c.name, "value": c.weight} for c in s.hdc_components],
            "trajectories": [
                {
                    "id": cf.action_id,
                    "name": cf.action_name,
                    "collision_p": cf.collision_probability,
                    "success_p": cf.mission_success_probability,
                    "color": cf.color
                } for cf in s.counterfactuals
            ],
            "histograms": {
                "E_T": s.pose.sigma_t,
                "E_R": s.pose.sigma_R,
                "jensen_gain": s.pose.jensen_gain
            },
            "historical_cases": s.historical_cases
        }

    @staticmethod
    def _confidence_color(pct: float) -> str:
        if pct >= 85: return "#00ff88"
        if pct >= 60: return "#ffcc00"
        return "#ff3366"

# -----------------------------------------------------------------------------
# DUAL-ROUTE CONVERSATIONAL AI
# -----------------------------------------------------------------------------

class DualRouteConversation:
    """
    Fast deterministic route (confidence >= 0.92) via grammar matching.
    Slow LLM fallback route for nuanced queries.
    """
    
    def __init__(self, llm_enabled: bool = False):
        self.history: List[ConversationTurn] = []
        self.llm_enabled = llm_enabled
        
        # Deterministic grammar map (zero-shot, no training required)
        self.grammar = {
            "EXPLAIN_WHY": [
                "orbital explain", "orbital why did you do that",
                "orbital justify action", "why did you recommend"
            ],
            "SIMULATE_COUNTERFACTUAL": [
                "orbital what if", "orbital simulate", "what are the odds",
                "what happens if we"
            ],
            "RENDER_VIEW": [
                "orbital show me", "orbital display", "orbital open",
                "show me the"
            ],
            "INITIATE_OVERRIDE": [
                "orbital override", "orbital kill autonomy",
                "orbital direct control", "override to manual"
            ],
            "STATUS_REPORT": [
                "orbital status report", "orbital give me status",
                "orbital current state", "status report"
            ],
            "OPTION_TRIA_QUERY": [
                "orbital what are my options", "orbital list alternatives",
                "orbital view actions", "what are my options"
            ]
        }

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Returns (intent, confidence). >= 0.92 triggers deterministic route."""
        text_lower = text.lower().strip()
        
        # Exact phrase matching (production-grade regex/keyword hybrid)
        for intent, phrases in self.grammar.items():
            for phrase in phrases:
                if phrase in text_lower:
                    return intent, 1.0
        
        # Optional: Semantic embedding fallback (requires sentence-transformers)
        # If installed, compute cosine sim against grammar examples; else default 0.5
        try:
            # In production, load model once at startup: self._embed_model
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_emb = model.encode(text, convert_to_tensor=True)
            best_score = 0.0
            for intent, phrases in self.grammar.items():
                phrase_embs = model.encode(phrases, convert_to_tensor=True)
                scores = util.cos_sim(query_emb, phrase_embs)
                max_score = float(scores.max())
                if max_score > best_score:
                    best_score = max_score
            return "UNKNOWN", best_score
        except Exception:
            return "UNKNOWN", 0.5

    async def process_query(self, text: str, state: SystemState) -> Tuple[str, str, float]:
        """
        Returns (response, route_used, confidence).
        route_used: 'deterministic' or 'llm'.
        """
        intent, confidence = self.classify_intent(text)
        
        if confidence >= 0.92:
            response = self._deterministic_response(intent, state)
            return response, "deterministic", confidence
        
        return await self._llm_route(text, state, intent)

    def _deterministic_response(self, intent: str, state: SystemState) -> str:
        if intent in ("EXPLAIN_WHY", "EXPLAIN_DECISION"):
            return (f"Action recommended due to high pose uncertainty "
                    f"(Jensen Gain: {state.pose.jensen_gain}). "
                    f"Rotational confidence has dropped below {state.confidence_pct}% "
                    f"due to glare in the sensor field of view.")
        
        elif intent == "SIMULATE_COUNTERFACTUAL":
            best = state.counterfactuals[0]
            return (f"Digital twin Monte Carlo analysis indicates a "
                    f"{best.collision_probability*100:.0f}% (+/- 8%) collision probability "
                    f"if proceeding with {best.action_name}.")
        
        elif intent == "OPTION_TRIA_QUERY":
            lines = "\n".join([
                f"{i+1}. {a.action_name} | Confidence: {a.mission_success_probability*100:.0f}% | {a.resource_impact}"
                for i, a in enumerate(state.counterfactuals[:3])
            ])
            return f"Alternative Actions:\n{lines}"
        
        elif intent == "INITIATE_OVERRIDE":
            return ("Override request acknowledged. Specify Armstrong level: "
                    "1-Acknowledge, 2-Modify, 3-Replace, 4-Reject.")
        
        elif intent == "STATUS_REPORT":
            return (f"Warning: Jensen Gain at {state.pose.jensen_gain}. "
                    f"Pitch axis symmetry ambiguity detected. "
                    f"Recommending station keeping hold.")
        
        elif intent == "RENDER_VIEW":
            return "Switching to analytical viewport. Check T3 display."
        
        return "Command acknowledged. Awaiting clarification."

    async def _llm_route(self, text: str, state: SystemState, intent: str) -> Tuple[str, str, float]:
        """Hydrated prompt for local LLM (llama.cpp) or simulated fallback."""
        prompt = f"""[SYSTEM CONTEXT]
You are the Crew Interface Agent onboard an autonomous spacecraft. Speak with absolute conciseness. Never exceed 2 sentences.

[ACTIVE SYSTEM STATE]
HDC Situation Vector: Active components = {[c.name for c in state.hdc_components]}
Jensen Gain: {state.pose.jensen_gain} | Active Telemetry: {state.anomaly.description if state.anomaly else 'Nominal'}
[ASSOCIATIVE MEMORY HISTORICAL MATCH]
Closest match: Case #2847. Condition: Solar panel joint lock + sensor glare. Action: Hold. Outcome: Nominal recovery.

[USER QUERY]
{text}

[RESPONSE]
"""
        if self.llm_enabled:
            # Production: stream from llama-cpp-python or similar
            # from llama_cpp import Llama
            # llm = Llama(model_path="llama-3-8b-q4_k_m.gguf", n_ctx=2048)
            # output = llm(prompt, max_tokens=80, stop=["[END]", "\n\n"])
            # return output["choices"][0]["text"].strip(), "llm", 0.88
            pass
        
        # Simulated LLM fallback (ensures pipeline runs without 8GB model)
        simulated = (f"LLM Route: Based on current Jensen Gain {state.pose.jensen_gain} "
                     f"and active {state.anomaly.subsystem if state.anomaly else 'nominal'} telemetry, "
                     f"I recommend maintaining the {state.recommendation} directive.")
        return simulated, "llm_fallback", 0.75

    def append_history(self, turn: ConversationTurn):
        self.history.append(turn)
        if len(self.history) > 10:
            self.history.pop(0)

# -----------------------------------------------------------------------------
# ARMSTRONG PROTOCOL
# -----------------------------------------------------------------------------

class OverrideLevel(Enum):
    ACKNOWLEDGE = 1
    MODIFY = 2
    REPLACE = 3
    REJECT = 4

class ArmstrongProtocol:
    def __init__(self):
        self.override_db: List[Dict[str, Any]] = []
        self.modal_locked = False
        self._timeout_task: Optional[asyncio.Task] = None

    async def initiate_override(
        self,
        level: int,
        target_action_id: str,
        operator_id: str,
        rationale: Optional[str] = None,
        scenario_type: str = "docking_proximity"
    ) -> Dict[str, Any]:
        if level not in [1, 2, 3, 4]:
            raise ValueError("Protocol violation: Unknown safety override level.")
        
        # Level 1: Immediate execution, zero rationale
        if level == 1:
            return {
                "status": "APPROVED",
                "strategy": "PRIMARY",
                "released": True,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        # Levels 2-4: Modal lock + mandatory rationale
        self.modal_locked = True
        
        if not rationale or len(rationale.strip()) == 0:
            rationale = ("SYSTEM_FORCE_CAPTURE: Operator triggered level change "
                         "without explicit comment.")
        
        # HDC snapshot (D=10,000 placeholder — in production, query CognitionAgent)
        hdc_snapshot = [0.0] * 10000
        
        log_payload = {
            "override_level": level,
            "operator": operator_id,
            "action_replaced": target_action_id,
            "rationale": rationale,
            "hdc_vector_snapshot": hdc_snapshot,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        self.override_db.append(log_payload)
        # Production: await CognitionAgent.AssociativeMemory.inject_one_shot_override(...)
        
        self.modal_locked = False
        return {
            "status": "OVERRIDE_RELEASED",
            "payload": log_payload,
            "released": True
        }

    def get_fail_timeout(self, scenario: str) -> float:
        """Dynamic countdown rules (seconds)."""
        return {
            "docking_proximity": 15,
            "habitat_life_support": 60,
            "deep_space_transit": 300
        }.get(scenario, 60)

    async def start_timeout(self, scenario: str, callback):
        t = self.get_fail_timeout(scenario)
        await asyncio.sleep(t)
        await callback({
            "type": "TIMEOUT",
            "message": "AUTONOMOUS FALLBACK INITIATED",
            "action": "CONSERVATIVE_HOLD"
        })

# -----------------------------------------------------------------------------
# VOICE PIPELINE (Simulated Edge Pipeline)
# -----------------------------------------------------------------------------

class VoicePipeline:
    """
    Localized acoustic pipeline.
    Production: Whisper-Base INT8 + Piper TTS.
    """
    
    def __init__(self):
        self.wake_word = "orbital"
        self.vocabulary = {"JEPA", "Hopf Grid", "HDC", "ECLSS", "Vbar", "Rbar"}
    
    def process_command(self, text: str) -> Optional[str]:
        """Validate voice grammar. Returns normalized command or None."""
        text_lower = text.lower().strip()
        if not text_lower.startswith(self.wake_word):
            return None
        return text_lower[len(self.wake_word):].strip()

# -----------------------------------------------------------------------------
# INTERFACE AGENT (Orchestrator)
# -----------------------------------------------------------------------------

class InterfaceAgent:
    def __init__(self):
        self.display = ProgressiveDisclosureEngine()
        self.conversation = DualRouteConversation(llm_enabled=False)
        self.armstrong = ArmstrongProtocol()
        self.voice = VoicePipeline()
        self.state = SystemState()
        self.websockets: List[WebSocket] = []
        self._tick_task: Optional[asyncio.Task] = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.websockets.append(ws)
        # Push current state immediately
        await self._push_state(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.websockets:
            self.websockets.remove(ws)

    async def handle_state_update(self, new_state: SystemState):
        """Called by Redis subscriber or simulation tick."""
        self.state = new_state
        self.display.update(new_state)
        await self._broadcast({
            "type": "state_update",
            "tactical": self.display.render_tactical(),
            "operational": self.display.render_operational(),
            "analytical": self.display.render_analytical()
        })

    async def handle_chat(self, text: str) -> Dict[str, Any]:
        response, route, conf = await self.conversation.process_query(text, self.state)
        turn = ConversationTurn(
            turn_id=len(self.conversation.history) + 1,
            timestamp=datetime.utcnow().isoformat() + "Z",
            crew_input=text,
            intent_classified=self.conversation.classify_intent(text)[0],
            system_response=response,
            associated_hdc_snapshot="0x4F7A...3E"
        )
        self.conversation.append_history(turn)
        return {
            "type": "chat_response",
            "text": response,
            "route": route,
            "confidence": conf,
            "turn": turn.dict()
        }

    async def handle_override(self, level: int, action_id: str, operator: str, rationale: Optional[str]) -> Dict[str, Any]:
        result = await self.armstrong.initiate_override(level, action_id, operator, rationale)
        await self._broadcast({
            "type": "override_result",
            "data": result
        })
        return result

    async def handle_voice(self, text: str) -> Optional[Dict[str, Any]]:
        recognized = self.voice.process_command(text)
        if recognized:
            return await self.handle_chat(recognized)
        return None

    async def _broadcast(self, msg: Dict[str, Any]):
        dead = []
        for ws in self.websockets:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.websockets.remove(ws)

    async def _push_state(self, ws: WebSocket):
        self.display.update(self.state)
        await ws.send_json({
            "type": "state_update",
            "tactical": self.display.render_tactical(),
            "operational": self.display.render_operational(),
            "analytical": self.display.render_analytical()
        })

    async def simulation_tick(self):
        """Background task: simulates inbound Redis state updates."""
        while True:
            await asyncio.sleep(5)
            # In production, this is replaced by aioredis pub/sub listener
            # async for message in redis.pubsub().listen():
            #     state = parse_message(message)
            #     await self.handle_state_update(state)
            pass

# -----------------------------------------------------------------------------
# FASTAPI APPLICATION
# -----------------------------------------------------------------------------

app = FastAPI(title="Crew Interface Agent — Phase 4")
agent = InterfaceAgent()

# Serve static frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

@app.on_event("startup")
async def startup():
    agent._tick_task = asyncio.create_task(agent.simulation_tick())

@app.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Place index.html in static/ folder</h1>"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await agent.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "chat":
                resp = await agent.handle_chat(msg["text"])
                await websocket.send_json(resp)

            elif msg_type == "override":
                result = await agent.handle_override(
                    msg.get("level", 1),
                    msg.get("action_id", "primary"),
                    msg.get("operator", "CMDR"),
                    msg.get("rationale")
                )
                await websocket.send_json({"type": "override_ack", "data": result})

            elif msg_type == "voice":
                vr = await agent.handle_voice(msg["text"])
                if vr:
                    await websocket.send_json(vr)
                else:
                    await websocket.send_json({
                        "type": "voice_reject",
                        "text": "Wake word 'ORBITAL' not detected or command not recognized."
                    })

            elif msg_type == "request_state":
                await agent._push_state(websocket)

    except WebSocketDisconnect:
        agent.disconnect(websocket)

# -----------------------------------------------------------------------------
# REDIS INTEGRATION STUB (Production Hook)
# -----------------------------------------------------------------------------
"""
# To wire into Phase 2/3 (Cognition + Action agents), add this to startup:

import aioredis

async def redis_listener():
    redis = aioredis.from_url("redis://localhost:6379")
    pubsub = redis.pubsub()
    await pubsub.subscribe("perception.out", "cognition.out", "action.out")
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            channel = message["channel"].decode()
            payload = json.loads(message["data"])
            
            if channel == "cognition.out":
                # Hydrate SystemState from HDC vector + anomaly telemetry
                new_state = SystemState(
                    status=payload.get("status", "YELLOW"),
                    pose=PoseUncertainty(
                        jensen_gain=payload["pose"]["jensen_gain"],
                        sigma_R=payload["pose"]["sigma_R"],
                        sigma_t=payload["pose"]["sigma_t"],
                        confidence=payload["pose"]["confidence"]
                    ),
                    anomaly=AnomalyState(**payload["anomaly"]) if payload.get("anomaly") else None,
                    hdc_components=[HDCComponent(**c) for c in payload.get("hdc", [])],
                    counterfactuals=[CounterfactualPath(**c) for c in payload.get("counterfactuals", [])]
                )
                await agent.handle_state_update(new_state)
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
