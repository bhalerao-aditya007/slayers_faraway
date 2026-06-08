"""
phase2_hdc_cognition.py
Hyperdimensional Cognition Layer for Spacecraft Autonomy
Dimensionality: D=10,000 | Bipolar HDC Algebra
Integrates with: Phase 1 (Perception) -> Phase 3 (Action) / Phase 5 (Orchestrator)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import json
import base64
import threading
from collections import defaultdict
from datetime import datetime, timezone

# ==============================================================================
# CONFIGURATION & ENUMS
# ==============================================================================

class DistanceBin(str, Enum):
    VERY_CLOSE = "very_close"
    CLOSE = "close"
    MEDIUM = "medium"
    FAR = "far"

class OrientationBin(str, Enum):
    ALIGNED = "aligned"
    OFFSET_45 = "offset_45"
    OFFSET_90 = "offset_90"
    INVERTED = "inverted"

class VelocityBin(str, Enum):
    STATIC = "static"
    DRIFTING = "drifting"
    APPROACHING = "approaching"
    RECEDING = "receding"

class HealthState(str, Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    CRITICAL = "critical"

class MissionPhase(str, Enum):
    APPROACH = "approach"
    DOCKING = "docking"
    BERTHING = "berthing"
    MAINTENANCE = "maintenance"

class JensenGainBin(str, Enum):
    LOW = "low"       # < 0.5
    MODERATE = "moderate"  # 0.5 - 2.0
    HIGH = "high"     # >= 2.0

# ==============================================================================
# INPUT DATACLASSES (Phase 1 -> Phase 2 Interface)
# ==============================================================================

@dataclass
class PoseEstimate:
    """Direct input from Phase 1 Perception Agent."""
    translation: np.ndarray          # 3D vector [x, y, z] in meters
    rotation: Optional[np.ndarray] = None  # 3x3 rotation matrix or quaternion
    confidence: str = "high"         # "high", "moderate", "low", "critical"
    jensen_gain: float = 0.0
    sigma_t: float = 0.0             # translation uncertainty
    sigma_R: float = 0.0             # rotation uncertainty

@dataclass
class Telemetry:
    """Habitat / spacecraft telemetry stream."""
    o2_level: float = 100.0          # %
    co2_level: float = 0.0           # %
    pressure_kpa: float = 101.3
    temp_c: float = 22.0
    humidity_pct: float = 50.0
    battery_pct: float = 100.0
    solar_generation_pct: float = 100.0
    load_distribution_pct: float = 50.0
    radiator_efficiency_pct: float = 100.0
    coolant_flow_lpm: float = 10.0
    heat_load_kw: float = 5.0

@dataclass
class AnomalyReport:
    """Anomaly descriptor from subsystem monitors."""
    failure_type: str = "none"
    severity: str = "nominal"        # "nominal", "degraded", "critical", "failed"
    propagation_risk: str = "low"   # "low", "medium", "high"

@dataclass
class DomainContext:
    lighting: str = "nominal"
    background: str = "deep_space"

# ==============================================================================
# CORE HDC ENGINE
# ==============================================================================

class HDCEngine:
    """
    Fundamental HDC algebra engine.
    Uses bipolar vectors {-1, +1}^D for mathematical cleanliness.
    """
    def __init__(self, dim: int = 10000, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[str, np.ndarray] = {}
        
    def generate_random_vector(self, label: Optional[str] = None) -> np.ndarray:
        """Generate a random bipolar hypervector. Cache if label provided."""
        if label and label in self._cache:
            return self._cache[label]
        vec = self.rng.choice(np.array([-1.0, 1.0]), size=self.dim).astype(np.float32)
        if label:
            self._cache[label] = vec
        return vec
    
    def bind(self, *vectors: np.ndarray) -> np.ndarray:
        """
        Binding (⊗): Element-wise multiplication.
        Self-inverse: bind(bind(a, b), b) == a
        """
        if not vectors:
            raise ValueError("At least one vector required for binding")
        result = vectors[0]
        for v in vectors[1:]:
            result = np.multiply(result, v)
        return result
    
    def bundle(self, vectors: List[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        Bundling (⊕): Element-wise addition with optional normalization.
        """
        if not vectors:
            return self.generate_random_vector("zero") * 0
        summed = np.sum(vectors, axis=0)
        if normalize:
            # Map zeros to +1 to preserve bipolar property
            return np.where(summed >= 0, 1.0, -1.0).astype(np.float32)
        return summed.astype(np.float32)
    
    def permute(self, v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Permutation (ρ): Cyclic shift. Non-commutative, distance-preserving."""
        return np.roll(v, shifts).astype(np.float32)
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def serialize(self, v: np.ndarray) -> str:
        """Serialize hypervector to base64 string for Redis/JSON messaging."""
        return base64.b64encode(v.astype(np.float32).tobytes()).decode('utf-8')
    
    def deserialize(self, s: str) -> np.ndarray:
        """Deserialize hypervector from base64 string."""
        bytes_data = base64.b64decode(s.encode('utf-8'))
        return np.frombuffer(bytes_data, dtype=np.float32)

# ==============================================================================
# ITEM MEMORY (IM)
# ==============================================================================

class ItemMemory:
    """
    Static dictionary mapping atomic symbols to pseudo-orthogonal hypervectors.
    Auto-generates vectors for unknown symbols on first access.
    """
    
    ATOMIC_SYMBOLS = [
        # Spacecraft Components
        "solar_panel", "antenna", "docking_port",
        # Anomaly Types
        "thermal_failure", "power_loss", "pressure_drop", "none",
        # Mission Phases
        "approach", "docking", "berthing", "maintenance",
        # Confidence Levels
        "high", "moderate", "low", "critical",
        # Health States
        "nominal", "degraded", "failed",
        # Distance
        "very_close", "close", "medium", "far",
        # Orientation
        "aligned", "offset_45", "offset_90", "inverted",
        # Velocity
        "static", "drifting", "approaching", "receding",
        # Jensen Gain
        "jensen_low", "jensen_moderate", "jensen_high",
        # Domain
        "lighting_nominal", "lighting_glare", "lighting_dark",
        "background_deep_space", "background_earth", "background_station",
        # Severity / Risk
        "severity_nominal", "severity_degraded", "severity_critical",
        "risk_low", "risk_medium", "risk_high",
        # Subsystems
        "o2", "co2", "pressure", "temperature", "humidity",
        "battery", "solar", "load",
        "radiator", "coolant", "heat",
    ]
    
    def __init__(self, engine: HDCEngine):
        self.engine = engine
        self.memory: Dict[str, np.ndarray] = {}
        self._init_core_symbols()
        
    def _init_core_symbols(self):
        for symbol in self.ATOMIC_SYMBOLS:
            self.memory[symbol] = self.engine.generate_random_vector(symbol)
            
    def get(self, symbol: str) -> np.ndarray:
        """Retrieve or auto-generate a hypervector for any symbol."""
        if symbol not in self.memory:
            self.memory[symbol] = self.engine.generate_random_vector(symbol)
        return self.memory[symbol].copy()

# ==============================================================================
# PHASE 2.2: POSE-TO-SYMBOL ENCODER
# ==============================================================================

class PoseEncoder:
    """
    Encodes continuous pose estimates into hyperdimensional symbols.
    """
    
    def __init__(self, item_memory: ItemMemory, engine: HDCEngine):
        self.im = item_memory
        self.engine = engine
        
    def _discretize_distance(self, t: np.ndarray) -> str:
        dist = float(np.linalg.norm(t))
        if dist < 5.0:
            return DistanceBin.VERY_CLOSE.value
        elif dist < 50.0:
            return DistanceBin.CLOSE.value
        elif dist < 500.0:
            return DistanceBin.MEDIUM.value
        return DistanceBin.FAR.value
    
    def _discretize_orientation(self, R: Optional[np.ndarray]) -> str:
        if R is None:
            return OrientationBin.ALIGNED.value
        # Simplified: compute angular deviation from identity
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
        if angle < 15:
            return OrientationBin.ALIGNED.value
        elif angle < 60:
            return OrientationBin.OFFSET_45.value
        elif angle < 150:
            return OrientationBin.OFFSET_90.value
        return OrientationBin.INVERTED.value
    
    def _discretize_velocity(self, velocity_mps: float = 0.0) -> str:
        # Default static if no velocity provided; user can override
        if abs(velocity_mps) < 0.01:
            return VelocityBin.STATIC.value
        elif abs(velocity_mps) < 0.1:
            return VelocityBin.DRIFTING.value
        elif velocity_mps < 0:
            return VelocityBin.APPROACHING.value
        return VelocityBin.RECEDING.value
    
    def _discretize_jensen_gain(self, g: float) -> str:
        if g < 0.5:
            return "jensen_low"
        elif g < 2.0:
            return "jensen_moderate"
        return "jensen_high"
    
    def encode(self, pose: PoseEstimate, velocity_mps: float = 0.0,
               domain: Optional[DomainContext] = None) -> Dict[str, np.ndarray]:
        """
        Returns:
            v_pose: bound spatial hypervector
            v_uncertainty: bound confidence hypervector
            v_domain: bound environmental hypervector
        """
        # Discretize
        d_bin = self._discretize_distance(pose.translation)
        o_bin = self._discretize_orientation(pose.rotation)
        v_bin = self._discretize_velocity(velocity_mps)
        j_bin = self._discretize_jensen_gain(pose.jensen_gain)
        
        # Retrieve base vectors
        v_dist = self.im.get(d_bin)
        v_ori = self.im.get(o_bin)
        v_vel = self.im.get(v_bin)
        v_conf = self.im.get(pose.confidence)
        v_jensen = self.im.get(j_bin)
        
        # Compose per document formulas
        v_pose = self.engine.bind(
            v_dist,
            self.engine.permute(v_ori, 1),
            self.engine.permute(v_vel, 2)
        )
        
        v_uncertainty = self.engine.bind(v_conf, v_jensen)
        
        # Domain context
        if domain is None:
            domain = DomainContext()
        v_light = self.im.get(f"lighting_{domain.lighting}")
        v_bg = self.im.get(f"background_{domain.background}")
        v_domain = self.engine.bind(v_light, v_bg)
        
        return {
            "v_pose": v_pose,
            "v_uncertainty": v_uncertainty,
            "v_domain": v_domain
        }

# ==============================================================================
# PHASE 2.3: ANOMALY STATE VECTOR ENCODER
# ==============================================================================

class AnomalyEncoder:
    """
    Encodes habitat telemetry and anomaly reports into hypervectors.
    """
    
    def __init__(self, item_memory: ItemMemory, engine: HDCEngine):
        self.im = item_memory
        self.engine = engine
        
    def _discretize(self, value: float, thresholds: List[float], 
                    labels: List[str]) -> str:
        for thresh, label in zip(thresholds, labels):
            if value <= thresh:
                return label
        return labels[-1]
    
    def encode_telemetry(self, tel: Telemetry) -> Dict[str, np.ndarray]:
        """
        Discretize continuous telemetry into symbolic health states and encode.
        Returns subsystem hypervectors.
        """
        # Life Support thresholds (example values; configurable)
        o2_state = self._discretize(tel.o2_level, [85, 90, 95], 
                                    ["failed", "critical", "degraded", "nominal"])
        co2_state = self._discretize(tel.co2_level, [0.5, 1.0, 2.0],
                                     ["nominal", "degraded", "critical", "failed"])
        press_state = self._discretize(abs(tel.pressure_kpa - 101.3), [5, 10, 15],
                                       ["nominal", "degraded", "critical", "failed"])
        
        # Power
        batt_state = self._discretize(tel.battery_pct, [20, 40, 60],
                                      ["failed", "critical", "degraded", "nominal"])
        solar_state = self._discretize(tel.solar_generation_pct, [20, 50, 80],
                                       ["failed", "critical", "degraded", "nominal"])
        
        # Thermal
        rad_state = self._discretize(tel.radiator_efficiency_pct, [30, 50, 80],
                                     ["failed", "critical", "degraded", "nominal"])
        
        # Encode individual symbols
        v_o2 = self.im.get(f"o2_{o2_state}")
        v_co2 = self.im.get(f"co2_{co2_state}")
        v_press = self.im.get(f"pressure_{press_state}")
        
        # Subsystem binding per document: v_subsystem = bind(v_O2, permute(v_power), permute^2(v_thermal))
        # We use bundled subsystem representatives for the formula
        v_power = self.engine.bundle([
            self.im.get(f"battery_{batt_state}"),
            self.im.get(f"solar_{solar_state}")
        ])
        v_thermal = self.engine.bundle([
            self.im.get(f"radiator_{rad_state}"),
            self.im.get(f"coolant_{self._discretize(tel.coolant_flow_lpm, [2, 5, 8], ['failed','critical','degraded','nominal'])}")
        ])
        
        v_subsystem = self.engine.bind(
            v_o2,
            self.engine.permute(v_power, 1),
            self.engine.permute(v_thermal, 2)
        )
        
        return {
            "v_subsystem": v_subsystem,
            "v_o2": v_o2,
            "v_power": v_power,
            "v_thermal": v_thermal,
            "states": {
                "o2": o2_state, "co2": co2_state, "pressure": press_state,
                "battery": batt_state, "solar": solar_state, "radiator": rad_state
            }
        }
    
    def encode_anomaly(self, anomaly: AnomalyReport) -> np.ndarray:
        """
        v_anomaly = bind(v_failure_type, v_severity, v_propagation_risk)
        """
        v_fail = self.im.get(anomaly.failure_type)
        v_sev = self.im.get(f"severity_{anomaly.severity}")
        v_risk = self.im.get(f"risk_{anomaly.propagation_risk}")
        return self.engine.bind(v_fail, v_sev, v_risk)

# ==============================================================================
# PHASE 2.4: SITUATIONAL AWARENESS & ASSOCIATIVE MEMORY
# ==============================================================================

@dataclass
class MemoryEntry:
    """Stored historical situation with outcome."""
    situation_vector: np.ndarray
    outcome: str
    success_rate: float
    action_taken: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self, engine: HDCEngine) -> Dict:
        return {
            "situation_vector_b64": engine.serialize(self.situation_vector),
            "outcome": self.outcome,
            "success_rate": self.success_rate,
            "action_taken": self.action_taken,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

class SituationalAwareness:
    """
    Binds all perception and anomaly vectors into holistic situation vector.
    Maintains Associative Memory (AM) for similarity-based retrieval.
    """
    
    def __init__(self, engine: HDCEngine, item_memory: ItemMemory,
                 similarity_threshold: float = 0.55):
        self.engine = engine
        self.im = item_memory
        self.threshold = similarity_threshold
        self.associative_memory: List[MemoryEntry] = []
        self.lock = threading.RLock()
        
    def bind_situation(self, v_pose: np.ndarray, v_anomaly: np.ndarray,
                       v_mission_phase: np.ndarray, v_uncertainty: np.ndarray) -> np.ndarray:
        """
        v_situation = v_pose ⊗ v_anomaly ⊗ v_mission_phase ⊗ v_uncertainty
        """
        return self.engine.bind(v_pose, v_anomaly, v_mission_phase, v_uncertainty)
    
    def add_memory(self, entry: MemoryEntry):
        with self.lock:
            self.associative_memory.append(entry)
            
    def search_similar(self, v_query: np.ndarray, k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """
        k-NN search via cosine similarity.
        Returns sorted list of (entry, similarity).
        """
        with self.lock:
            if not self.associative_memory:
                return []
            
            scores = []
            for entry in self.associative_memory:
                sim = self.engine.cosine_similarity(v_query, entry.situation_vector)
                scores.append((entry, sim))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]
    
    def is_known_situation(self, v_query: np.ndarray) -> Tuple[bool, float]:
        """Check if query matches known situations above threshold."""
        top_matches = self.search_similar(v_query, k=1)
        if not top_matches:
            return False, 0.0
        best_match, score = top_matches[0]
        return score > self.threshold, score

# ==============================================================================
# PHASE 2.5: ONE-SHOT ANOMALY DETECTION
# ==============================================================================

class OneShotDetector:
    """
    Detects novel situations and executes tri-tier adaptation strategy.
    """
    
    def __init__(self, situational_awareness: SituationalAwareness,
                 novelty_threshold: float = 0.45, k_neighbors: int = 5):
        self.sa = situational_awareness
        self.novelty_threshold = novelty_threshold
        self.k = k_neighbors
        
    def detect(self, v_novel: np.ndarray) -> Dict[str, Any]:
        """
        Returns:
            is_novel: bool
            max_similarity: float
            strategy: str
            nearest_cases: list of dicts
            recommendation: str
        """
        neighbors = self.sa.search_similar(v_novel, k=self.k)
        
        if not neighbors:
            max_sim = 0.0
        else:
            max_sim = neighbors[0][1]
            
        is_novel = max_sim < self.novelty_threshold
        
        if is_novel:
            strategy = "conservative"
            recommendation = "HOLD_POSITION"
        elif max_sim < 0.6:
            strategy = "exploratory"
            recommendation = "PROBE_MINIMAL_RISK"
        else:
            strategy = "historical"
            recommendation = neighbors[0][0].action_taken if neighbors else "HOLD_POSITION"
            
        nearest_cases = [
            {
                "case_id": i,
                "similarity": round(sim, 3),
                "outcome": entry.outcome,
                "success_rate": entry.success_rate,
                "action": entry.action_taken
            }
            for i, (entry, sim) in enumerate(neighbors[:3])
        ]
        
        return {
            "is_novel": is_novel,
            "max_similarity": round(max_sim, 4),
            "strategy": strategy,
            "recommendation": recommendation,
            "nearest_cases": nearest_cases,
            "human_prompt": self._generate_prompt(is_novel, nearest_cases)
        }
    
    def _generate_prompt(self, is_novel: bool, cases: List[Dict]) -> str:
        if not is_novel:
            return ""
        base = "Novel failure mode detected. "
        if cases:
            base += f"Recommendations based on closest known case: {cases[0]['action']} (similarity: {cases[0]['similarity']})."
        else:
            base += "No historical precedent. Defaulting to maximum safety protocols."
        return base

# ==============================================================================
# PHASE 2.6: EXPLAINABILITY INTERFACE
# ==============================================================================

class ExplainabilityInterface:
    """
    Decomposes HDC situation vectors into human-readable components.
    """
    
    def __init__(self, engine: HDCEngine, item_memory: ItemMemory):
        self.engine = engine
        self.im = item_memory
        
    def decompose(self, v_situation: np.ndarray, 
                  components: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        HDC decomposition via unbinding.
        For each component C, unbind all other components and measure
        similarity to C. This yields the 'influence' percentage.
        """
        total = 0.0
        raw_scores = {}
        
        names = list(components.keys())
        vectors = list(components.values())
        
        for i, (name, vec) in enumerate(components.items()):
            # Unbind all other components from situation
            others = [vectors[j] for j in range(len(vectors)) if j != i]
            if others:
                v_unbound = self.engine.bind(v_situation, *others)
            else:
                v_unbound = v_situation
            sim = self.engine.cosine_similarity(v_unbound, vec)
            # Clamp negative similarities to 0 for interpretability
            sim = max(0.0, sim)
            raw_scores[name] = sim
            total += sim
            
        if total == 0:
            return {name: 0.0 for name in names}
            
        # Normalize to percentages
        return {
            name: round((score / total) * 100, 1)
            for name, score in raw_scores.items()
        }
    
    def similarity_heatmap(self, v_situation: np.ndarray,
                          top_cases: List[Tuple[MemoryEntry, float]]) -> List[Dict]:
        """Generate heatmap data for UI rendering."""
        return [
            {
                "case_id": i,
                "similarity_pct": round(sim * 100, 1),
                "outcome": entry.outcome,
                "success_rate": entry.success_rate,
                "action": entry.action_taken,
                "timestamp": entry.timestamp
            }
            for i, (entry, sim) in enumerate(top_cases)
        ]
    
    def generate_narrative(self, decomposition: Dict[str, float],
                           top_cases: List[Tuple[MemoryEntry, float]],
                           pose_data: Dict[str, Any],
                           anomaly_data: Dict[str, Any]) -> str:
        """
        Generate natural language explanation for crew interface.
        """
        # Sort components by influence
        sorted_comp = sorted(decomposition.items(), key=lambda x: x[1], reverse=True)
        comp_str = ", ".join([f"{k}: {v}%" for k, v in sorted_comp])
        
        if top_cases:
            best = top_cases[0]
            case = best[0]
            sim = best[1]
            narrative = (
                f"System detected {anomaly_data.get('failure_type', 'anomaly')} "
                f"({round(sim*100)}% similarity to historical case). "
                f"Pose uncertainty: {pose_data.get('jensen_gain', 'N/A')} Jensen Gain. "
                f"Decision drivers: {comp_str}. "
                f"Recommended action: {case.action_taken} "
                f"(historical success rate: {case.success_rate}%)."
            )
        else:
            narrative = (
                f"System detected {anomaly_data.get('failure_type', 'anomaly')}. "
                f"Pose uncertainty: {pose_data.get('jensen_gain', 'N/A')} Jensen Gain. "
                f"No historical precedent. Decision drivers: {comp_str}. "
                f"Defaulting to conservative hold."
            )
        return narrative

# ==============================================================================
# MAIN INTEGRATION CLASS: PHASE 2 COGNITION LAYER
# ==============================================================================

class HyperdimensionalCognitionLayer:
    """
    Unified Phase 2 interface.
    Instantiate once at system startup. Thread-safe for multi-agent use.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.dim = self.config.get("dim", 10000)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.55)
        self.novelty_threshold = self.config.get("novelty_threshold", 0.45)
        
        self.engine = HDCEngine(dim=self.dim, seed=self.config.get("seed", 42))
        self.item_memory = ItemMemory(self.engine)
        self.pose_encoder = PoseEncoder(self.item_memory, self.engine)
        self.anomaly_encoder = AnomalyEncoder(self.item_memory, self.engine)
        self.situational_awareness = SituationalAwareness(
            self.engine, self.item_memory, self.similarity_threshold
        )
        self.one_shot = OneShotDetector(self.situational_awareness, self.novelty_threshold)
        self.explainer = ExplainabilityInterface(self.engine, self.item_memory)
        
        self.logger = logging.getLogger("Phase2.HDC")
        self.logger.info(f"HDC Cognition Layer initialized (D={self.dim})")
        
    def process(self, 
                pose_estimate: PoseEstimate,
                telemetry: Telemetry,
                anomaly_report: AnomalyReport,
                mission_phase: str,
                domain: Optional[DomainContext] = None,
                velocity_mps: float = 0.0) -> Dict[str, Any]:
        """
        Main Phase 2 pipeline.
        
        Input: Phase 1 pose + telemetry + anomaly + mission context
        Output: Situation vector + recommendation + explanation + metadata
        """
        # 1. Encode pose
        pose_encoded = self.pose_encoder.encode(pose_estimate, velocity_mps, domain)
        v_pose = pose_encoded["v_pose"]
        v_uncertainty = pose_encoded["v_uncertainty"]
        v_domain = pose_encoded["v_domain"]
        
        # 2. Encode anomaly & telemetry
        telem_encoded = self.anomaly_encoder.encode_telemetry(telemetry)
        v_subsystem = telem_encoded["v_subsystem"]
        v_anomaly_sig = self.anomaly_encoder.encode_anomaly(anomaly_report)
        
        # Bundle subsystem health with anomaly signature
        v_anomaly = self.engine.bundle([v_subsystem, v_anomaly_sig])
        
        # 3. Mission phase
        v_mission = self.item_memory.get(mission_phase)
        
        # 4. Bind holistic situation
        v_situation = self.situational_awareness.bind_situation(
            v_pose, v_anomaly, v_mission, v_uncertainty
        )
        
        # 5. One-shot detection & recommendation
        detection = self.one_shot.detect(v_situation)
        
        # 6. Explainability
        components = {
            "pose": v_pose,
            "anomaly": v_anomaly,
            "mission_phase": v_mission,
            "uncertainty": v_uncertainty
        }
        decomposition = self.explainer.decompose(v_situation, components)
        
        # Retrieve similar cases for narrative
        similar_cases = self.situational_awareness.search_similar(v_situation, k=5)
        heatmap = self.explainer.similarity_heatmap(v_situation, similar_cases)
        
        pose_meta = {
            "jensen_gain": pose_estimate.jensen_gain,
            "confidence": pose_estimate.confidence,
            "distance_m": float(np.linalg.norm(pose_estimate.translation))
        }
        anomaly_meta = {
            "failure_type": anomaly_report.failure_type,
            "severity": anomaly_report.severity,
            "telemetry_states": telem_encoded["states"]
        }
        
        narrative = self.explainer.generate_narrative(
            decomposition, similar_cases, pose_meta, anomaly_meta
        )
        
        # 7. Package output for Phase 3/5
        result = {
            "agent_id": "cognition",
            "message_type": "situation_assessment",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "situation_vector_b64": self.engine.serialize(v_situation),
                "is_novel": detection["is_novel"],
                "max_similarity": detection["max_similarity"],
                "strategy": detection["strategy"],
                "recommended_action": detection["recommendation"],
                "nearest_cases": detection["nearest_cases"],
                "human_prompt": detection["human_prompt"],
                "explanation": {
                    "narrative": narrative,
                    "component_breakdown": decomposition,
                    "similarity_heatmap": heatmap
                },
                "pose_meta": pose_meta,
                "anomaly_meta": anomaly_meta
            }
        }
        
        self.logger.info(
            f"Processed situation: strategy={detection['strategy']}, "
            f"action={detection['recommendation']}, novel={detection['is_novel']}"
        )
        return result
    
    def learn_outcome(self, situation_vector_b64: str, action_taken: str,
                      outcome: str, success_rate: float, metadata: Optional[Dict] = None):
        """
        Online learning: store verified outcome in Associative Memory.
        Call this after human override or action completion.
        """
        v_sit = self.engine.deserialize(situation_vector_b64)
        entry = MemoryEntry(
            situation_vector=v_sit,
            outcome=outcome,
            success_rate=success_rate,
            action_taken=action_taken,
            metadata=metadata or {}
        )
        self.situational_awareness.add_memory(entry)
        self.logger.info(f"Learned outcome: {outcome} for action {action_taken}")

# ==============================================================================
# DEMONSTRATION & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
    
    # Initialize Phase 2
    hdc = HyperdimensionalCognitionLayer(config={
        "dim": 10000,
        "similarity_threshold": 0.55,
        "novelty_threshold": 0.45
    })
    
    # Simulate Phase 1 input: uncertain pose + thermal anomaly
    pose = PoseEstimate(
        translation=np.array([12.0, 0.0, 0.0]),  # 12m away -> "close"
        rotation=np.eye(3),  # aligned
        confidence="low",
        jensen_gain=2.8,  # HIGH uncertainty
        sigma_t=0.05,
        sigma_R=0.12
    )
    
    telemetry = Telemetry(
        o2_level=94.0,      # degraded
        co2_level=0.8,      # degraded
        pressure_kpa=101.3,
        battery_pct=85.0,
        radiator_efficiency_pct=45.0,  # critical
        coolant_flow_lpm=3.0           # degraded
    )
    
    anomaly = AnomalyReport(
        failure_type="thermal_failure",
        severity="critical",
        propagation_risk="high"
    )
    
    # Run pipeline
    result = hdc.process(
        pose_estimate=pose,
        telemetry=telemetry,
        anomaly_report=anomaly,
        mission_phase="docking",
        domain=DomainContext(lighting="glare", background="station"),
        velocity_mps=-0.5  # approaching
    )
    
    print("\n" + "="*60)
    print("PHASE 2 OUTPUT (Ready for Phase 3/5 Consumption)")
    print("="*60)
    print(json.dumps(result, indent=2, default=str))
    
    # Simulate learning from a historical case to show AM functionality
    print("\n" + "="*60)
    print("SEEDING ASSOCIATIVE MEMORY WITH HISTORICAL CASE")
    print("="*60)
    
    # Create a similar situation vector manually for demo
    similar_pose = PoseEstimate(
        translation=np.array([11.0, 0.0, 0.0]),
        confidence="low",
        jensen_gain=2.5
    )
    similar_telemetry = Telemetry(
        o2_level=93.0,
        radiator_efficiency_pct=40.0
    )
    similar_anomaly = AnomalyReport(
        failure_type="thermal_failure",
        severity="critical",
        propagation_risk="high"
    )
    
    similar_result = hdc.process(
        pose_estimate=similar_pose,
        telemetry=similar_telemetry,
        anomaly_report=similar_anomaly,
        mission_phase="docking"
    )
    
    # Learn this as a successful historical case
    hdc.learn_outcome(
        situation_vector_b64=similar_result["payload"]["situation_vector_b64"],
        action_taken="HOLD_POSITION",
        outcome="success",
        success_rate=91.0,
        metadata={"case_id": 2847, "notes": "Thermal leak during docking"}
    )
    
    # Re-run original query to see similarity jump
    print("\n" + "="*60)
    print("RE-RUNNING AFTER MEMORY SEED")
    print("="*60)
    result2 = hdc.process(
        pose_estimate=pose,
        telemetry=telemetry,
        anomaly_report=anomaly,
        mission_phase="docking",
        domain=DomainContext(lighting="glare", background="station"),
        velocity_mps=-0.5
    )
    print(json.dumps(result2["payload"]["explanation"], indent=2, default=str))
    print(f"\nRecommended action: {result2['payload']['recommended_action']}")
    print(f"Is novel: {result2['payload']['is_novel']}")