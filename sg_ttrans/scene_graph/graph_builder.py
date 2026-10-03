"""
Dynamic Spatio-Temporal Scene Graph builder.
"""

from typing import Dict, List
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.scene_graph.ttc_calculator import compute_ttc, compute_hazard_weight
from sg_ttrans.types import SceneGraph, SceneNode

class DynamicSceneGraphBuilder:
    """Builds an exterior dynamic interaction scene graph from raw actor detections."""
    def __init__(self, cfg: STHGSTConfig = None):
        self.cfg = cfg or STHGSTConfig()

    def build(self, raw_detections: List[Dict]) -> SceneGraph:
        nodes = []
        positions = []
        for det in raw_detections:
            dist = float(det.get("distance", 15.0))
            rel_vel = float(det.get("rel_vel", 0.0))
            ttc = compute_ttc(dist, rel_vel)
            h_weight = compute_hazard_weight(ttc, self.cfg)
            bbox = det.get("bbox", (0.0, 0.0, 0.1, 0.1))
            
            node = SceneNode(
                track_id=int(det.get("id", len(nodes))),
                class_id=int(det.get("class_id", 0)),
                bbox=bbox,
                velocity=(float(det.get("vx", 0.0)), float(rel_vel)),
                ttc=ttc,
                hazard_weight=h_weight,
            )
            nodes.append(node)
            positions.append([bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0])

        n_nodes = len(nodes)
        if n_nodes < 2:
            return SceneGraph(
                nodes=nodes,
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_attr=np.zeros((0, 2), dtype=np.float32),
            )

        # Build fully connected directed edges between distinct nodes
        edge_sources = []
        edge_targets = []
        edge_attrs = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_sources.append(i)
                    edge_targets.append(j)
                    p_i = np.array(positions[i], dtype=np.float32)
                    p_j = np.array(positions[j], dtype=np.float32)
                    dist = float(np.linalg.norm(p_i - p_j))
                    edge_attrs.append([dist, 1.0 / (dist + 1e-4)])

        return SceneGraph(
            nodes=nodes,
            edge_index=np.array([edge_sources, edge_targets], dtype=np.int64),
            edge_attr=np.array(edge_attrs, dtype=np.float32),
        )
