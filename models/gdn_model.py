"""
GDN Model and Loss Functions

Graph Deviation Network (GDN) for anomaly detection in automotive sensor data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GDN(nn.Module):
    """
    Graph Deviation Network (GDN) for anomaly detection.

    Architecture:
    - Single-layer unidirectional GRU for temporal encoding
    - Graph Attention Network (GAT) for spatial relationships
    - LayerNorm and residual connections for training stability
    - Graph caching for efficiency

    Features:
    - Contrastive learning provides embedding separation
    - Residual connections preserve sensor distinctiveness
    - LayerNorm stabilizes training
    - Graph caching speeds up training
    """

    def __init__(
        self,
        num_nodes,
        window_size,
        embed_dim=32,
        top_k=5,
        hidden_dim=64,
        rebuild_graph_every=50,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.top_k = min(top_k, num_nodes - 1)
        self.hidden_dim = hidden_dim
        self.rebuild_graph_every = rebuild_graph_every

        # Graph caching
        self.cached_edge_index = None
        self._graph_step_counter = 0

        # Learned sensor embeddings (Xavier init so diversity loss has gradient from start)
        self.sensor_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.sensor_embeddings)
        # Projects sensor identity into hidden_dim space for differentiable node injection
        self.sensor_proj = nn.Linear(embed_dim, hidden_dim, bias=False)

        # Single-layer unidirectional GRU (baseline - fast)
        self.temporal_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Single GAT layer with improvements
        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=2,  # Reduced from 4 (less over-smoothing)
            concat=False,
            dropout=0.4,  # Increased from 0.2 (better regularization)
            add_self_loops=True,  # Changed from False (preserve own signal)
        )

        # LayerNorm after GAT for stability
        self.gat_norm = nn.LayerNorm(hidden_dim)

        # Per-sensor anomaly classifier (returns logits, no sigmoid)
        self.sensor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            # No sigmoid - use BCEWithLogitsLoss instead
        )

        # Global window classifier: concat(mean_pool(h_graph), sensor_logits) for coupling
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_nodes, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            # No sigmoid - use BCEWithLogitsLoss instead
        )

    def build_graph_from_embeddings(self, force_rebuild=False):
        """
        Build Top-K graph from learned embeddings with caching.

        Args:
            force_rebuild: If True, force rebuild even if cached

        Returns:
            edge_index: (2, num_edges) tensor of graph edges
        """
        # Check cache
        if self.cached_edge_index is not None and not force_rebuild:
            return self.cached_edge_index

        # Compute pairwise similarities
        emb_norm = F.normalize(self.sensor_embeddings, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())  # (N, N)

        # Remove self-loops
        sim_matrix.fill_diagonal_(-1e9)

        # Top-K graph
        topk_values, topk_indices = torch.topk(sim_matrix, self.top_k, dim=1)

        # PyTorch Geometric requires CPU edge_index (doesn't fully support MPS/CUDA edge_index)
        src_nodes = torch.arange(
            self.num_nodes, device="cpu", dtype=torch.long
        ).repeat_interleave(self.top_k)
        dst_nodes = topk_indices.flatten().cpu().to(torch.long)

        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

        # Cache the result
        self.cached_edge_index = edge_index
        return edge_index

    def forward(self, x, return_global=False, return_sensor_embeddings=False):
        """
        Forward pass through the model.

        Args:
            x: (B, W, N) input tensor where B=batch_size, W=window_size, N=num_sensors
            return_global: If True, also return global window anomaly logits
            return_sensor_embeddings: If True, also return per-sensor embeddings (post-GAT)

        Returns:
            - sensor_logits: (B, N) logits for each sensor being anomalous
            - global_logits: (B,) logits for window having any anomaly (optional, if return_global=True)
            - sensor_embeddings: (B, N, hidden_dim) normalized post-GAT embeddings (optional, if return_sensor_embeddings=True)
        """
        B, W, N = x.shape

        # Temporal encoding per sensor (simple: last hidden state)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)  # (B*N, W, hidden_dim)
        h_last = h_temporal[:, -1, :].reshape(B, N, -1)  # (B, N, hidden_dim)

        # Graph attention with caching
        force_rebuild = self._graph_step_counter % self.rebuild_graph_every == 0
        edge_index = self.build_graph_from_embeddings(force_rebuild=force_rebuild)
        self._graph_step_counter += 1

        # Normalize before GAT for stability
        h_last_norm = F.normalize(h_last, p=2, dim=2)  # (B, N, hidden_dim)
        # Inject sensor identity — gives sensor_embeddings a differentiable gradient path
        sensor_bias = self.sensor_proj(self.sensor_embeddings)  # (N, hidden_dim)
        h_last_norm = h_last_norm + sensor_bias.unsqueeze(0)  # broadcast over batch

        # GAT processing (per-batch loop, but with residual)
        # PyTorch Geometric requires CPU for node features and edge_index
        h_graph_list = []
        for i in range(B):
            h_node = h_last_norm[i]  # (N, hidden_dim)
            # Move to CPU for GAT (PyG doesn't fully support CUDA/MPS)
            h_node_cpu = h_node.cpu()
            h_gat_cpu = self.gat(h_node_cpu, edge_index)
            h_gat = h_gat_cpu.to(h_node.device)
            # Residual connection: preserve sensor distinctiveness
            h_gat = h_gat + h_last_norm[i]  # Residual
            h_gat = self.gat_norm(h_gat)  # Normalize
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Per-sensor anomaly logits (no sigmoid - use BCEWithLogitsLoss)
        sensor_logits = self.sensor_classifier(h_graph).squeeze(-1)  # (B, N)

        # Prepare return values
        return_values = [sensor_logits]

        if return_global:
            # Global window anomaly logits: concat(mean_pool(h_graph), sensor_logits)
            global_input = torch.cat(
                [h_graph.mean(dim=1), sensor_logits], dim=1
            )  # (B, hidden_dim + N)
            global_logits = self.global_classifier(global_input).squeeze(-1)  # (B,)
            return_values.append(global_logits)

        if return_sensor_embeddings:
            # Return normalized post-GAT sensor embeddings
            sensor_embeddings = F.normalize(h_graph, p=2, dim=2)  # (B, N, hidden_dim)
            return_values.append(sensor_embeddings)

        if len(return_values) == 1:
            return return_values[0]
        else:
            return tuple(return_values)

    def get_embeddings(self, x):
        """
        Extract normalized embeddings for center loss and distance-based scoring.

        Args:
            x: (B, W, N) input tensor

        Returns:
            embeddings: (B, hidden_dim) L2-normalized embeddings per window
        """
        B, W, N = x.shape

        # Temporal encoding (same as forward)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, -1)  # (B, N, hidden_dim)

        # Graph attention (same as forward)
        force_rebuild = self._graph_step_counter % self.rebuild_graph_every == 0
        edge_index = self.build_graph_from_embeddings(force_rebuild=force_rebuild)
        self._graph_step_counter += 1

        # Normalize before GAT
        h_last_norm = F.normalize(h_last, p=2, dim=2)  # (B, N, hidden_dim)
        # Inject sensor identity — gives sensor_embeddings a differentiable gradient path
        sensor_bias = self.sensor_proj(self.sensor_embeddings)  # (N, hidden_dim)
        h_last_norm = h_last_norm + sensor_bias.unsqueeze(0)  # broadcast over batch

        # GAT processing with residual
        # PyTorch Geometric requires CPU for node features and edge_index
        h_graph_list = []
        for i in range(B):
            h_node = h_last_norm[i]  # (N, hidden_dim)
            # Move to CPU for GAT (PyG doesn't fully support CUDA/MPS)
            h_node_cpu = h_node.cpu()
            h_gat_cpu = self.gat(h_node_cpu, edge_index)
            h_gat = h_gat_cpu.to(h_node.device)
            h_gat = h_gat + h_last_norm[i]  # Residual
            h_gat = self.gat_norm(h_gat)  # Normalize
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Aggregate across sensors: mean pooling
        embeddings = h_graph.mean(dim=1)  # (B, hidden_dim)

        # L2 normalize for stability
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_sensor_embeddings(self, x):
        """
        Extract per-sensor embeddings for sensor-level fault attribution.

        Uses POST-GAT embeddings (h_graph) to match the window embedding space.
        Window embeddings use normalize(mean(h_graph)), so sensor embeddings use
        normalize(h_graph) to ensure they're in the same embedding space.

        Args:
            x: (B, W, N) input tensor

        Returns:
            sensor_embeddings: (B, N, hidden_dim)
                L2-normalized post-GAT embeddings (same space as window embeddings)
        """
        B, W, N = x.shape

        # Temporal encoding (same as forward)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, -1)  # (B, N, hidden_dim)

        # Graph attention (same as forward)
        force_rebuild = self._graph_step_counter % self.rebuild_graph_every == 0
        edge_index = self.build_graph_from_embeddings(force_rebuild=force_rebuild)
        self._graph_step_counter += 1

        # Normalize before GAT
        h_last_norm = F.normalize(h_last, p=2, dim=2)  # (B, N, hidden_dim)
        # Inject sensor identity — gives sensor_embeddings a differentiable gradient path
        sensor_bias = self.sensor_proj(self.sensor_embeddings)  # (N, hidden_dim)
        h_last_norm = h_last_norm + sensor_bias.unsqueeze(0)  # broadcast over batch

        # GAT processing with residual
        # PyTorch Geometric requires CPU for node features and edge_index
        h_graph_list = []
        for i in range(B):
            h_node = h_last_norm[i]  # (N, hidden_dim)
            # Move to CPU for GAT (PyG doesn't fully support CUDA/MPS)
            h_node_cpu = h_node.cpu()
            h_gat_cpu = self.gat(h_node_cpu, edge_index)
            h_gat = h_gat_cpu.to(h_node.device)
            h_gat = h_gat + h_last_norm[i]  # Residual
            h_gat = self.gat_norm(h_gat)  # Normalize
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Use POST-GAT embeddings to match window embedding space
        # Window embeddings: normalize(mean(h_graph))
        # Sensor embeddings: normalize(h_graph) - same space, just not aggregated
        h_sensor = F.normalize(h_graph, p=2, dim=2)  # Normalize over hidden_dim

        return h_sensor
