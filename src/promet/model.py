import dgl  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import Dict, Tuple

from dgl.nn.pytorch.conv import EGNNConv  # type: ignore


class EGNNBlock(nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) Block with residual connections and normalization.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, edge_feat_dim, output_dim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNConv(
                in_size=(input_dim if i == 0 else output_dim),
                hidden_size=hidden_dim,
                out_size=output_dim,
                edge_feat_size=edge_feat_dim
            ) for i in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(output_dim) for _ in range(num_layers)])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, graph, node_features, edge_features=None, coord_features=None):
        for i, layer in enumerate(self.layers):
            residual = node_features
            node_features, _ = layer(graph, node_features, coord_features, edge_features)
            node_features = self.norm_layers[i](node_features)
            node_features = self.activation(node_features)
            node_features = self.dropout(node_features)
            node_features += residual  # Residual connection
        return node_features


class NodeFeatureEncoder(nn.Module):
    """
    Node feature encoder with initial embedding and EGNN-based feature extraction.
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate, output_dim, egnn_args):
        super().__init__()
        self.initial_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.egnn_block = EGNNBlock(**egnn_args)

    def forward(self, graph: dgl.DGLGraph):
        node_features = graph.ndata["L0"]
        # Edge features are only consumed when egnn_args.edge_feat_dim > 0 (0 for the released model).
        edge_features = graph.edata["L0"] if "L0" in graph.edata else None
        coord_features = graph.ndata["xyz"]

        node_features = self.initial_embedding(node_features.to(torch.float32)).squeeze()
        node_features = self.egnn_block(graph, node_features, edge_features, coord_features)

        return node_features, edge_features


class LinearFeatureTransform(nn.Module):
    """
    Linear transformation module with multiple layers, normalization, and dropout.
    """
    def __init__(self, input_dim, output_dim, num_layers=2, dropout_rate=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AttentionFeatureTransform(nn.Module):
    """
    Self-attention-based feature transformation module.
    """
    def __init__(self, input_dim, embed_dim, output_dim, num_heads=8, dropout_rate=0.1, num_layers=1):
        super().__init__()
        self.reduction_layer = nn.Linear(input_dim, embed_dim, bias=True)
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, output_dim, bias=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduction_layer(x)
        for attn_block in self.attention_blocks:
            attn_output, _ = attn_block(x, x, x)
            x = self.layer_norm(x + attn_output)
            x = self.activation(x)
            x = self.dropout(x)

        return self.output_layer(x)


class GraphDecoder(nn.Module):
    """
    Decoder module for predicting probability, class type, and binning information.
    """
    def __init__(self, use_attention_prob, use_attention_type, binning_layer_args, prob_transform_args, type_transform_args):
        super().__init__()
        self.prob_transform = self._select_transform(prob_transform_args, use_attention_prob)
        self.type_transform = self._select_transform(type_transform_args, use_attention_type)
        self.binning_layer = LinearFeatureTransform(**binning_layer_args) 
    
    @staticmethod
    def _select_transform(args, use_attention):
        if use_attention:
            return AttentionFeatureTransform(**args["attention"])
        return LinearFeatureTransform(**args["linear"])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob_prediction = self.prob_transform(x)
        binning_prediction = self.binning_layer(x)
        type_prediction = self.type_transform(x)
        return prob_prediction, type_prediction, binning_prediction


class Model(nn.Module):
    """
    PrOMet EGNN: node embedding -> 5 EGNN layers -> occupancy / metal-type heads.
    """
    def __init__(self, encoder_args: Dict, decoder_args: Dict):
        super().__init__()
        self.encoder = NodeFeatureEncoder(**encoder_args)
        self.decoder = GraphDecoder(**decoder_args)

    def forward(self, graph: dgl.DGLGraph):
        node_features, _ = self.encoder(graph)
        prob_prediction, type_prediction, binning_prediction = self.decoder(node_features)
        return prob_prediction, type_prediction, binning_prediction

