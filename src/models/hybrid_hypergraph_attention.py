import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

class HypergraphConvLayer(nn.Module):
    """
    Hypergraph convolution layer that performs a linear transformation followed by
    multiplication with the hypergraph adjacency matrix.
    """
    def __init__(self, in_features, out_features):
        super(HypergraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        """
        Forward pass of the hypergraph convolution layer.
        
        Parameters:
        x (Tensor): Input features.
        adjacency_matrix (Tensor): Hypergraph adjacency matrix.
        
        Returns:
        Tensor: Transformed features.
        """
        x = torch.matmul(adjacency_matrix, x)
        x = self.linear(x)
        return x

class AttentionLayer(nn.Module):
    """
    Attention layer that applies a softmax function to compute attention scores
    and then weights the input features by these scores.
    """
    def __init__(self, in_features, out_features):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Forward pass of the attention layer.
        
        Parameters:
        x (Tensor): Input features.
        
        Returns:
        Tensor: Attention-weighted features.
        """
        attention_scores = F.softmax(self.attention(x), dim=1)
        x = x * attention_scores
        return x

class HybridHypergraphAttentionNetwork(nn.Module):
    """
    Hybrid Hypergraph Attention Network consisting of two hypergraph convolution layers
    each followed by an attention layer, and a final fully connected layer for classification.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HybridHypergraphAttentionNetwork, self).__init__()
        self.hypergraph_conv1 = HypergraphConvLayer(input_dim, hidden_dim)
        self.attention1 = AttentionLayer(hidden_dim, hidden_dim)
        self.hypergraph_conv2 = HypergraphConvLayer(hidden_dim, hidden_dim)
        self.attention2 = AttentionLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        """
        Forward pass of the Hybrid Hypergraph Attention Network.
        
        Parameters:
        x (Tensor): Input features.
        adjacency_matrix (Tensor): Hypergraph adjacency matrix.
        
        Returns:
        Tensor: Output features.
        """
        x = F.relu(self.hypergraph_conv1(x, adjacency_matrix))
        x = self.attention1(x)
        x = F.relu(self.hypergraph_conv2(x, adjacency_matrix))
        x = self.attention2(x)
        x = self.fc(x)
        return x

def build_model(input_dim, hidden_dim, output_dim):
    """
    Function to build and return the Hybrid Hypergraph Attention Network model.
    
    Parameters:
    input_dim (int): Dimension of input features.
    hidden_dim (int): Dimension of hidden features.
    output_dim (int): Dimension of output features (number of classes).
    
    Returns:
    HybridHypergraphAttentionNetwork: The initialized model.
    """
    model = HybridHypergraphAttentionNetwork(input_dim, hidden_dim, output_dim)
    return model

def create_hypergraph_adjacency_matrix(superpixels, n_superpixels):
    """
    Function to create the hypergraph adjacency matrix from superpixel segments.
    
    Parameters:
    superpixels (list of np.ndarray): List of superpixel segments.
    n_superpixels (int): Number of unique superpixels.
    
    Returns:
    Tensor: The hypergraph adjacency matrix.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_superpixels))
    for superpixel in superpixels:
        nodes = np.unique(superpixel)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
    return adjacency_matrix

# Test functions to ensure correctness
if __name__ == "__main__":
    # Dummy data for testing
    superpixels = [np.random.randint(0, 10, size=(10, 10)) for _ in range(5)]
    n_superpixels = 10
    adjacency_matrix = create_hypergraph_adjacency_matrix(superpixels, n_superpixels)
    
    input_dim = 30
    hidden_dim = 64
    output_dim = 16
    model = build_model(input_dim, hidden_dim, output_dim)
    
    # Dummy input data
    x = torch.rand((n_superpixels, input_dim))
    
    # Forward pass
    output = model(x, adjacency_matrix)
    print("Output shape:", output.shape)
