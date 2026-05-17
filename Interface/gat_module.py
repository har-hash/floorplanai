import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    A single Message Passing layer using Graph Attention (GAT).
    This allows the AI to understand relationships between rooms, e.g.,
    learning that a Kitchen (node i) strongly attends to a Dining Room (node j)
    if they are connected by an edge.
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # Linear transformation weight matrix W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Self-attention parameters a
        # Concatenated features will have shape 2*out_features
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        Args:
            h (torch.Tensor): Node feature matrix of shape (Batch, N, in_features)
                              where N is the number of rooms.
            adj (torch.Tensor): Adjacency matrix of shape (Batch, N, N)
                                representing room connectivity.
        """
        # Linear Transformation: h * W
        # h shape: (B, N, in) -> Wh shape: (B, N, out)
        Wh = torch.matmul(h, self.W)
        
        B, N, _ = Wh.size()
        
        # Prepare for attention mechanism: [Wh_i || Wh_j]
        # We need to compute attention for all pairs of nodes (N x N)
        # a_input shape: (B, N, N, 2 * out_features)
        
        # Repeat Wh to match the N x N combinations
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # (B, N*N, out)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)            # (B, N*N, out)
        
        # Concatenate along the feature dimension
        a_input = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        a_input = a_input.view(B, N, N, 2 * self.out_features)
        
        # Compute self-attention scores
        # e shape: (B, N, N)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        # Masking: We only want rules/messages to pass between connected nodes
        # If there is no edge (adj == 0), set attention to a huge negative number so softmax -> 0
        # FIX C5: Use -1e4 instead of -9e15 to stay within float16 range under AMP
        # -9e15 > float16 max (65504) → becomes -inf → exp(-inf) can NaN before softmax
        zero_vec = -1e4 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax over the neighborhood
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregate messages from neighbors
        # attention shape: (B, N, N), Wh shape: (B, N, out)
        # Result shape: (B, N, out)
        h_prime = torch.bmm(attention, Wh)
        
        return F.elu(h_prime)

class RoomGraphNetwork(nn.Module):
    """
    A Graph Neural Network composed of multiple GAT layers.
    It takes raw room embeddings and injects topological/relational "message passing"
    context based on the floorplan adjacency matrix.
    """
    def __init__(self, num_room_types, embedding_dim, hidden_dim):
        super(RoomGraphNetwork, self).__init__()
        
        # Initial embedding table for room types
        self.room_embedding = nn.Embedding(num_room_types, embedding_dim)
        
        # Stack two layers of Graph Attention
        self.gat1 = GraphAttentionLayer(embedding_dim, hidden_dim, dropout=0.2)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=0.2)
        
    def forward(self, room_types, adj):
        """
        Args:
            room_types (torch.Tensor): Shape (B, N) containing integer room IDs
            adj (torch.Tensor): Shape (B, N, N) containing binary adjacency (1=connected, 0=not)
        """
        # (B, N) -> (B, N, embedding_dim)
        x = self.room_embedding(room_types)
        
        # Message Passing Layer 1
        x = self.gat1(x, adj)
        
        # Message Passing Layer 2
        # After this layer, a 'Kitchen' node's embedding will contain mixed data 
        # from its neighbors (like a 'Dining Room').
        x = self.gat2(x, adj)
        
        return x

if __name__ == "__main__":
    print("Initializing Checkpoint 2.1: Graph Attention Network...")
    
    # Let's say we have 10 room types
    # e.g., 0=Living, 1=Bed, 2=Bath, 3=Kitchen, 4=Dining
    model = RoomGraphNetwork(num_room_types=10, embedding_dim=32, hidden_dim=64)
    model.eval()
    
    # Simulate a batch with 1 floorplan containing N=5 rooms
    B, N = 1, 5
    
    # Room indices for this sample:
    # Node 0: Living, Node 1: Bed, Node 2: Bath, Node 3: Kitchen, Node 4: Dining
    dummy_room_types = torch.tensor([[0, 1, 2, 3, 4]]) 
    
    # We define adjacency. Let's strictly connect Kitchen (idx 3) to Dining (idx 4)
    # as well as Living (idx 0) to everything.
    # Shape: (B, N, N)
    dummy_adj = torch.zeros((B, N, N))
    
    # Fill in connections (undirected graph)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), # Living to everything
        (3, 4)                          # Kitchen (3) to Dining (4)
    ]
    for u, v in edges:
        dummy_adj[0, u, v] = 1.0
        dummy_adj[0, v, u] = 1.0
        
    # Also add self-loops so nodes retain their own base features during message passing
    for i in range(N):
        dummy_adj[0, i, i] = 1.0

    print("\n--- Network Input ---")
    print(f"Room Types Tensor Shape: {dummy_room_types.shape} -> (Batch, Nodes)")
    print(f"Adjacency Matrix Shape:  {dummy_adj.shape} -> (Batch, Nodes, Nodes)")
    
    print(f"\nExample Adjacency Matrix (Node 3 is Kitchen, Node 4 is Dining):")
    print(dummy_adj[0])
    
    # Forward Pass through GAT
    with torch.no_grad():
        node_embeddings = model(dummy_room_types, dummy_adj)
        
    print(f"\n--- Network Output ---")
    print(f"Post-Message Passing Node Embedding Shape: {node_embeddings.shape} -> (Batch, Nodes, Features)")
    print("\nThe GAT has successfully compiled neighbor features. The 'Kitchen' node has passed structural messages to the 'Dining Room' node through the adjacency attention filter!")
