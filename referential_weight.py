import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class SelfRefFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight_y, weight_q, weight_k, weight_beta):
        """
        Forward pass for the self-referential model.

        Args:
            inputs (torch.Tensor): Input tensor of shape [N, H, L, E]
            weight_y (torch.Tensor): Weight tensor of shape [N, H, E, M]
            weight_q (torch.Tensor): Weight tensor of shape [N, H, E, M]
            weight_k (torch.Tensor): Weight tensor of shape [N, H, E, M]
            weight_beta (torch.Tensor): Weight tensor of shape [N, H, E, 4]

        Returns:
            output (torch.Tensor): Output tensor of shape [N, H, L, M]
        """
        batch_size, num_heads, seq_len, embed_dim = inputs.shape
        _, _, _, model_dim = weight_y.shape

        # Initialize tensors
        queries = torch.zeros(batch_size, num_heads, seq_len, embed_dim, device=inputs.device, dtype=inputs.dtype)
        keys = torch.zeros(batch_size, num_heads, seq_len, embed_dim, device=inputs.device, dtype=inputs.dtype)
        betas = torch.zeros(batch_size, num_heads, seq_len, 4, device=inputs.device, dtype=inputs.dtype)
        y_diffs = torch.zeros(batch_size, num_heads, seq_len, model_dim, device=inputs.device, dtype=inputs.dtype)
        q_diffs = torch.zeros(batch_size, num_heads, seq_len, model_dim, device=inputs.device, dtype=inputs.dtype)
        k_diffs = torch.zeros(batch_size, num_heads, seq_len, model_dim, device=inputs.device, dtype=inputs.dtype)
        beta_diffs = torch.zeros(batch_size, num_heads, seq_len, 4, device=inputs.device, dtype=inputs.dtype)
        outputs = torch.zeros(batch_size, num_heads, seq_len, model_dim, device=inputs.device, dtype=inputs.dtype)

        # Clone weights
        curr_wy = weight_y.clone()
        curr_wq = weight_q.clone()
        curr_wk = weight_k.clone()
        curr_wb = weight_beta.clone()

        for t in range(seq_len):
            curr_input = inputs[:, :, t, :]

            # Main transformations
            y_out = torch.einsum('nhe,nhem->nhm', curr_input, curr_wy)
            q_out = torch.einsum('nhe,nhem->nhe', curr_input, curr_wq)
            k_out = torch.einsum('nhe,nhem->nhe', curr_input, curr_wk)
            beta_out = torch.einsum('nhe,nhef->nhf', curr_input, curr_wb)

            # Apply activations
            beta_out = torch.sigmoid(beta_out)
            betas[:, :, t, :] = beta_out

            # Softmax for queries and keys
            q_out = torch.softmax(q_out, dim=-1)
            k_out = torch.softmax(k_out, dim=-1)

            # Store queries and keys
            queries[:, :, t, :] = q_out
            keys[:, :, t, :] = k_out

            # Query transformations
            y_query = torch.einsum('nhem,nhe->nhm', curr_wy, q_out)
            q_query = torch.einsum('nhem,nhe->nhm', curr_wq, q_out)
            k_query = torch.einsum('nhem,nhe->nhm', curr_wk, q_out)
            beta_query = torch.einsum('nhef,nhe->nhf', curr_wb, q_out)

            # Key transformations
            y_key = torch.einsum('nhem,nhe->nhm', curr_wy, k_out)
            q_key = torch.einsum('nhem,nhe->nhm', curr_wq, k_out)
            k_key = torch.einsum('nhem,nhe->nhm', curr_wk, k_out)
            beta_key = torch.einsum('nhef,nhe->nhf', curr_wb, k_out)

            # Compute differences
            y_diff = y_query - y_key
            q_diff = q_query - q_key
            k_diff = k_query - k_key
            beta_diff = beta_query - beta_key

            # Store differences
            y_diffs[:, :, t, :] = y_diff
            q_diffs[:, :, t, :] = q_diff
            k_diffs[:, :, t, :] = k_diff
            beta_diffs[:, :, t, :] = beta_diff

            # Compute inserts
            y_insert = beta_out[:, :, 0:1] * y_diff
            q_insert = beta_out[:, :, 1:2] * q_diff
            k_insert = beta_out[:, :, 2:3] * k_diff
            b_insert = beta_out[:, :, 3:4] * beta_diff

            # Update weights
            curr_wy = curr_wy + torch.einsum('nhe,nhm->nhem', k_out, y_insert)
            curr_wq = curr_wq + torch.einsum('nhe,nhm->nhem', k_out, q_insert)
            curr_wk = curr_wk + torch.einsum('nhe,nhm->nhem', k_out, k_insert)
            curr_wb = curr_wb + torch.einsum('nhe,nhf->nhef', k_out, b_insert)

            # Store output
            outputs[:, :, t, :] = y_out

        # Save for backward pass
        ctx.save_for_backward(inputs, queries, keys, betas, y_diffs, q_diffs, k_diffs, beta_diffs, 
                            weight_y, weight_q, weight_k, weight_beta)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the self-referential model.

        Args:
            grad_output (torch.Tensor): Gradient of the output tensor

        Returns:
            Gradients with respect to inputs and weights.
        """
        inputs, queries, keys, betas, y_diffs, q_diffs, k_diffs, beta_diffs, \
        weight_y, weight_q, weight_k, weight_beta = ctx.saved_tensors
        
        batch_size, num_heads, seq_len, embed_dim = inputs.shape
        _, _, _, model_dim = weight_y.shape

        # Initialize gradients
        grad_inputs = torch.zeros_like(inputs)
        grad_wy = torch.zeros_like(weight_y)
        grad_wq = torch.zeros_like(weight_q)
        grad_wk = torch.zeros_like(weight_k)
        grad_wb = torch.zeros_like(weight_beta)

        # Initialize current weights
        curr_wy = weight_y.clone()
        curr_wq = weight_q.clone()
        curr_wk = weight_k.clone()
        curr_wb = weight_beta.clone()

        # Backward pass
        for t in reversed(range(seq_len)):
            curr_input = inputs[:, :, t, :]
            curr_query = queries[:, :, t, :]
            curr_key = keys[:, :, t, :]
            curr_beta = betas[:, :, t, :]
            curr_y_diff = y_diffs[:, :, t, :]
            curr_q_diff = q_diffs[:, :, t, :]
            curr_k_diff = k_diffs[:, :, t, :]
            curr_beta_diff = beta_diffs[:, :, t, :]
            curr_grad = grad_output[:, :, t, :]

            # Gradient for main output
            grad_y = curr_grad

            # Weight gradients
            grad_wy += torch.einsum('nhe,nhm->nhem', curr_input, grad_y)
            grad_inputs[:, :, t, :] += torch.einsum('nhm,nhem->nhe', grad_y, curr_wy)

            # Compute inserts
            y_insert = curr_beta[:, :, 0:1] * curr_y_diff
            q_insert = curr_beta[:, :, 1:2] * curr_q_diff
            k_insert = curr_beta[:, :, 2:3] * curr_k_diff
            b_insert = curr_beta[:, :, 3:4] * curr_beta_diff

            # Update weight gradients
            grad_wy += torch.einsum('nhe,nhm->nhem', curr_key, y_insert)
            grad_wq += torch.einsum('nhe,nhm->nhem', curr_key, q_insert)
            grad_wk += torch.einsum('nhe,nhm->nhem', curr_key, k_insert)
            grad_wb += torch.einsum('nhe,nhf->nhef', curr_key, b_insert)

            # Key gradients
            grad_k = (
                torch.einsum('nhem,nhm->nhe', curr_wy, grad_y) +
                torch.einsum('nhem,nhm->nhe', curr_wy, y_insert) +
                torch.einsum('nhem,nhm->nhe', curr_wq, q_insert) +
                torch.einsum('nhem,nhm->nhe', curr_wk, k_insert) +
                torch.einsum('nhef,nhf->nhe', curr_wb, b_insert)
            )

            # Apply softmax gradient
            grad_k = torch.softmax(curr_key, dim=-1) * (
                grad_k - (grad_k * curr_key).sum(dim=-1, keepdim=True)
            )

            # Input gradients
            grad_inputs[:, :, t, :] += torch.einsum('nhe,nhem->nhe', grad_k, curr_wk)

            # Weight gradients from key
            grad_wk += torch.einsum('nhe,nhm->nhem', curr_input, 
                                  torch.einsum('nhe,nhem->nhm', grad_k, curr_wk))

            # Update current weights
            curr_wy = curr_wy - torch.einsum('nhe,nhm->nhem', curr_key, y_insert)
            curr_wq = curr_wq - torch.einsum('nhe,nhm->nhem', curr_key, q_insert)
            curr_wk = curr_wk - torch.einsum('nhe,nhm->nhem', curr_key, k_insert)
            curr_wb = curr_wb - torch.einsum('nhe,nhf->nhef', curr_key, b_insert)

        return grad_inputs, grad_wy, grad_wq, grad_wk, grad_wb

def self_ref_forward(inputs, weight_y, weight_q, weight_k, weight_beta):
    """
    Wrapper function for the self-referential model.

    Args:
        inputs (torch.Tensor): Input tensor of shape [N, H, L, E]
        weight_y (torch.Tensor): Weight tensor of shape [N, H, E, M]
        weight_q (torch.Tensor): Weight tensor of shape [N, H, E, M]
        weight_k (torch.Tensor): Weight tensor of shape [N, H, E, M]
        weight_beta (torch.Tensor): Weight tensor of shape [N, H, E, 4]

    Returns:
        output (torch.Tensor): Output tensor of shape [N, H, L, M]
    """
    return SelfRefFunction.apply(inputs, weight_y, weight_q, weight_k, weight_beta)

class SRWMLayer(nn.Module):
    def __init__(self, num_heads, head_dim, input_dim, dropout=0.1, use_layer_norm=True,
                 use_input_softmax=False, beta_init=-1.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.input_dim = input_dim
        self.use_layer_norm = use_layer_norm
        self.use_input_softmax = use_input_softmax

        # Initialize weights
        std = 1.0 / math.sqrt(head_dim)
        self.weight_y = nn.Parameter(torch.randn(1, num_heads, head_dim, head_dim) * std)
        self.weight_q = nn.Parameter(torch.randn(1, num_heads, head_dim, head_dim) * std)
        self.weight_k = nn.Parameter(torch.randn(1, num_heads, head_dim, head_dim) * std)
        self.weight_beta = nn.Parameter(torch.randn(1, num_heads, head_dim, 4) * std + beta_init)

        # Output layers
        self.output_proj = nn.Linear(num_heads * head_dim, input_dim, bias=False)
        self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        # hidden shape: [seq_len, batch_size, input_dim]
        seq_len, batch_size, _ = hidden.shape
        
        # Reshape input
        x = hidden.reshape(seq_len, batch_size, self.num_heads, self.head_dim)
        if self.use_input_softmax:
            x = F.softmax(x, dim=-1)
        x = x.permute(1, 2, 0, 3)  # [batch_size, num_heads, seq_len, head_dim]

        # Expand weights
        weight_y = self.weight_y.expand(batch_size, -1, -1, -1)
        weight_q = self.weight_q.expand(batch_size, -1, -1, -1)
        weight_k = self.weight_k.expand(batch_size, -1, -1, -1)
        weight_beta = self.weight_beta.expand(batch_size, -1, -1, -1)

        # Forward pass
        out = self_ref_forward(x, weight_y, weight_q, weight_k, weight_beta)

        # Reshape output
        out = out.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        out = out.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        out = out.transpose(0, 1)  # [seq_len, batch_size, num_heads * head_dim]

        # Output transformations
        out = self.output_proj(out)
        out = self.dropout(out)

        # Layer norm and residual
        if self.layer_norm is not None:
            out = self.layer_norm(hidden + out)
        else:
            out = hidden + out

        return out

class SequenceDataset(Dataset):
    def __init__(self, seq_len=10, num_sequences=1000):
        """Generate sequences where each number is the sum of previous two numbers"""
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []
        
        for _ in range(num_sequences):
            # Start with random numbers between 0 and 1
            seq = [np.random.random(), np.random.random()]
            # Generate sequence
            for i in range(seq_len - 1):
                next_val = seq[-1] + seq[-2]
                seq.append(next_val)
            
            # Input sequence is all but last number
            self.sequences.append(seq[:-1])
            # Target is last number
            self.targets.append(seq[-1])
            
        self.sequences = torch.FloatTensor(self.sequences)
        self.targets = torch.FloatTensor(self.targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class SRWMPredictor(nn.Module):
    def __init__(self, seq_len, num_heads, head_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = num_heads * head_dim
        
        # Layers
        self.input_proj = nn.Linear(1, self.hidden_dim)
        self.srwm = SRWMLayer(
            num_heads=num_heads,
            head_dim=head_dim,
            input_dim=self.hidden_dim,
            dropout=dropout,
            use_layer_norm=True,
            use_input_softmax=False,
            beta_init=-1.0
        )
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        
        # Project input
        x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # SRWM processing
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        x = self.srwm(x)
        
        # Get final prediction
        x = x[-1]  # [batch_size, hidden_dim]
        out = self.output_proj(x)  # [batch_size, 1]
        return out.squeeze(-1)

def train_model():
    # Configuration
    config = {
        'seq_len': 10,
        'batch_size': 32,
        'num_heads': 4,
        'head_dim': 8,
        'num_epochs': 50,
        'learning_rate': 0.001
    }
    
    # Datasets
    train_data = SequenceDataset(seq_len=config['seq_len'], num_sequences=1000)
    val_data = SequenceDataset(seq_len=config['seq_len'], num_sequences=200)
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRWMPredictor(
        config['seq_len'],
        config['num_heads'],
        config['head_dim']
    ).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                predictions = model(sequences)
                val_loss += criterion(predictions, targets).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')

    return model

if __name__ == "__main__":
    model = train_model()
    
    # Evaluation
    test_data = SequenceDataset(seq_len=10, num_sequences=500)
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for i in range(5):
            sequence = test_data[i][0].unsqueeze(0).to(device)
            target = test_data[i][1].item()
            prediction = model(sequence).item()
            
            print(f"\nSequence: {sequence.cpu().numpy()[0][-5:]}")  # Last 5 numbers
            print(f"Target: {target:.4f}")
            print(f"Prediction: {prediction:.4f}")
