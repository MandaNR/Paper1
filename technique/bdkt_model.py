"""
Bayesian Deep Knowledge Tracing (BDKT) Model
2-layer LSTM-like architecture with MC-Dropout and probabilistic skill layer
Implemented with NumPy and scikit-learn (no PyTorch dependency)
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleLSTMLayer:
    """Simple LSTM-like layer using dense connections"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        
        # LSTM-like weights (simplified)
        self.W_i = np.random.randn(input_size, hidden_size) * 0.01
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros(hidden_size)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through LSTM layer
        x: (seq_len, input_size) or (batch, seq_len, input_size)
        """
        # Handle both single sequence and batch
        if x.ndim == 2:
            seq_len, _ = x.shape
            h = np.zeros(self.hidden_size)
            outputs = np.zeros((seq_len, self.hidden_size))
            
            for t in range(seq_len):
                # LSTM-like computation
                h = np.tanh(x[t] @ self.W_i + h @ self.W_h + self.b)
                
                # MC-Dropout
                if self.dropout_p > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_p, h.shape)
                    h = h * mask / (1 - self.dropout_p)
                
                outputs[t] = h
            
            return outputs
        else:
            # Batch processing
            batch_size, seq_len, _ = x.shape
            outputs = np.zeros((batch_size, seq_len, self.hidden_size))
            
            for b in range(batch_size):
                h = np.zeros(self.hidden_size)
                for t in range(seq_len):
                    h = np.tanh(x[b, t] @ self.W_i + h @ self.W_h + self.b)
                    if self.dropout_p > 0:
                        mask = np.random.binomial(1, 1 - self.dropout_p, h.shape)
                        h = h * mask / (1 - self.dropout_p)
                    outputs[b, t] = h
            
            return outputs


class BDKTModel:
    """
    Bayesian Deep Knowledge Tracing Model (NumPy implementation)
    
    Architecture:
    - Input: multi-hot skill encoding + time features
    - 2-layer LSTM-like (hidden_size=128)
    - MC-Dropout (p=0.2)
    - Probabilistic skill layer
    - Output: response prediction + skill mastery + uncertainty
    """
    
    def __init__(
        self,
        num_skills: int,
        hidden_size: int = 128,
        dropout_p: float = 0.2,
        beta: float = 1.0,
        gamma: float = 0.05,
        delta: float = 0.1,
    ):
        self.num_skills = num_skills
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.beta = beta      # KL divergence weight
        self.gamma = gamma    # L2 regularization weight
        self.delta = delta    # Skill uncertainty weight
        
        # Input projection
        self.W_proj = np.random.randn(num_skills + 1, hidden_size) * 0.01
        self.b_proj = np.zeros(hidden_size)
        
        # LSTM-like layers
        self.lstm1 = SimpleLSTMLayer(hidden_size, hidden_size, dropout_p)
        self.lstm2 = SimpleLSTMLayer(hidden_size, hidden_size, dropout_p)
        
        # Probabilistic skill layer
        self.W_skill_mean = np.random.randn(hidden_size, num_skills) * 0.01
        self.b_skill_mean = np.zeros(num_skills)
        self.W_skill_logvar = np.random.randn(hidden_size, num_skills) * 0.01
        self.b_skill_logvar = np.zeros(num_skills)
        
        # Response prediction head
        self.W_resp1 = np.random.randn(hidden_size + num_skills, hidden_size // 2) * 0.01
        self.b_resp1 = np.zeros(hidden_size // 2)
        self.W_resp2 = np.random.randn(hidden_size // 2, 1) * 0.01
        self.b_resp2 = np.zeros(1)
        
        self.params = [
            self.W_proj, self.b_proj,
            self.lstm1.W_i, self.lstm1.W_h, self.lstm1.b,
            self.lstm2.W_i, self.lstm2.W_h, self.lstm2.b,
            self.W_skill_mean, self.b_skill_mean,
            self.W_skill_logvar, self.b_skill_logvar,
            self.W_resp1, self.b_resp1, self.W_resp2, self.b_resp2,
        ]
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def forward(
        self,
        skill_input: np.ndarray,
        time_input: np.ndarray,
        return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass
        
        Args:
            skill_input: (seq_len, num_skills) multi-hot encoding
            time_input: (seq_len, 1) log-transformed time gaps
            return_uncertainty: if True, return skill uncertainty
        
        Returns:
            response_pred: (seq_len, 1) predicted response probability
            skill_mastery: (seq_len, num_skills) skill mastery estimates
            skill_uncertainty: (seq_len, num_skills) skill uncertainty (if requested)
        """
        seq_len = skill_input.shape[0]
        
        # Concatenate skill and time features
        x = np.concatenate([skill_input, time_input], axis=-1)  # (seq_len, num_skills+1)
        
        # Project to hidden dimension
        x = self.relu(x @ self.W_proj + self.b_proj)  # (seq_len, hidden_size)
        
        # LSTM layers
        x = self.lstm1.forward(x)  # (seq_len, hidden_size)
        x = self.lstm2.forward(x)  # (seq_len, hidden_size)
        
        # Probabilistic skill layer
        skill_mean = self.sigmoid(x @ self.W_skill_mean + self.b_skill_mean)  # (seq_len, num_skills)
        skill_logvar = x @ self.W_skill_logvar + self.b_skill_logvar  # (seq_len, num_skills)
        skill_var = np.exp(np.clip(skill_logvar, -10, 10))
        
        # Sample skills with reparameterization trick
        eps = np.random.randn(*skill_mean.shape)
        skill_samples = skill_mean + np.sqrt(skill_var + 1e-8) * eps
        skill_samples = np.clip(skill_samples, 0, 1)
        
        # Response prediction
        combined = np.concatenate([x, skill_samples], axis=-1)
        h = self.relu(combined @ self.W_resp1 + self.b_resp1)
        
        # MC-Dropout
        if self.dropout_p > 0:
            mask = np.random.binomial(1, 1 - self.dropout_p, h.shape)
            h = h * mask / (1 - self.dropout_p)
        
        response_pred = self.sigmoid(h @ self.W_resp2 + self.b_resp2)  # (seq_len, 1)
        
        if return_uncertainty:
            skill_std = np.sqrt(skill_var + 1e-8)
            return response_pred, skill_mean, skill_std
        
        return response_pred, skill_mean, None
    
    def compute_loss(
        self,
        response_pred: np.ndarray,
        response_true: np.ndarray,
        skill_mean: np.ndarray,
        skill_logvar: np.ndarray,
        skill_input: np.ndarray,
    ) -> Tuple[float, Dict]:
        """
        Compute negative ELBO loss with regularizers
        
        Loss = BCE(response) + β*KL(skills) + γ*L2(weights) + δ*skill_uncertainty
        """
        # Flatten predictions and targets
        response_pred = response_pred.flatten()
        response_true = response_true.flatten()
        
        # Response prediction loss (BCE)
        eps = 1e-7
        response_pred = np.clip(response_pred, eps, 1 - eps)
        response_loss = -np.mean(
            response_true * np.log(response_pred) + 
            (1 - response_true) * np.log(1 - response_pred)
        )
        
        # KL divergence for skill mastery (Gaussian prior N(0.5, 0.1))
        prior_mean = 0.5
        prior_logvar = np.log(0.1 ** 2)
        
        kl_loss = -0.5 * np.mean(
            1 + skill_logvar - prior_logvar
            - (skill_mean - prior_mean) ** 2 / np.exp(prior_logvar)
            - np.exp(skill_logvar) / np.exp(prior_logvar)
        )
        
        # L2 regularization on weights
        l2_loss = 0.0
        for param in self.params:
            l2_loss += np.sum(param ** 2)
        l2_loss = l2_loss / len(self.params)
        
        # Skill uncertainty regularization
        skill_var = np.exp(np.clip(skill_logvar, -10, 10))
        uncertainty_loss = np.mean(skill_var)
        
        # Total loss
        total_loss = (
            response_loss
            + self.beta * kl_loss
            + self.gamma * l2_loss
            + self.delta * uncertainty_loss
        )
        
        return total_loss, {
            'response_loss': float(response_loss),
            'kl_loss': float(kl_loss),
            'l2_loss': float(l2_loss),
            'uncertainty_loss': float(uncertainty_loss),
        }


if __name__ == "__main__":
    # Test model
    model = BDKTModel(num_skills=30, hidden_size=128)
    
    seq_len = 100
    skill_input = np.random.randint(0, 2, (seq_len, 30)).astype(np.float32)
    time_input = np.random.randn(seq_len, 1).astype(np.float32)
    response_true = np.random.randint(0, 2, seq_len).astype(np.float32)
    
    response_pred, skill_mean, _ = model.forward(skill_input, time_input)
    print(f"Response pred shape: {response_pred.shape}")
    print(f"Skill mean shape: {skill_mean.shape}")
    
    # Test loss
    skill_logvar = np.random.randn(seq_len, 30)
    loss, loss_dict = model.compute_loss(response_pred, response_true[:, np.newaxis], skill_mean, skill_logvar, skill_input)
    print(f"Loss: {loss:.4f}")
    print(f"Loss components: {loss_dict}")
