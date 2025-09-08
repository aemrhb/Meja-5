import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ===================== Weight Initialization =====================
def jepa_init_weights(m):
    """Improved JEPA-style initialization with better scaling."""
    if isinstance(m, nn.Linear):
        # Use fan-in scaling for better gradient flow
        fan_in = m.weight.size(1)
        std = math.sqrt(2.0 / fan_in) * 0.5  # Scale down for stability
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)

def init_attention_weights(m):
    """Specialized initialization for attention layers."""
    if isinstance(m, nn.Linear):
        # Smaller initialization for attention to prevent saturation
        nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# ===================== Multi-Head Attention Block =====================
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(MultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embedding_dim // self.num_heads
        self.dropout = dropout
        # 'cls' to use a learnable class token, 'avg' to use masked average pooling
        self.summary_mode = summary_mode
        self.use_relative_positional_encoding = use_relative_positional_encoding

        # Linear projections for query, key, and value
        self.query = nn.Linear(embedding_dim, embedding_dim).float()
        self.key = nn.Linear(embedding_dim, embedding_dim).float()
        self.value = nn.Linear(embedding_dim, embedding_dim).float()
        self.norm = nn.LayerNorm(embedding_dim)

        # Learnable token embedding (used when summary_mode == 'cls')
        self.token_embedding = nn.Parameter(torch.randn(1, embedding_dim))

        # Relative positional encoding heads
        if self.use_relative_positional_encoding:
            # Simple MLPs to map Δp -> bias per head and value per head-dim
            self.rel_bias_mlp = nn.Sequential(
                nn.Linear(6, 64), nn.GELU(), nn.Linear(64, num_heads)
            )
            self.rel_value_mlp = nn.Sequential(
                nn.Linear(6, 128), nn.GELU(), nn.Linear(128, embedding_dim)
            )



    def forward(self, padded_batches, masks, sum_token_in=None, positions_3d=None):
        # Shapes: padded_batches [B, N, T, D], masks [B, N, T]
        B, N, T, D = padded_batches.shape
        H = self.num_heads
        Hd = self.head_dim


        if self.summary_mode == 'cls':
            # Use passed summary token if provided, otherwise initialize from learnable embedding
            cls_token = sum_token_in if sum_token_in is not None else self.token_embedding.expand(B, N, 1, D)
            x = torch.cat((cls_token, padded_batches), dim=-2)  # [B, N, T+1, D]
            L = T + 1

            q = self.query(x.float())
            k = self.key(x.float())
            v = self.value(x.float())

            # Build float attention mask (large_neg = masked/ignore per SDPA semantics)
            prep = torch.ones(B, N, 1, device=masks.device, dtype=masks.dtype)
            key_keep = torch.cat((prep, masks), dim=-1)  # [B, N, L] 1=valid, 0=masked
            large_neg = torch.finfo(torch.float32).min
            key_mask = torch.where(key_keep == 0, large_neg, 0.0).view(B * N, 1, 1, L)

            # Reshape to heads: [B*N, H, L, Hd]
            q = q.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            k = k.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            v = v.view(B * N, L, H, Hd).permute(0, 2, 1, 3)

            if not self.use_relative_positional_encoding:
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=self.dropout)
            else:
                # Compute attention logits
                scale = 1.0 / math.sqrt(Hd)
                attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B*N, H, L, L]
                # Add relative positional bias if positions provided
                if positions_3d is not None:
                    # positions_3d: [B, N, T, 3] -> with cls
                    pos = positions_3d[:, :, :T, :]
                    pos = torch.cat((torch.zeros(B, N, 1, 3, device=pos.device, dtype=pos.dtype), pos), dim=-2)
                    pos = pos.view(B * N, L, 3)
                    # Pairwise Δp features: [dx,dy,dz,||Δp||, r^2, 1]
                    diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B*N, L, L, 3]
                    dist = torch.norm(diff, dim=-1, keepdim=True)
                    dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
                    rel_feat = torch.cat([diff, dist, dist2], dim=-1)  # [B*N, L, L, 6]
                    rel_bias = self.rel_bias_mlp(rel_feat)  # [B*N, L, L, H]
                    rel_bias = rel_bias.permute(0, 3, 1, 2).contiguous()  # [B*N, H, L, L]
                    # Zero out interactions with CLS token (index 0)
                    rel_bias[:, :, 0, :] = 0
                    rel_bias[:, :, :, 0] = 0
                    attn_logits = attn_logits + rel_bias
                # Add mask
                attn_logits = attn_logits + key_mask
                attn_weights = F.softmax(attn_logits, dim=-1)
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
                # Base output
                attn = torch.matmul(attn_weights, v)
                # Add positional value contribution
                if positions_3d is not None:
                    pos = positions_3d[:, :, :T, :]
                    pos = torch.cat((torch.zeros(B, N, 1, 3, device=pos.device, dtype=pos.dtype), pos), dim=-2)
                    pos = pos.view(B * N, L, 3)
                    diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B*N, L, L, 3]
                    dist = torch.norm(diff, dim=-1, keepdim=True)
                    dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
                    rel_feat = torch.cat([diff, dist, dist2], dim=-1)  # [B*N, L, L, 6]
                    rel_val = self.rel_value_mlp(rel_feat)  # [B*N, L, L, D]
                    rel_val = rel_val.view(B * N, L, L, H, Hd).permute(0, 3, 1, 2, 4)  # [B*N, H, L, L, Hd]
                    rel_val[:, :, 0, :, :] = 0
                    rel_val[:, :, :, 0, :] = 0
                    pos_out = torch.einsum('bhij,bhijc->bhic', attn_weights, rel_val)
                    attn = attn + pos_out
            # print('q.shape in local attention', q.shape)
            # print('k.shape in local attention', k.shape)
            # print('v.shape in local attention', v.shape)
            # print('x.shape in local attention', x.shape)
            # Merge heads: [B*N, L, D]
            attn = attn.permute(0, 2, 1, 3).contiguous().view(B, N, L, D)

            # Separate class token and residual
            sum_token = attn[:, :, :1, :]
            sum_token = self.norm(sum_token + x[:, :, :1, :])
            out = attn[:, :, 1:, :] + x[:, :, 1:, :]
            return out, sum_token
        else:
            x = padded_batches
            L = T

            q = self.query(x.float())
            k = self.key(x.float())
            v = self.value(x.float())

            # Mask: [B, N, T] -> [B*N, 1, 1, T] (large_neg=masked)
            large_neg = torch.finfo(torch.float32).min
            key_mask = torch.where(masks == 0, large_neg, 0.0).view(B * N, 1, 1, L)

            q = q.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            k = k.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            v = v.view(B * N, L, H, Hd).permute(0, 2, 1, 3)

            if not self.use_relative_positional_encoding:
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=self.dropout)
            else:
                scale = 1.0 / math.sqrt(Hd)
                attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale
                if positions_3d is not None:
                    pos = positions_3d.view(B * N, L, 3)
                    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
                    dist = torch.norm(diff, dim=-1, keepdim=True)
                    dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
                    rel_feat = torch.cat([diff, dist, dist2], dim=-1)
                    rel_bias = self.rel_bias_mlp(rel_feat).permute(0, 3, 1, 2).contiguous()
                    attn_logits = attn_logits + rel_bias
                attn_logits = attn_logits + key_mask
                attn_weights = F.softmax(attn_logits, dim=-1)
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
                attn = torch.matmul(attn_weights, v)
                if positions_3d is not None:
                    pos = positions_3d.view(B * N, L, 3)
                    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
                    dist = torch.norm(diff, dim=-1, keepdim=True)
                    dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
                    rel_feat = torch.cat([diff, dist, dist2], dim=-1)
                    rel_val = self.rel_value_mlp(rel_feat).view(B * N, L, L, H, Hd).permute(0, 3, 1, 2, 4)
                    pos_out = torch.einsum('bhij,bhijc->bhic', attn_weights, rel_val)
                    attn = attn + pos_out

            attn = attn.permute(0, 2, 1, 3).contiguous().view(B, N, L, D)
            out = attn + x
            sum_token = torch.zeros_like(out[:, :, :1, :])
            return out, sum_token

# ===================== Global Multi-Head Attention Block =====================
class GlobalMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1, summary_mode='avg', use_relative_positional_encoding=False):
        super(GlobalMultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embedding_dim // self.num_heads
        self.dropout = dropout
        # 'cls' uses provided sum_token; 'avg' recomputes masked average as summary token
        self.summary_mode = summary_mode
        self.use_relative_positional_encoding = use_relative_positional_encoding

        # Linear projections for query, key, and value
        self.query = nn.Linear(embedding_dim, embedding_dim).float()
        self.key = nn.Linear(embedding_dim, embedding_dim).float()
        self.value = nn.Linear(embedding_dim, embedding_dim).float()
        self.norm = nn.LayerNorm(embedding_dim)

        # Learnable token embedding
        self.token_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        if self.use_relative_positional_encoding:
            self.rel_bias_mlp = nn.Sequential(
                nn.Linear(6, 64), nn.GELU(), nn.Linear(64, num_heads)
            )
            self.rel_value_mlp = nn.Sequential(
                nn.Linear(6, 128), nn.GELU(), nn.Linear(128, embedding_dim)
            )



    def forward(self, padded_batches, sum_token, masks, positions_3d=None):
        # padded_batches: [B, N, T, D], sum_token: [B, N, 1, D], masks: [B, N, T]
        B, N, T, D = padded_batches.shape
        H = self.num_heads
        Hd = self.head_dim


        # Choose summary token for global attention
        if self.summary_mode == 'avg':
            weights = masks.unsqueeze(-1).to(padded_batches.dtype)
            summed = (padded_batches * weights).sum(dim=-2, keepdim=True)
            denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            summary_token = summed / denom  # [B, N, 1, D]
        else:
            summary_token = sum_token

        # Compute Q from all tokens, K/V from summary tokens
        q = self.query(padded_batches.float())      # [B, N, T, D]
        k = self.key(summary_token.float())         # [B, N, 1, D]
        v = self.value(summary_token.float())       # [B, N, 1, D]

        # Reshape to heads for FlashAttention
        q = q.view(B * N, T, H, Hd).permute(0, 2, 1, 3)   # [B*N, H, T, Hd]
        k = k.view(B * N, 1, H, Hd).permute(0, 2, 1, 3)   # [B*N, H, 1, Hd]
        v = v.view(B * N, 1, H, Hd).permute(0, 2, 1, 3)   # [B*N, H, 1, Hd]
        # print('q.shape in global attention', q.shape)
        # print('k.shape in global attention', k.shape)
        # print('v.shape in global attention', v.shape)
        # print('padded_batches.shape in global attention', padded_batches.shape)
        # print('summary_token.shape in global attention', summary_token.shape)
        # Create attention mask for global attention
        # masks: [B, N, T] -> [B*N, H, T, 1] (large_neg=masked)
        large_neg = torch.finfo(torch.float32).min
        attn_mask = torch.where(masks == 0, large_neg, 0.0).view(B * N, 1, T, 1).expand(-1, H, -1, -1)

        # Perform attention with proper masking
        if not self.use_relative_positional_encoding:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)
        else:
            scale = 1.0 / math.sqrt(Hd)
            attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B*N, H, T, 1]
            if positions_3d is not None and self.summary_mode == 'avg':
                # Build summary positions as masked average of positions
                pos = positions_3d  # [B, N, T, 3]
                weights = masks.unsqueeze(-1).to(pos.dtype)
                pos_sum = (pos * weights).sum(dim=-2, keepdim=True)
                denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
                pos_summary = pos_sum / denom  # [B, N, 1, 3]
                pos_q = pos.view(B * N, T, 3)
                pos_k = pos_summary.view(B * N, 1, 3)
                diff = pos_q.unsqueeze(2) - pos_k.unsqueeze(1)  # [B*N, T, 1, 3]
                dist = torch.norm(diff, dim=-1, keepdim=True)
                dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
                rel_feat = torch.cat([diff, dist, dist2], dim=-1)  # [B*N, T, 1, 6]
                rel_bias = self.rel_bias_mlp(rel_feat).permute(0, 3, 1, 2)  # [B*N, H, T, 1]
                attn_logits = attn_logits + rel_bias
            attn_logits = attn_logits + attn_mask
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn = torch.matmul(attn_weights, v)
            if positions_3d is not None and self.summary_mode == 'avg':
                diff = pos_q.unsqueeze(2) - pos_k.unsqueeze(1)
                dist = torch.norm(diff, dim=-1, keepdim=True)
                dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
                rel_feat = torch.cat([diff, dist, dist2], dim=-1)
                rel_val = self.rel_value_mlp(rel_feat).view(B * N, T, 1, H, Hd).permute(0, 3, 1, 2, 4)
                pos_out = (attn_weights * rel_val).sum(dim=-2)
                attn = attn + pos_out

        # Merge heads and restore shape
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, N, T, D)

        out = attn + padded_batches
        return out

# ===================== Feed Forward Block =====================
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)  # Add dropout layer
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)  # Apply dropout after the first linear layer
        x = self.linear2(x)
        x = self.norm(x + residual)  # Residual connection + normalization
        return x

# ===================== Transformer Block =====================
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout=dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, masks, sum_token_in=None, positions_3d=None):
        x, sum_token = self.attention(x, masks, sum_token_in=sum_token_in, positions_3d=positions_3d)
        x = self.feed_forward(x)
        return x, sum_token

# ===================== Global Transformer Block =====================
class GlobalTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(GlobalTransformerBlock, self).__init__()
        self.attention = GlobalMultiHeadAttention(embedding_dim, num_heads, dropout=dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, sum_token, masks, positions_3d=None):
        x = self.attention(x, sum_token, masks, positions_3d=positions_3d)
        x = self.feed_forward(x)
        return x

# ===================== Global-Only Multi-Head Attention Block =====================
class GlobalOnlyMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1, summary_mode='avg', use_gating=False, use_relative_positional_encoding=False):
        super(GlobalOnlyMultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embedding_dim // self.num_heads
        self.dropout = dropout
        self.summary_mode = summary_mode
        self.use_gating = use_gating
        self.use_relative_positional_encoding = use_relative_positional_encoding

        # Linear projections for query, key, and value
        self.query = nn.Linear(embedding_dim, embedding_dim).float()
        self.key = nn.Linear(embedding_dim, embedding_dim).float()
        self.value = nn.Linear(embedding_dim, embedding_dim).float()
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Learnable gating mechanism for combining local and global information
        self.gate_proj = nn.Linear(embedding_dim * 2, embedding_dim).float()
        self.gate_norm = nn.LayerNorm(embedding_dim)

        if self.use_relative_positional_encoding:
            self.rel_bias_mlp = nn.Sequential(
                nn.Linear(6, 64), nn.GELU(), nn.Linear(64, num_heads)
            )
            self.rel_value_mlp = nn.Sequential(
                nn.Linear(6, 128), nn.GELU(), nn.Linear(128, embedding_dim)
            )

    def forward(self, padded_batches, sum_token, masks, positions_3d=None):
        """
        Global-only attention where only summary tokens (global tokens) interact with each other.
        The individual tokens within clusters don't participate in this attention.
        """
        # Choose summary token for global attention
        if self.summary_mode == 'avg':
            # masks: [B, N, T]
            weights = masks.unsqueeze(-1).to(padded_batches.dtype)
            summed = (padded_batches * weights).sum(dim=-2, keepdim=True)
            denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            summary_token = summed / denom  # [B, N, 1, D]
        else:
            summary_token = sum_token

        # print('summary_token.shape in global-only attention', summary_token.shape)
        # Only summary tokens participate in attention: Q/K/V from summary tokens
        q = self.query(summary_token.float())  # [B, N, 1, D]
        k = self.key(summary_token.float())    # [B, N, 1, D]
        v = self.value(summary_token.float())  # [B, N, 1, D]
        B, N, Lg, D = q.shape  # Lg = 1
        H = self.num_heads
        Hd = self.head_dim
    #    print('q.shape in global-only attention', q.shape)
    #    print('k.shape in global-only attention', k.shape)
    #    print('v.shape in global-only attention', v.shape)
        
        # For global attention, we want all summary tokens to attend to each other
        # Reshape to [B, N, H, Hd] for multi-head attention
        q = q.squeeze(2).view(B, N, H, Hd).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        k = k.squeeze(2).view(B, N, H, Hd).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        v = v.squeeze(2).view(B, N, H, Hd).permute(0, 2, 1, 3)  # [B, H, N, Hd]
    #    print('q.shape in global-only attention', q.shape)
    #    print('k.shape in global-only attention', k.shape)
    #    print('v.shape in global-only attention', v.shape)
        
        # Perform attention (no masking needed for global-only as all summary tokens are valid)
        if not self.use_relative_positional_encoding:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)  # [B, H, N, Hd]
        else:
            scale = 1.0 / math.sqrt(Hd)
            attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, H, N, N]
            if positions_3d is not None:
                # Compute cluster summary positions
                if self.summary_mode == 'avg':
                    pos = positions_3d  # [B, N, T, 3]
                    weights = masks.unsqueeze(-1).to(pos.dtype)
                    pos_sum = (pos * weights).sum(dim=-2, keepdim=True)
                    denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
                    pos_summary = pos_sum / denom  # [B, N, 1, 3]
                else:
                    # If provided sum_token corresponds to some latent, we fallback to avg of positions
                    pos = positions_3d
                    weights = masks.unsqueeze(-1).to(pos.dtype)
                    pos_sum = (pos * weights).sum(dim=-2, keepdim=True)
                    denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
                    pos_summary = pos_sum / denom
                pos_summary = pos_summary.squeeze(2)  # [B, N, 3]
                diff = pos_summary.unsqueeze(2) - pos_summary.unsqueeze(1)  # [B, N, N, 3]
                dist = torch.norm(diff, dim=-1, keepdim=True)
                dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
                rel_feat = torch.cat([diff, dist, dist2], dim=-1)  # [B, N, N, 6]
                rel_bias = self.rel_bias_mlp(rel_feat).permute(0, 3, 1, 2).contiguous()  # [B, H, N, N]
                attn_logits = attn_logits + rel_bias
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn = torch.matmul(attn_weights, v)
            if positions_3d is not None:
                rel_val = self.rel_value_mlp(rel_feat).view(B, N, N, H, Hd).permute(0, 3, 1, 2, 4)  # [B, H, N, N, Hd]
                pos_out = (attn_weights * rel_val).sum(dim=-2)
                attn = attn + pos_out
        # print('attn.shape in global-only attention', attn.shape)
        
        # Merge heads back
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, N, D)  # [B, N, D]
        attn = attn.unsqueeze(2)  # [B, N, 1, D]
    #    print('attn.shape in global-only attention 2', attn.shape)
        # Residual on the summary token
        updated_summary = attn + summary_token
        
        # Broadcast the updated summary information back to all tokens in each cluster
        # This allows the global information to influence local tokens
        expanded_summary = updated_summary.expand(-1, -1, padded_batches.size(2), -1)
        
        # Normalize both local and global information to prevent domination
        local_norm = torch.norm(padded_batches, dim=-1, keepdim=True).clamp(min=1e-6)
        global_norm = torch.norm(expanded_summary, dim=-1, keepdim=True).clamp(min=1e-6)
        
        local_normalized = padded_batches / local_norm
        global_normalized = expanded_summary / global_norm
        
        # Combine local and global information according to selected mode
        if self.use_gating:
            combined_features = torch.cat([local_normalized, global_normalized], dim=-1)
            pre_norm_gate = torch.sigmoid(self.gate_proj(combined_features))
            gate = self.gate_norm(pre_norm_gate)
            # Debug: print gate weights (global vs local) per feature occasionally
            if torch.rand(1).item() < 0.1:
                with torch.no_grad():
                    # Pre-norm gate is in [0,1] and interpretable as global share
                    global_share_pre = pre_norm_gate.mean(dim=(0, 1, 2)).detach().cpu()
                    local_share_pre = (1.0 - global_share_pre)
                    # Post-norm gate mean (not a probability, for reference)
                    global_share_post_mean = gate.mean(dim=(0, 1, 2)).detach().cpu().mean().item()
                    print(
                        f"[GlobalOnlyMultiHeadAttention] gate shares per feature (D={gate.shape[-1]}):\n"
                        f"  pre_norm global_share_mean={global_share_pre.mean().item():.4f}, pre_norm local_share_mean={local_share_pre.mean().item():.4f}\n"
                        f"  post_norm global_share_mean={global_share_post_mean:.4f}"
                    )
                    print("  pre_norm global_share_per_feature:", global_share_pre.tolist())
                    print("  pre_norm local_share_per_feature:", local_share_pre.tolist())
            output = (1 - gate) * padded_batches + gate * expanded_summary
        else:
            output = padded_batches + expanded_summary
        

        
        return output, updated_summary  # [B, N, 1, D]

# ===================== Global-Only Transformer Block =====================
class GlobalOnlyTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, ff_dim, num_heads=4, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(GlobalOnlyTransformerBlock, self).__init__()
        self.attention = GlobalOnlyMultiHeadAttention(embedding_dim, num_heads, dropout=dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, sum_token, masks, positions_3d=None):
        x, updated_sum_token = self.attention(x, sum_token, masks, positions_3d=positions_3d)
        x = self.feed_forward(x)
        return x, updated_sum_token

# ===================== Hierarchical Transformer Block =====================
class HierarchicalTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        """
        Hierarchical processing block that follows the pattern:
        1. Local processing (tokens within clusters interact)
        2. Global-only processing (only global tokens interact with each other)
        3. Local processing again (tokens return to their clusters)
        """
        super(HierarchicalTransformerBlock, self).__init__()
        
        # Local attention blocks (before and after global)
        self.local_attention_1 = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.local_attention_2 = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        
        # Global-only attention block
        self.global_only_attention = GlobalOnlyTransformerBlock(embedding_dim, ff_dim, num_heads=num_heads, dropout=dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        
        # Feed forward layers
        self.feed_forward_1 = FeedForward(embedding_dim, ff_dim, dropout)
        self.feed_forward_2 = FeedForward(embedding_dim, ff_dim, dropout)
        self.feed_forward_3 = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, masks, sum_token_in=None, positions_3d=None):
        # Store input for final residual connection
        input_residual = x
        
        # Stage 1: Local attention within clusters
        x, sum_token = self.local_attention_1(x, masks, sum_token_in=sum_token_in, positions_3d=positions_3d)
    #    x = self.feed_forward_1(x)
        local_residual_1 = x
        
        # Stage 2: Global-only attention (only summary tokens interact)
        x, final_sum_token = self.global_only_attention(x, sum_token, masks, positions_3d=positions_3d)
    #    x = self.feed_forward_2(x)
    #    global_residual = x
        
    #    Stage 3: Local attention within clusters again (using updated global info)
    #    x, final_sum_token = self.local_attention_2(x, masks, sum_token_in=updated_sum_token, positions_3d=positions_3d)
    #    x = self.feed_forward_3(x)
        
        # Apply hierarchical residual connections
        # Combine information from all stages
        x = x + input_residual
        
        return x, final_sum_token



# ===================== Global-Local Transformer Block =====================
class G_L_TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(G_L_TransformerBlock, self).__init__()
        self.attention_local = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.attention_global = GlobalTransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, masks, sum_token_in=None, positions_3d=None):
        # Store input for final residual connection
        input_residual = x
        
        # Local attention block: self-attention within each cluster
        x, sum_token = self.attention_local(x, masks, sum_token_in=sum_token_in, positions_3d=positions_3d)
        x = self.feed_forward(x)
        
        # Store intermediate state for local residual connection
        local_residual = x
        
        # Global attention block: cross-cluster attention using summary token
        x = self.attention_global(x, sum_token, masks, positions_3d=positions_3d)
        x = self.feed_forward(x)
        
        # Apply residual connections:
        # 1. Local residual: from after local attention to after global attention
     #   x = x + local_residual
        # 2. Input residual: from original input to final output
        x = x + input_residual
        return x, sum_token

# ===================== Nomeformer Model =====================
class nomeformer(nn.Module):
    def __init__(self, feature_dim, embedding_dim, num_heads, num_attention_blocks, dropout=0.1, summary_mode='cls', use_hierarchical=False, num_hierarchical_stages=1, fourier=True, relative_positional_encoding=False):
        """
        Args:
            feature_dim: Input feature dimension
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            num_attention_blocks: Number of transformer blocks
            dropout: Dropout rate
            summary_mode: Summary mode for attention ('cls' or 'avg')
            use_hierarchical: Whether to use hierarchical processing (True) or original G_L blocks (False)
            num_hierarchical_stages: Number of hierarchical processing stages (only used if use_hierarchical=True)
            fourier: Whether to use positional fourier encoding (True) or not (False)
            relative_positional_encoding: If True, use relative 3D positional encoding (bias + value)
        """
        super(nomeformer, self).__init__()
        ff_dim = 2 * embedding_dim
        self.use_hierarchical = use_hierarchical
        self.num_hierarchical_stages = num_hierarchical_stages
        self.fourier = fourier
        self.relative_positional_encoding = relative_positional_encoding
        
        # Set up embedding layer based on fourier flag
        if self.fourier:
            # When fourier=True, embedding only processes features after the first 9
            self.embedding = nn.Linear(feature_dim - 9, embedding_dim)
            # Add positional fourier encoding
            self.fourier_encoder = PositionalFourierEncoding(
                in_dim=9,  # First 9 features (XYZ, normals, etc.)
                num_frequencies=8,  # Reduced from 16 to 8
                out_dim=embedding_dim
            )
            # Learnable scale for fourier encoding contribution
            self.pos_scale = nn.Parameter(torch.tensor(0.1))
        else:
            # When fourier=False, embedding processes all features
            self.embedding = nn.Linear(feature_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)  # Add dropout layer after embedding
        self.norm = nn.LayerNorm(embedding_dim)
        
        if use_hierarchical: 
             
            self.attention_blocks = nn.ModuleList([
                HierarchicalTransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=self.relative_positional_encoding) 
                for _ in range(num_attention_blocks)
            ])
        else:
            # Stack of original global-local transformer blocks
            self.attention_blocks = nn.ModuleList([
                G_L_TransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=self.relative_positional_encoding) 
                for _ in range(num_attention_blocks)
            ])
        
        self.apply(jepa_init_weights)

    def forward(self, input_tensor, masks_x):
        # When fourier=True, use features after the first 9 for embedding
        # When fourier=False, use all features for embedding
        if self.fourier:
            # Extract features after the first 9 (skip position features)
            features_to_embed = input_tensor[:, :, :, 9:].float()  # [B, N, T, feature_dim-9]
            # Extract first 9 features for fourier encoding
            position_features = input_tensor[:, :, :, :9].float()  # [B, N, T, 9]
        else:
            # Use all features for embedding
            features_to_embed = input_tensor.float()
            position_features = None
        
        # Embed input features
        embedded_input = self.embedding(features_to_embed)
        embedded_input = self.dropout(embedded_input)  # Apply dropout to the embedded input
        
        # Apply positional fourier encoding if enabled
        if self.fourier:
            print("Applying positional fourier encoding")
            B, N, T, _ = position_features.shape
            print("position_features.shape", position_features.shape)
            # Reshape for fourier encoding: [B*N, T, 9]
            position_flat = position_features.view(B * N, T, 9)
            
            # Apply fourier encoding
            fourier_encoded = self.fourier_encoder(position_flat)  # [B*N, T, embedding_dim]
            
            # Reshape back to original shape: [B, N, T, embedding_dim]
            fourier_encoded = fourier_encoded.view(B, N, T, -1)
            
            # Normalize fourier encoding to match embedded features scale
            fourier_encoded = F.layer_norm(fourier_encoded, fourier_encoded.shape[-1:])
            
            # Calculate and log weight contributions
            embedded_norm = torch.norm(embedded_input, dim=-1).mean().item()
            fourier_norm = torch.norm(fourier_encoded, dim=-1).mean().item()
            scaled_fourier_norm = torch.norm(self.pos_scale * fourier_encoded, dim=-1).mean().item()
            
            print(f"  Embedded features norm: {embedded_norm:.4f}")
            print(f"  Fourier features norm: {fourier_norm:.4f}")
            print(f"  Scaled fourier norm: {scaled_fourier_norm:.4f}")
            print(f"  Position scale value: {self.pos_scale.item():.4f}")
            print(f"  Fourier contribution ratio: {scaled_fourier_norm / (embedded_norm + 1e-8):.4f}")
            
            # Add scaled fourier encoding to embedded features
            embedded_input = embedded_input + self.pos_scale * fourier_encoded
            
            # Log final feature composition
            final_norm = torch.norm(embedded_input, dim=-1).mean().item()
            embedded_contribution = torch.norm(embedded_input - self.pos_scale * fourier_encoded, dim=-1).mean().item()
            fourier_contribution = torch.norm(self.pos_scale * fourier_encoded, dim=-1).mean().item()
            
            print(f"  === FINAL FEATURE COMPOSITION ===")
            print(f"  Final output norm: {final_norm:.4f}")
            print(f"  Embedded contribution: {embedded_contribution:.4f} ({100*embedded_contribution/final_norm:.1f}%)")
            print(f"  Fourier contribution: {fourier_contribution:.4f} ({100*fourier_contribution/final_norm:.1f}%)")
            print(f"  Position scale: {self.pos_scale.item():.4f}")
            print(f"  =================================")
        
        output = embedded_input

        # Prepare 3D positions for relative positional encoding, if enabled
        if self.relative_positional_encoding:
            if self.fourier:
                positions_3d = position_features[..., :3]  # [B, N, T, 3]
            else:
                positions_3d = input_tensor[..., :3].float()
        else:
            positions_3d = None

        # Initialize persistent CLS/summary token once and pass through blocks
        B, N, T, D = output.shape
        persistent_sum_token = None  # None -> first block will initialize from its learnable embedding

        # Pass through each attention block
        for attention_block in self.attention_blocks:
            if isinstance(attention_block, G_L_TransformerBlock):
                output, persistent_sum_token = attention_block(output, masks_x, sum_token_in=persistent_sum_token, positions_3d=positions_3d)
            elif isinstance(attention_block, HierarchicalTransformerBlock):
                # Hierarchical block currently does not support relative positions
                output, persistent_sum_token = attention_block(output, masks_x, sum_token_in=persistent_sum_token)
            else:
                # Fallback for any block conforming to (x, masks, sum_token_in)
                try:
                    output, persistent_sum_token = attention_block(output, masks_x, sum_token_in=persistent_sum_token)
                except TypeError:
                    output = attention_block(output, masks_x)
        
     #   output = self.norm(output)
     #   return F.normalize(output, dim=-1)
        return output

class PositionalFourierEncoding(nn.Module):
    def __init__(self, in_dim=9, num_frequencies=16, out_dim=384):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.linear = nn.Linear(2 * in_dim * num_frequencies, out_dim)

        # Precompute frequencies
        frequencies = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        # x: [B, N, 9] (includes XYZ, normals, etc.)
        x_exp = x.unsqueeze(-1) * self.frequencies  # [B, N, 9, num_frequencies]
        x_sin = torch.sin(2 * np.pi * x_exp)
        x_cos = torch.cos(2 * np.pi * x_exp)
        x_fourier = torch.cat([x_sin, x_cos], dim=-1)  # [B, N, 9, 2 * num_frequencies]
        x_flat = x_fourier.view(x.shape[0], x.shape[1], -1)  # [B, N, 9 * 2 * num_frequencies]
        return self.linear(x_flat)  # [B, N, out_dim]


# ===================== Nomeformer Predictor (JEPA-style) =====================
# Note: The following class assumes the existence of get_2d_sincos_pos_embed, apply_masks, and repeat_interleave_batch functions.
class NomeformerPredictor(nn.Module):
    """
    JEPA-style predictor built with nomeformer blocks.
    
    The predictor:
      1. Projects the input tokens (from an encoder) from embed_dim to predictor_embed_dim.
      2. Adds positional embeddings to the context tokens (using masks_x).
      3. Prepares learned mask tokens with their own positional embeddings (using masks).
      4. Concatenates context tokens and mask tokens.
      5. Processes them through a series of transformer blocks (G_L or Hierarchical).
      6. Extracts and projects the outputs corresponding to the mask tokens back to embed_dim.
    """
    def __init__(
        self,
        feature_dim,
        embed_dim=768,
        num_heads=1,
        num_attention_blocks=1,
        dropout=0.1,
        use_hierarchical=False,
        num_hierarchical_stages=1,
    ):
        super(NomeformerPredictor, self).__init__()
        # FIXED predictor dimension: half of embed_dim
        predictor_embed_dim = embed_dim // 2
        
        # Map from encoder embed_dim to predictor space
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        # MLP for processing first 9 features
        self.mlp_9 = nn.Sequential(
            nn.Linear(9, predictor_embed_dim),
            nn.LayerNorm(predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim)
        )
        self.fourier_encoder = PositionalFourierEncoding(
            in_dim=9,
            num_frequencies=16,
            out_dim=predictor_embed_dim
        )


        
        # Learned mask token (in predictor space)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        # Positional embedding
#        self.predictor_pos_embed = nn.Parameter(
#            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
#        )
        
        # Nomeformer blocks
        ff_dim = int(2 * predictor_embed_dim)
        self.use_hierarchical = use_hierarchical
        self.num_hierarchical_stages = num_hierarchical_stages
        
        if use_hierarchical:
            if num_hierarchical_stages > 1:
                # Use multi-stage hierarchical blocks
                self.attention_blocks = nn.ModuleList([
                    MultiStageHierarchicalTransformerBlock(
                        predictor_embed_dim, num_heads, ff_dim, dropout, 
                        num_stages=num_hierarchical_stages
                    ) 
                    for _ in range(num_attention_blocks)
                ])
            else:
                # Use single-stage hierarchical blocks
                self.attention_blocks = nn.ModuleList([
                    HierarchicalTransformerBlock(predictor_embed_dim, num_heads, ff_dim, dropout) 
                    for _ in range(num_attention_blocks)
                ])
        else:
            # Use original G_L blocks
            self.attention_blocks = nn.ModuleList([
                G_L_TransformerBlock(predictor_embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_attention_blocks)
            ])
        
        # Normalization and projection layers
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        # Initialize weights
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.apply(jepa_init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, batch, x, masks_x, masks):
        """
        Args:
            batch: Raw input tensor [B, N, D]
            x: Context encoder output [B, N, embed_dim]
            masks_x: Context masks
            masks: Target masks
        """
        # Ensure masks are lists
        
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        
        B = len(x) // len(masks_x)
        
        # Project encoder output to predictor space
        x = self.predictor_embed(x)
        
        # Process first 9 features from raw input
        
        first_9_inputs = batch[:, :, :,:9]  # Shape: [B, N, 9]
     #   flat = first_9_inputs.reshape(-1, 9)
     #   std_per_feature = flat.std(dim=0)
     #   print("STD for each of the 9 features:", std_per_feature)
     #   overall_std = first_9_inputs.std().item()
     #   print("Overall STD across all batch, cluster, face, and feature dimensions:", overall_std)
     #   mean_norm = flat.norm(dim=1).mean().item()
     #   print("Mean norm for pred_tokens:", mean_norm)
        first_9_inputs = first_9_inputs.float()
        B, N, C, D = first_9_inputs.shape
        flat_input = first_9_inputs.view(B * N * C, D)  # [B*N*C, 9]
    #    processed_9 = self.mlp_9(first_9_inputs)  # Shape: [B, N, predictor_embed_dim]
        first_9_inputs = first_9_inputs * 1  # Try different scales
        processed_9 = self.fourier_encoder(first_9_inputs.view(-1, first_9_inputs.shape[-2], 9))
        processed_9 = processed_9.view(*first_9_inputs.shape[:3], -1)  # [B, N, Fe, predictor_embed_dim]

    #    flat = processed_9.reshape(-1, processed_9.shape[-1])
    #    std_per_feature = flat.std(dim=0)
    #    print("after MLP STD for each of the processed_9 features:", std_per_feature)
    #    overall_std = processed_9.std().item()
    #    print("after MLP Overall STD across all batch, cluster, face, and processed_9 feature dimensions:", overall_std)
    #    mean_norm = flat.norm(dim=1).mean().item()
    #    print("Mean norm for pred_tokens:", mean_norm)
        
        # Add positional embedding to context tokens  
        N_ctxt = x.shape[1]
        # batch shape: [B, N, predictor_embed_dim]
        B, C, Fe,_ = batch.shape
        # Use repeat to create actual copies of mask_token
        mask_token_repeated = self.mask_token.repeat(B, C, Fe,1)  # [B, N, predictor_embed_dim]
        pred_tokens = self.mask_token.repeat(B, C, Fe, 1)
        pred_tokens = pred_tokens + processed_9  # Add all three components
     #   flat = pred_tokens.reshape(-1, processed_9.shape[-1])
     #   std_per_feature = flat.std(dim=0)
     #   print("final  STD for each of the processed_9 features:", std_per_feature)
     #   overall_std = processed_9.std().item()
     #   print("final Overall STD across all batch, cluster, face, and processed_9 feature dimensions:", overall_std)
     #   mean_norm = flat.norm(dim=1).mean().item()
     #   print("Mean norm for pred_tokens:", mean_norm)
        
        # Replace tokens in x with pred_tokens where masks is True
        Mb, Mrept , Mc,Mf = masks.shape 
        masks = masks.permute(0, 2, 1, 3).reshape(Mb, Mrept * Mc, Mf)

        mask = masks.bool().unsqueeze(-1)  # [B, N, Fe, 1]
        mask = mask.expand(-1, -1, -1, x.size(-1))  # [B, N, Fe, D]
        
        # Now, x and pred_tokens have shape [B, N, Fe, D], and so does mask.
      #  print('x',x.shape)
        x = torch.repeat_interleave(x, Mrept, axis=1)
        pred_tokens = torch.repeat_interleave(pred_tokens, Mrept, axis=1)
    #    print('x2',x.shape)
        x = torch.where(mask, pred_tokens, x)
     #   print('masks_x[0]',masks_x[0].shape)
        
        # Create attention mask from masks_x and update with masks
        attn_mask = masks_x[0].clone()  # Start with the first mask_x
        # Repeat each row Mrept times (along axis 0)
        attn_mask = torch.repeat_interleave(attn_mask, Mrept, axis=1)
        
#        for i, m in enumerate(masks):
        attn_mask = attn_mask | masks

        # Process through transformer blocks with persistent CLS/summary token
        persistent_sum_token = None
        for block in self.attention_blocks:
            if isinstance(block, G_L_TransformerBlock):
                x, persistent_sum_token = block(x, attn_mask, sum_token_in=persistent_sum_token)
            elif isinstance(block, HierarchicalTransformerBlock):
                x, persistent_sum_token = block(x, attn_mask, sum_token_in=persistent_sum_token)
            else:
                try:
                    x, persistent_sum_token = block(x, attn_mask, sum_token_in=persistent_sum_token)
                except TypeError:
                    x = block(x, attn_mask)
        
        # Final processing
        x = self.predictor_norm(x)
    #    x = x[:, N_ctxt:]  # Extract mask token predictions
        x = self.predictor_proj(x)  # Project back to original dimension
     #   return F.normalize(x, dim=-1)
        return x

        
        
        