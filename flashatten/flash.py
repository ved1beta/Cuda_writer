import torch, math
 
def flash_attention_pytorch(Q, K, V, block_size=16):
    """
    Compute attention scores with memory-efficient block-wise operations.
 
    Args:
        Q: Query matrix [N_out x d]
        K: Key matrix [N_inp x d]
        V: Value matrix [N_inp x d]
        block_size: Size of blocks for tiled computation
 
    Returns:
        O: Output matrix [N_out x d]
    """
    N_out, d = Q.shape
    N_inp = K.shape[0]
 
    # Initialize output tensors
    O = torch.zeros(N_out, d, device=Q.device)
    L = torch.zeros(N_out, 1, device=Q.device)
 
    # Calculate number of blocks needed
    T_c = (N_inp + block_size - 1) // block_size  # Ceiling division
    T_r = (N_out + block_size - 1) // block_size
 
    scale_factor = 1 / math.sqrt(d)
 
    # Process Q and O in T_r blocks; K, V in T_c blocks
    for i in range(T_r):
        # Get current block of queries
        q_start = i * block_size
        q_end = min((i + 1) * block_size, N_out)
        Q_block = Q[q_start:q_end]
 
        # Initialize block accumulators
        O_block = torch.zeros(q_end - q_start, d, device=Q.device)
        L_block = torch.zeros(q_end - q_start, 1, device=Q.device)
        m_block = torch.full((q_end - q_start, 1), float('-inf'), device=Q.device)
        last_m = m_block.clone()
 
        # Process K,V in blocks
        for j in range(T_c):
            k_start = j * block_size
            k_end = min((j + 1) * block_size, N_inp)
            K_block = K[k_start:k_end]
            V_block = V[k_start:k_end]
 
            # Compute attention scores for this block
            S_block = scale_factor * (Q_block @ K_block.T)  # [B_r x B_c]
 
            # Update running maximum for numerical stability
            m_block = torch.maximum(m_block, S_block.max(dim=-1, keepdim=True).values)
 
            # Compute attention weights with numerical stability
            P_block = torch.exp(S_block - m_block)  # [B_r x B_c]
 
            # Update accumulators with scaling factor from updated maximum
            scaling_factor = torch.exp(last_m - m_block)
            L_block = scaling_factor * L_block + P_block.sum(dim=-1, keepdim=True)
            O_block = scaling_factor * O_block + P_block @ V_block
 
            last_m = m_block.clone()
 
        # Store results for this block
        O[q_start:q_end] = O_block / L_block  # Normalize with accumulated sum
        L[q_start:q_end] = L_block
 
    return O