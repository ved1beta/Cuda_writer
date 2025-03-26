constexpr int BLOCK_SIZE = 16;
constexpr int HIDDEN_DIM = 128;
 
extern "C" __global__ void flash_attention_cuda(
    float* output,      // [N_out x d]
    float* output_lse,  // [N_out]
    const float* query, // [N_out x d]
    const float* key,   // [N_inp x d]
    const float* value, // [N_inp x d]
    const float scale,
    const int N_out,
    const int N_inp) {
 
    // Shared memory for block-wise computation
    __shared__ float q_block[BLOCK_SIZE][HIDDEN_DIM];
    __shared__ float k_block[BLOCK_SIZE][HIDDEN_DIM];
    __shared__ float v_block[BLOCK_SIZE][HIDDEN_DIM];
 
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.x * BLOCK_SIZE + tx;
 
    // Local accumulators in registers
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[HIDDEN_DIM] = {0.0f};
 
    // Process input in tiles
    for (int tile = 0; tile < (N_inp + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Load query block (done once per outer loop)
        if (tile == 0 && row < N_out) {
            for (int d = 0; d < HIDDEN_DIM; d += blockDim.y) {
                int d_idx = d + ty;
                if (d_idx < HIDDEN_DIM) {
                    q_block[tx][d_idx] = query[row * HIDDEN_DIM + d_idx];
                }
            }
        }
        __syncthreads();
 
        // Load key and value blocks
        if (tile * BLOCK_SIZE + ty < N_inp && row < N_out) {
            for (int d = 0; d < HIDDEN_DIM; d += blockDim.y) {
                int d_idx = d + ty;
                if (d_idx < HIDDEN_DIM) {
                    k_block[tx][d_idx] = key[(tile * BLOCK_SIZE + tx) * HIDDEN_DIM + d_idx];
                    v_block[tx][d_idx] = value[(tile * BLOCK_SIZE + tx) * HIDDEN_DIM + d_idx];
                }
            }
        }
        __syncthreads();
 
        // Compute attention scores and update accumulators
        if (row < N_out) {
            float m_prev = m_i;
 
            // Compute scores and find max for stability
            float max_score = -INFINITY;
            float scores[BLOCK_SIZE];
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE && tile * BLOCK_SIZE + j < N_inp; ++j) {
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HIDDEN_DIM; ++d) {
                    score += q_block[tx][d] * k_block[j][d];
                }
                scores[j] = scale * score;
                max_score = max(max_score, scores[j]);
            }
 
            // Update running max and scale previous results
            m_i = max(m_i, max_score);
            float scale_factor = exp(m_prev - m_i);
            l_i *= scale_factor;
 
            #pragma unroll
            for (int d = 0; d < HIDDEN_DIM; ++d) {
                o_i[d] *= scale_factor;
            }
 
            // Compute attention and update output
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE && tile * BLOCK_SIZE + j < N_inp; ++j) {
                float p_ij = exp(scores[j] - m_i);
                l_i += p_ij;
 
                #pragma unroll
                for (int d = 0; d < HIDDEN_DIM; ++d) {
                    o_i[d] += p_ij * v_block[j][d];
                }
            }
        }
        __syncthreads();
    }
 
    // Write final output
    if (row < N_out) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < HIDDEN_DIM; ++d) {
            output[row * HIDDEN_DIM + d] = o_i[d] * inv_l;
        }
        output_lse[row] = l_i;
    }
}