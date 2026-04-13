// ============================================================================
// FUSED FLASH ATTENTION — The Good Stuff
// ============================================================================
//
// This is the core of the project. A single CUDA kernel that computes:
//
//   O = softmax(Q @ K^T / sqrt(d)) @ V
//
// WITHOUT ever materialising the N×N attention matrix in HBM.
//
// The algorithm:
//   1. Load a tile of Q (Br rows) into shared memory — stays there for the
//      entire inner loop.
//   2. Loop over K and V in tiles of Bc rows:
//      a. Load K_tile and V_tile into shared memory
//      b. Compute S_tile = Q_tile @ K_tile^T in shared memory (Br × Bc)
//      c. Update running online softmax statistics (max, sum)
//      d. Accumulate output: O += softmax_weights @ V_tile
//   3. Final normalization: O /= running_sum
//
// Memory traffic (actual):
//   Q: read once from HBM                          = N·d
//   K: read once per Q tile = (N/Br) times          = N·d · (N/Br)
//   V: read once per Q tile = (N/Br) times          = N·d · (N/Br)
//   O: written once                                  = N·d
//   Total: O(N²·d / Br) per head
//
//   Still a massive win over naive's O(N²) intermediate traffic
//   because we never write/read the N×N attention matrix to HBM.
//   For N=4096, d=128, Br=32: we avoid 64MB of intermediate writes
//   at the cost of repeated K/V reads (each only N·d = 2MB).
//
//   Note: FlashAttention-2 improves this further by swapping the
//   loop order (outer over K/V, inner over Q) to reduce K/V reloads.
//
// Thread block mapping:
//   Grid:  (N / Br, B * H)    — one block per Q-tile per (batch, head)
//   Block: (Bc, Br)           — threads cooperate on tile computations
//
// ============================================================================

#include "flash_attn.cuh"

// ============================================================================
// The Fused Kernel
// ============================================================================

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,     // [B, H, N, d]
    const float* __restrict__ K,     // [B, H, N, d]
    const float* __restrict__ V,     // [B, H, N, d]
    float* __restrict__ O,           // [B, H, N, d]
    int B, int H, int N, int d,
    float scale                      // 1 / sqrt(d)
) {
    // ========================================================================
    // Step 0: Figure out what this thread block is responsible for
    // ========================================================================

    // blockIdx.x = which Q tile (along sequence dimension)
    // blockIdx.y = which (batch, head) pair
    int tile_idx = blockIdx.x;       // Which Br-sized chunk of queries
    int bh_idx   = blockIdx.y;       // Flattened (batch, head) index

    int b = bh_idx / H;             // Batch index
    int h = bh_idx % H;             // Head index

    // This thread's position within the tile
    int tx = threadIdx.x;           // Column index within tile (0..Bc-1)
    int ty = threadIdx.y;           // Row index within tile    (0..Br-1)

    // Global row index for this thread's query
    int q_row = tile_idx * TILE_BR + ty;

    // Base pointer for this (batch, head) slice
    // Layout: [B, H, N, d] → offset = ((b * H) + h) * N * d
    int bh_offset = (b * H + h) * N * d;
    const float* Qi = Q + bh_offset;
    const float* Ki = K + bh_offset;
    const float* Vi = V + bh_offset;
    float* Oi       = O + bh_offset;

    // ========================================================================
    // Step 1: Allocate shared memory for tiles
    // ========================================================================
    // We need 4 tiles in shared memory simultaneously:
    //   Q_tile: [Br, d]  — loaded once, reused across all K/V blocks
    //   K_tile: [Bc, d]  — reloaded each iteration
    //   V_tile: [Bc, d]  — reloaded each iteration
    //   S_tile: [Br, Bc] — the local attention scores (never leaves SRAM!)
    //
    // IMPORTANT: We use a fixed max dimension (128) for the d axis.
    // For your actual head dim, only [0..d-1] is used.

    const int MAX_D = 64;
    __shared__ float Q_tile[TILE_BR][MAX_D];
    __shared__ float K_tile[TILE_BC][MAX_D];
    __shared__ float V_tile[TILE_BC][MAX_D];
    __shared__ float S_tile[TILE_BR][TILE_BC];

    // ========================================================================
    // Step 2: Load Q tile into shared memory (stays for entire inner loop)
    // ========================================================================
    // Each thread loads one element of Q_tile. Since Q_tile is [Br, d] but
    // our thread block is [Bc, Br], we need multiple loads if d > Bc.
    //
    // Thread (tx, ty) loads Q_tile[ty][tx], Q_tile[ty][tx + Bc], etc.

    if (q_row < N) {
        for (int j = tx; j < d; j += TILE_BC) {
            Q_tile[ty][j] = Qi[q_row * d + j];
        }
    } else {
        // Zero-pad if this row is beyond sequence length (handles N % Br != 0)
        for (int j = tx; j < d; j += TILE_BC) {
            Q_tile[ty][j] = 0.0f;
        }
    }
    __syncthreads();

    // ========================================================================
    // Step 3: Online softmax state — per query row
    // ========================================================================
    // These are the running statistics for online softmax.
    // Each thread (ty) maintains the state for query row q_row.
    // But since multiple threads (different tx) share the same ty,
    // only one tx per ty needs to maintain the state. We let all of them
    // maintain it and converge — wasteful but simple. The key outputs
    // (row_max, row_sum, O_acc) are per-row, stored in registers.
    //
    // row_max: running maximum of all attention scores seen so far
    // row_sum: running sum of exp(score - row_max) across all K positions
    // O_acc:   running weighted sum, needs rescaling when row_max changes

    float row_max = -FLT_MAX;  // Will grow as we see more K blocks
    float row_sum = 0.0f;      // Will accumulate exp values

    // Per-thread output accumulator — each thread handles a subset of d dims
    // Thread tx handles dimensions tx, tx+Bc, tx+2*Bc, ...
    float O_acc[4] = {0};  // Supports up to d=128 with Bc=32 → 4 iterations
    // TODO: make this dynamic or use a larger static array for bigger d

    // ========================================================================
    // Step 4: Main loop — iterate over K/V blocks
    // ========================================================================
    // This is where online softmax happens. For each block of K and V:
    //   1. Load K_tile and V_tile into shared memory
    //   2. Compute S_tile = Q_tile @ K_tile^T (local attention scores)
    //   3. Find the local block max, update running max
    //   4. Rescale old accumulated values if max changed
    //   5. Accumulate new softmax weights × V_tile into output

    int num_kv_blocks = (N + TILE_BC - 1) / TILE_BC;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {

        // --- 4a. Load K_tile and V_tile ---
        // int kv_row = kv_block * TILE_BC + ty;  // Note: ty indexes within tile

        // Each thread loads elements for its row of K_tile and V_tile
        // Thread (tx, ty) loads K_tile[ty][tx + k*Bc] for k = 0, 1, ...
        // BUT: we need K_tile indexed by [Bc, d], and ty ranges 0..Br-1
        // Since Br might != Bc, we remap: use ty for the Bc dimension
        // Actually, let's use (tx, ty) more carefully:

        // For K and V loading, we want TILE_BC rows loaded.
        // Our block is (TILE_BC, TILE_BR) threads.
        // tx ranges [0, TILE_BC), ty ranges [0, TILE_BR).
        // We use tx as the "which row of K" and ty to stride over d.
        // Wait — that's confusing. Let's keep it simple:
        // Thread (tx, ty) loads K_tile[tx][ty...] where tx is the K row
        // and ty strides over the d dimension.

        // ACTUALLY: The cleanest approach — every thread cooperates to load.
        // Thread linear ID = ty * TILE_BC + tx, total threads = TILE_BR * TILE_BC.
        // Each thread loads one or more elements of K_tile[Bc][d].

        int linear_tid = ty * TILE_BC + tx;
        int total_threads = TILE_BR * TILE_BC;
        int k_elements = TILE_BC * d;  // Total elements to load for K_tile

        for (int i = linear_tid; i < k_elements; i += total_threads) {
            int kr = i / d;                          // Row in K_tile
            int kc = i % d;                          // Col in K_tile (dimension)
            int global_k_row = kv_block * TILE_BC + kr;
            if (global_k_row < N) {
                K_tile[kr][kc] = Ki[global_k_row * d + kc];
            } else {
                K_tile[kr][kc] = 0.0f;               // Zero-pad
            }
        }

        // Same for V_tile
        for (int i = linear_tid; i < k_elements; i += total_threads) {
            int vr = i / d;
            int vc = i % d;
            int global_v_row = kv_block * TILE_BC + vr;
            if (global_v_row < N) {
                V_tile[vr][vc] = Vi[global_v_row * d + vc];
            } else {
                V_tile[vr][vc] = 0.0f;
            }
        }
        __syncthreads();

        // --- 4b. Compute S_tile = Q_tile @ K_tile^T ---
        // S_tile[ty][tx] = dot(Q_tile[ty][:], K_tile[tx][:])
        // = sum over j of Q_tile[ty][j] * K_tile[tx][j]
        //
        // Note: K_tile is [Bc, d], and we want S = Q @ K^T.
        // S[i][j] = Q[i] · K[j], so S_tile[ty][tx] uses K_tile[tx][:]
        //
        // Thread (tx, ty) computes exactly one element of S_tile.

        float score = 0.0f;
        if (q_row < N) {
            for (int j = 0; j < d; j++) {
                score += Q_tile[ty][j] * K_tile[tx][j];
            }
            score *= scale;  // Don't forget the 1/sqrt(d) scaling!

            // Zero out scores for padding positions
            int global_k_col = kv_block * TILE_BC + tx;
            if (global_k_col >= N) {
                score = -FLT_MAX;  // Will become 0 after softmax
            }
        } else {
            score = -FLT_MAX;
        }
        S_tile[ty][tx] = score;
        __syncthreads();

        // --- 4c. Online softmax: find block max, update running max ---
        // Each thread ty needs the max across all tx values in its row.
        // We use a simple serial reduction since TILE_BC is small (32).

        if (q_row < N) {
            float block_max = -FLT_MAX;
            for (int j = 0; j < TILE_BC; j++) {
                block_max = fmaxf(block_max, S_tile[ty][j]);
            }

            // New running max = max(old running max, this block's max)
            float new_max = fmaxf(row_max, block_max);

            // --- 4d. Rescale old accumulated values ---
            // This is THE KEY STEP of online softmax.
            //
            // Everything we've accumulated so far was computed relative to
            // row_max. Now the max has changed to new_max. We need to
            // multiply by exp(row_max - new_max) to correct.
            //
            // If row_max == -FLT_MAX (first iteration), correction = 0,
            // which correctly zeroes out the (empty) old accumulator.

            float correction = expf(row_max - new_max);

            // Rescale the running sum and output accumulator
            row_sum *= correction;
            for (int i = 0; i < 4; i++) {
                O_acc[i] *= correction;
            }

            // --- 4e. Compute new softmax weights and accumulate ---
            // For each K position in this block:
            //   weight = exp(score - new_max)
            //   row_sum += weight
            //   O_acc += weight * V_tile[k_pos][:]
            //
            // We iterate over tx dimension serially (TILE_BC = 32, manageable).

            for (int j = 0; j < TILE_BC; j++) {
                float weight = expf(S_tile[ty][j] - new_max);
                row_sum += weight;

                // Accumulate weighted V contribution
                // Thread tx handles d dimensions: tx, tx+Bc, tx+2*Bc, ...
                int acc_idx = 0;
                for (int dim = tx; dim < d; dim += TILE_BC) {
                    O_acc[acc_idx] += weight * V_tile[j][dim];
                    acc_idx++;
                }
            }

            row_max = new_max;
        }
        __syncthreads();
    }

    // ========================================================================
    // Step 5: Final normalization and write output
    // ========================================================================
    // O_acc currently holds: sum_j [ exp(s_j - max) * V_j ]
    // We need to divide by row_sum to get the proper softmax-weighted average.

    if (q_row < N) {
        int acc_idx = 0;
        for (int dim = tx; dim < d; dim += TILE_BC) {
            Oi[q_row * d + dim] = O_acc[acc_idx] / row_sum;
            acc_idx++;
        }
    }
}

// ============================================================================
// Launch Function
// ============================================================================

void launch_flash_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d
) {
    float scale = 1.0f / sqrtf((float)d);

    // Grid dimensions:
    //   x = number of Q tiles along sequence dimension
    //   y = total (batch, head) pairs
    // Each thread block handles one Q tile for one (batch, head).

    dim3 grid((N + TILE_BR - 1) / TILE_BR, B * H);

    // Block dimensions:
    //   x = TILE_BC (threads iterate over K positions and d dimensions)
    //   y = TILE_BR (one thread-row per query row in the tile)

    dim3 block(TILE_BC, TILE_BR);

    // Shared memory is statically allocated in the kernel,
    // but if you needed more, you'd set it here:
    // int smem_size = (TILE_BR + 2 * TILE_BC) * MAX_D * sizeof(float) + ...;

    flash_attention_kernel<<<grid, block>>>(Q, K, V, O, B, H, N, d, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
}
