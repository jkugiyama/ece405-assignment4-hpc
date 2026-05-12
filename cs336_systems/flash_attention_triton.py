import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    FlashAttention-2 forward kernel in Triton.
    
    Each Triton program instance processes one query tile and one batch index.
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index multiplied with the batch stride
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q tile into SRAM
    Q = tl.load(Q_block_ptr)

    # Initialize accumulators
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Get query indices for causal masking
    q_start = query_tile_index * Q_TILE_SIZE
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE)

    # Loop over key tiles
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(num_key_tiles):
        # Load K and V tiles
        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)

        # Compute attention scores: S = Q @ K^T * scale
        S = tl.dot(Q, tl.trans(K))
        S = S * scale

        # Apply causal mask if needed
        if is_causal:
            k_start = j * K_TILE_SIZE
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)
            # Mask where q_idx < k_idx (future tokens)
            mask = q_indices[:, None] >= k_indices[None, :]
            S = tl.where(mask, S, float("-inf"))

        # Compute row-wise max for numerical stability
        m_new = tl.max(S, axis=1)
        # Compute exponentials with stable numerics
        P = tl.exp(S - m_new[:, None])

        # Update m and l using the recurrence relation
        alpha = tl.exp(m - m_new)
        l = alpha * l + tl.sum(P, axis=1)
        O = alpha[:, None] * O + tl.dot(P, V)
        m = m_new

        # Advance block pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Finalize output
    O = O / l[:, None]
    # Cast to output dtype
    O = O.to(Q.dtype)

    # Store O and L
    tl.store(O_block_ptr, O)
    tl.store(L_block_ptr, tl.log(l) + m)


@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    FlashAttention-2 backward kernel in Triton.
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Set up block pointers for backward pass
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load forward pass data
    Q = tl.load(Q_block_ptr)
    O = tl.load(O_block_ptr)
    L = tl.load(L_block_ptr)
    dO = tl.load(dO_block_ptr)

    # Initialize dQ
    dQ = tl.zeros_like(Q)

    # Compute O * dO for each position
    O_dO = tl.sum(O * dO, axis=1)

    # Get query indices for causal masking
    q_start = query_tile_index * Q_TILE_SIZE
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE)

    # First loop: accumulate gradients for dQ
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    dK_accum = tl.zeros((N_KEYS, D), dtype=tl.float32)
    dV_accum = tl.zeros((N_KEYS, D), dtype=tl.float32)

    K_block_ptr_fwd = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr_fwd = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    for j in range(num_key_tiles):
        K = tl.load(K_block_ptr_fwd)
        V = tl.load(V_block_ptr_fwd)

        # Recompute attention
        S = tl.dot(Q, tl.trans(K))
        S = S * scale

        if is_causal:
            k_start = j * K_TILE_SIZE
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)
            mask = q_indices[:, None] >= k_indices[None, :]
            S = tl.where(mask, S, float("-inf"))

        P = tl.exp(S - L[:, None])
        dP = tl.dot(dO, tl.trans(V))
        dS = P * (dP - O_dO[:, None])

        dQ += tl.dot(dS, K) * scale
        dK_accum_tile = tl.dot(tl.trans(dS), Q) * scale
        dV_accum_tile = tl.dot(tl.trans(P), dO)

        K_block_ptr_fwd = tl.advance(K_block_ptr_fwd, (K_TILE_SIZE, 0))
        V_block_ptr_fwd = tl.advance(V_block_ptr_fwd, (K_TILE_SIZE, 0))

    # Store dQ
    tl.store(dQ_block_ptr, dQ.to(Q.dtype))


class FlashAttnTritonFunc(torch.autograd.Function):
    """
    FlashAttention-2 using Triton kernels for forward and backward.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.dim() == K.dim() == V.dim() == 3
        assert Q.device.type == "cuda"

        ctx.is_causal = is_causal
        B, T_q, D = Q.shape
        _, T_k, _ = K.shape

        # Tile sizes
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        # Allocate output tensors
        O = torch.empty_like(Q)
        L = torch.empty((B, T_q), device=Q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(D)

        # Launch grid: (number of query tiles, batch size)
        grid = (
            triton.cdiv(T_q, Q_TILE_SIZE),
            B,
        )

        # Call kernel
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            T_q, T_k,
            scale,
            D, Q_TILE_SIZE, K_TILE_SIZE,
            is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE

        B, T_q, D = Q.shape
        _, T_k, _ = K.shape

        scale = 1.0 / math.sqrt(D)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # Launch grid for backward
        grid = (
            triton.cdiv(T_q, Q_TILE_SIZE),
            B,
        )

        # Call backward kernel
        flash_bwd_kernel[grid](
            Q, K, V, O, L, dO,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            T_q, T_k,
            scale,
            D, Q_TILE_SIZE, K_TILE_SIZE,
            is_causal,
        )

        return dQ, dK, dV, None
