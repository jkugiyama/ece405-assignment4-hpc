from dataclasses import dataclass
import math
import torch
from einops import einsum


class FlashAttnTorchFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.dim() == K.dim() == V.dim() == 3

        ctx.is_causal = is_causal
        B, T, D = Q.shape

        O = torch.empty_like(Q)
        L = torch.empty((B, T), device=Q.device, dtype=torch.float32)

        B_q = FlashAttnTorchFunc._auto_batch_size(T)
        B_k = FlashAttnTorchFunc._auto_batch_size(T)

        T_q = math.ceil(T / B_q)

        for b in range(B):
            for i in range(T_q):
                args = FlashAttnTorchFunc._ForwardArgs(
                    b, i, Q, K, V, O, L, B_q, B_k, is_causal
                )
                FlashAttnTorchFunc._forward_kernel(args)

        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        dO = grad_outputs[0]
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = FlashAttnTorchFunc._backward_impl(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None

    # ---------------- FORWARD ----------------

    @staticmethod
    def _forward_kernel(args):
        Q, K, V = args.Q, args.K, args.V
        b = args.b

        T, D = Q.shape[1], Q.shape[2]

        q0, q1 = FlashAttnTorchFunc._boundary(args.i, args.B_q, T)
        Q_tile = Q[b, q0:q1]

        m = torch.full((2, q1 - q0), float("-inf"), device=Q.device, dtype=torch.float32)
        l = torch.zeros((q1 - q0,), device=Q.device, dtype=torch.float32)
        o = torch.zeros((q1 - q0, D), device=Q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(D)

        for j in range(math.ceil(T / args.B_k)):
            k0, k1 = FlashAttnTorchFunc._boundary(j, args.B_k, T)

            if args.is_causal and k0 >= q1:
                continue

            K_tile = K[b, k0:k1]
            V_tile = V[b, k0:k1]

            S = (Q_tile @ K_tile.T) * scale

            if args.is_causal and k1 > q0:
                qi = torch.arange(q0, q1, device=S.device)[:, None]
                kj = torch.arange(k0, k1, device=S.device)[None, :]
                S = S.masked_fill(qi < kj, float("-inf"))

            m_new = torch.maximum(m[0], S.max(dim=-1).values)
            P = torch.exp(S - m_new.unsqueeze(-1))

            alpha = torch.exp(m[0] - m_new)

            l = alpha * l + P.sum(dim=-1)
            o = alpha.unsqueeze(-1) * o + P @ V_tile

            m[0], m[1] = m_new, m[0]

        args.O[b, q0:q1] = (o / l.unsqueeze(-1)).to(args.O.dtype)
        args.L[b, q0:q1] = torch.log(l) + m[0]

    # ---------------- BACKWARD ----------------

    @staticmethod
    def _backward_impl(Q, K, V, O, dO, L, is_causal):
        B, T, _ = Q.shape

        # Equation 16: D_i = dO_i · O_i
        D_vec = (O * dO).sum(dim=-1)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        B_q = FlashAttnTorchFunc._auto_batch_size(T)
        B_k = FlashAttnTorchFunc._auto_batch_size(T)

        T_q = math.ceil(T / B_q)
        T_k = math.ceil(T / B_k)

        for b in range(B):
            for i in range(T_q):
                args = FlashAttnTorchFunc._BackwardArgs(
                    b, i, 0, Q, K, V, O, L, dO, D_vec, dQ, dK, dV, B_q, B_k, is_causal
                )
                FlashAttnTorchFunc._backward_first(args)

        for b in range(B):
            for j in range(T_k):
                args = FlashAttnTorchFunc._BackwardArgs(
                    b, 0, j, Q, K, V, O, L, dO, D_vec, dQ, dK, dV, B_q, B_k, is_causal
                )
                FlashAttnTorchFunc._backward_second(args)

        return dQ, dK, dV

    @staticmethod
    @torch.compile(fullgraph=False, backend="aot_eager")
    def _backward_tile(Q_tile, K_tile, V_tile, O_tile, dO_tile, L_tile, D_tile, qi, kj, is_causal, scale):
        S = (Q_tile @ K_tile.T) * scale

        if is_causal:
            S = S.masked_fill(qi < kj, float("-inf"))

        P = torch.exp(S - L_tile.unsqueeze(-1))
        dP = dO_tile @ V_tile.T
        dS = P * (dP - D_tile.unsqueeze(-1))

        dQ_tile = (dS @ K_tile) * scale
        dK_tile = (dS.T @ Q_tile) * scale
        dV_tile = P.T @ dO_tile

        return dQ_tile, dK_tile, dV_tile

    @staticmethod
    def _backward_first(args):
        Q, K, V = args.Q, args.K, args.V
        b = args.b
        B_q, B_k = args.B_q, args.B_k

        T, D = Q.shape[1], Q.shape[2]
        scale = 1.0 / math.sqrt(D)

        q0, q1 = FlashAttnTorchFunc._boundary(args.i, B_q, T)
        Q_tile = Q[b, q0:q1]
        O_tile = args.O[b, q0:q1]
        dO_tile = args.dO[b, q0:q1]
        L_tile = args.L[b, q0:q1]
        D_tile = args.D_vec[b, q0:q1]

        for j in range(math.ceil(T / B_k)):
            k0, k1 = FlashAttnTorchFunc._boundary(j, B_k, T)

            if args.is_causal and k0 >= q1:
                continue

            K_tile = K[b, k0:k1]
            V_tile = V[b, k0:k1]
            qi = torch.arange(q0, q1, device=Q.device)[:, None]
            kj = torch.arange(k0, k1, device=Q.device)[None, :]

            dQ_tile, _, _ = FlashAttnTorchFunc._backward_tile(
                Q_tile,
                K_tile,
                V_tile,
                O_tile,
                dO_tile,
                L_tile,
                D_tile,
                qi,
                kj,
                args.is_causal and k1 > q0,
                scale,
            )
            args.dQ[b, q0:q1] += dQ_tile

    @staticmethod
    def _backward_second(args):
        Q, K, V = args.Q, args.K, args.V
        b = args.b

        T, D = K.shape[1], K.shape[2]
        scale = 1.0 / math.sqrt(D)

        k0, k1 = FlashAttnTorchFunc._boundary(args.j, args.B_k, T)
        K_tile = K[b, k0:k1]
        V_tile = V[b, k0:k1]

        T_q = math.ceil(T / args.B_q)

        for i in range(T_q):
            q0, q1 = FlashAttnTorchFunc._boundary(i, args.B_q, T)

            if args.is_causal and k0 >= q1:
                continue

            Q_tile = Q[b, q0:q1]
            O_tile = args.O[b, q0:q1]
            dO_tile = args.dO[b, q0:q1]
            L_tile = args.L[b, q0:q1]
            D_tile = args.D_vec[b, q0:q1]
            qi = torch.arange(q0, q1, device=Q.device)[:, None]
            kj = torch.arange(k0, k1, device=Q.device)[None, :]

            _, dK_tile, dV_tile = FlashAttnTorchFunc._backward_tile(
                Q_tile,
                K_tile,
                V_tile,
                O_tile,
                dO_tile,
                L_tile,
                D_tile,
                qi,
                kj,
                args.is_causal and k1 > q0,
                scale,
            )

            args.dV[b, k0:k1] += dV_tile
            args.dK[b, k0:k1] += dK_tile

    # ---------------- utils ----------------

    @staticmethod
    def _boundary(i, tile, T):
        s = i * tile
        return s, min(s + tile, T)

    @staticmethod
    def _auto_batch_size(T):
        return 32 if T > 32 else T

    # ---------------- dataclasses ----------------

    @dataclass
    class _ForwardArgs:
        b: int; i: int
        Q: torch.Tensor; K: torch.Tensor; V: torch.Tensor
        O: torch.Tensor; L: torch.Tensor
        B_q: int; B_k: int; is_causal: bool

    @dataclass
    class _BackwardArgs:
        b: int; i: int; j: int
        Q: torch.Tensor; K: torch.Tensor; V: torch.Tensor
        O: torch.Tensor; L: torch.Tensor
        dO: torch.Tensor; D_vec: torch.Tensor
        dQ: torch.Tensor; dK: torch.Tensor; dV: torch.Tensor
        B_q: int; B_k: int; is_causal: bool
