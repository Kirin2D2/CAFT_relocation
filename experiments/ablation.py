"""Mean-ablation of a subspace on the residual stream, as a forward hook.

h <- h - (h U) U^T + mu U^T   at every position, for each hooked layer.
Directions file: {"layers": {int: {"U": [d,k] orthonormal, "mu": [k]}}, "meta": {...}}.
U must already be orthonormal (asserted; no QR here, so mu's basis is unambiguous).
The projection is computed in fp32 and cast back to the activation dtype.
Hooks re-fire under gradient-checkpoint recompute; U/mu are constants, so that is safe.
"""
import torch as t


class MeanAblator:
    def __init__(self, directions, model, name="ablator"):
        self.model, self.name = model, name
        self.U, self.mu = {}, {}
        dev = next(model.parameters()).device
        for l, d in directions["layers"].items():
            U = d["U"].float()
            if U.dim() == 1:
                U = U[:, None]
            gram = U.T @ U
            assert t.allclose(gram, t.eye(U.shape[1]), atol=1e-3), f"layer {l}: U not orthonormal"
            self.U[int(l)] = U.to(dev).detach()
            self.mu[int(l)] = d["mu"].float().reshape(-1).to(dev).detach()
        self.layers = sorted(self.U)
        self.handles = []

    def _hook(self, l):
        U, mu = self.U[l], self.mu[l]
        shift = (mu @ U.T)                      # [d]
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            hf = hf - (hf @ U) @ U.T + shift
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    def enable(self):
        if self.handles:
            return self
        for l in self.layers:
            self.handles.append(self.model.model.layers[l].register_forward_hook(self._hook(l)))
        return self

    def disable(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        return self

    @property
    def active(self):
        return bool(self.handles)

    def __enter__(self):
        return self.enable()

    def __exit__(self, *a):
        self.disable()


def load_directions(path):
    return t.load(path, weights_only=False)


def save_directions(dirs, path):
    t.save(dirs, path)
    return path


def slice_dirs(dirs, layers, k):
    """(layers, k) slice of a rank-kmax directions file; source tag encodes the dose."""
    out = {"layers": {}, "meta": {**{kk: v for kk, v in dirs["meta"].items() if kk != "stats"},
                                  "k": k, "layers": list(layers),
                                  "source": f"{dirs['meta']['source']}_L{len(layers)}_k{k}"}}
    for l in layers:
        d = dirs["layers"][l]
        out["layers"][l] = {"U": d["U"][:, :k].contiguous(), "mu": d["mu"][:k].contiguous()}
    return out
