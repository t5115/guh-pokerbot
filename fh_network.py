# Adapted from advitrocks9/fullhouse-bot (MIT, Advit Arora 2026).
# See FULLHOUSE_LICENSE.txt. Original weights and feature layout preserved.
import numpy as np

def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.dot(x, weight.T) + bias

def _leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0.01 * x)

def _layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def _require_finite(name: str, x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return x

class Network:
    def __init__(self, weights):
        self.weights = weights

    def _forward_single(
        self, x: np.ndarray, legal_f: np.ndarray, model: dict
    ) -> np.ndarray:
        h = _require_finite(
            "trunk linear 0", _linear(x, model["trunk_w0"], model["trunk_b0"])
        )
        h = _require_finite(
            "trunk layer norm 0",
            _layer_norm(h, model["trunk_ln0_g"], model["trunk_ln0_b"]),
        )
        h = _require_finite("trunk activation 0", _leaky_relu(h))

        h = _require_finite(
            "trunk linear 1", _linear(h, model["trunk_w1"], model["trunk_b1"])
        )
        h = _require_finite(
            "trunk layer norm 1",
            _layer_norm(h, model["trunk_ln1_g"], model["trunk_ln1_b"]),
        )
        h = _require_finite("trunk activation 1", _leaky_relu(h))

        v = _require_finite(
            "value linear 0", _linear(h, model["val_w0"], model["val_b0"])
        )
        v = _require_finite("value activation", _leaky_relu(v))
        v = _require_finite(
            "value linear 1", _linear(v, model["val_w1"], model["val_b1"])
        )

        a = _require_finite(
            "adv linear 0", _linear(h, model["adv_w0"], model["adv_b0"])
        )
        a = _require_finite("adv activation", _leaky_relu(a))
        a = _require_finite(
            "adv linear 1", _linear(a, model["adv_w1"], model["adv_b1"])
        )

        a_masked = a * legal_f
        n_legal = legal_f.sum()
        if n_legal <= 0:
            raise ValueError("legal_mask has no legal actions")
        a_mean = a_masked.sum() / n_legal
        return _require_finite(
            "dueling output", (v + a_masked - a_mean * legal_f) * legal_f
        )

    def _strategy_single(
        self, x: np.ndarray, legal_f: np.ndarray, model: dict
    ) -> np.ndarray:
        h = _require_finite(
            "strategy linear 0", _linear(x, model["strategy_w0"], model["strategy_b0"])
        )
        h = _require_finite(
            "strategy layer norm 0",
            _layer_norm(h, model["strategy_ln0_g"], model["strategy_ln0_b"]),
        )
        h = _require_finite("strategy activation 0", _leaky_relu(h))

        h = _require_finite(
            "strategy linear 1", _linear(h, model["strategy_w1"], model["strategy_b1"])
        )
        h = _require_finite(
            "strategy layer norm 1",
            _layer_norm(h, model["strategy_ln1_g"], model["strategy_ln1_b"]),
        )
        h = _require_finite("strategy activation 1", _leaky_relu(h))

        logits = _require_finite(
            "strategy logits",
            _linear(h, model["strategy_out_w"], model["strategy_out_b"]),
        )
        masked = np.where(legal_f > 0, logits, -1e9)
        max_logit = np.max(masked[legal_f > 0])
        exp = np.exp(masked - max_logit) * legal_f
        total = exp.sum()
        if total > 0:
            return _require_finite("average strategy", (exp / total).astype(np.float32))
        n_legal = legal_f.sum()
        if n_legal <= 0:
            raise ValueError("legal_mask has no legal actions")
        return legal_f.astype(np.float32) / n_legal

    def _model_strategy(
        self, features: np.ndarray, legal_f: np.ndarray, model: dict
    ) -> np.ndarray:
        if model.get("_model_type") == "average_strategy":
            return self._strategy_single(features, legal_f, model)

        out = self._forward_single(features, legal_f, model)
        adv = _require_finite("positive regrets", np.maximum(out, 0.0) * legal_f)
        total = adv.sum()
        if total > 0:
            return _require_finite("strategy", (adv / total).astype(np.float32))
        n_legal = legal_f.sum()
        if n_legal <= 0:
            raise ValueError("legal_mask has no legal actions")
        return legal_f.astype(np.float32) / n_legal

    def strategy(self, features, legal):
        features = _require_finite("features", np.asarray(features, dtype=np.float32))
        if features.shape != (51,):
            raise ValueError("Fullhouse requires exactly 51 features")
        legal = np.asarray(legal, dtype=np.float32)
        return self._model_strategy(features, legal, self.weights)
