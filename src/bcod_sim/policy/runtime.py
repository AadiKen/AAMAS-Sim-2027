"""Constrained ONNX Runtime inference path."""

from typing import Mapping

import numpy as np

from bcod_sim.core.errors import PolicyContractMismatchError
from bcod_sim.policy.bundle import PolicyBundle


class ONNXPolicyRuntime:
    def __init__(self, bundle: PolicyBundle) -> None:
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(bundle.root/"model.onnx"),
                                                providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise PolicyContractMismatchError("ONNX model failed constrained runtime validation") from exc
        self.bundle = bundle
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        self.output_names = tuple(item.name for item in self.session.get_outputs())
        if not self.input_names or not self.output_names:
            raise PolicyContractMismatchError("ONNX model requires declared input and output")
        if (self.input_names != tuple(bundle.observation_contract["model_inputs"]) or
                self.output_names != tuple(bundle.action_contract["model_outputs"])):
            raise PolicyContractMismatchError("ONNX input/output bindings disagree with policy contracts")

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        if set(inputs) != set(self.input_names):
            raise PolicyContractMismatchError("ONNX input names do not match model contract")
        if self.bundle.preprocessing != {"operation": "identity"}:
            raise PolicyContractMismatchError("Unsupported preprocessing operation")
        mean = np.asarray(self.bundle.normalization.get("mean"), dtype=np.float32)
        scale = np.asarray(self.bundle.normalization.get("scale"), dtype=np.float32)
        if mean.ndim != 1 or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise PolicyContractMismatchError("Invalid normalization statistics")
        prepared = {}
        for name, value in inputs.items():
            array = np.asarray(value)
            if array.shape[-1:] != mean.shape:
                raise PolicyContractMismatchError("Policy input shape disagrees with normalization")
            prepared[name] = (array-mean)/scale
        values = self.session.run(list(self.output_names), prepared)
        return dict(zip(self.output_names, values))
