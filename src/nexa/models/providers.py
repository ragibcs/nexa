import onnxruntime

PROVIDER_ORDER = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

def get_providers(use_gpu: bool = False) -> list[str]:
    """Return the list of ONNX execution providers based on preference."""
    available = onnxruntime.get_available_providers()
    if use_gpu:
        return [p for p in PROVIDER_ORDER if p in available]
    return ["CPUExecutionProvider"]
