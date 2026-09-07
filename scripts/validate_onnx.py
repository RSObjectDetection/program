#!/usr/bin/env python3
"""Check an ONNX graph and benchmark raw ONNX Runtime inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = onnx.load(args.model)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    shape = [dimension if isinstance(dimension, int) else 1 for dimension in model_input.shape]
    sample = np.zeros(shape, dtype=np.float32)
    for _ in range(args.warmup):
        outputs = session.run(None, {model_input.name: sample})
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        outputs = session.run(None, {model_input.name: sample})
        timings.append((time.perf_counter() - started) * 1000)
    result = {
        "model": str(args.model.resolve()),
        "onnx_checker": "passed",
        "providers": session.get_providers(),
        "input_name": model_input.name,
        "input_shape": shape,
        "output_shapes": [list(output.shape) for output in outputs],
        "raw_inference_mean_ms": float(np.mean(timings)),
        "raw_inference_p95_ms": float(np.percentile(timings, 95)),
        "iterations": args.iterations,
        "note": "Raw graph execution only; preprocessing and NMS are excluded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
