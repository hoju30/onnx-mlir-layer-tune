#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


def parse_shape(s: str) -> tuple[int, int, int, int]:
    t = s.replace("x", ",")
    parts = [p.strip() for p in t.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"shape must be N,C,H,W or NxCxHxW, got: {s}")
    out = tuple(int(p) for p in parts)
    return out  # type: ignore[return-value]


class TxtDirReader(CalibrationDataReader):
    def __init__(self, txt_dir: Path, input_name: str, shape: tuple[int, int, int, int], limit: int):
        self.txt_dir = txt_dir
        self.input_name = input_name
        self.shape = shape
        self.expect = int(np.prod(np.array(shape, dtype=np.int64)))
        files = sorted(txt_dir.glob("*.txt"))
        if limit > 0:
            files = files[:limit]
        self.files = files
        self._iter: Iterator[Path] | None = None

    def get_next(self):
        if self._iter is None:
            self._iter = iter(self.files)
        try:
            p = next(self._iter)
        except StopIteration:
            return None
        data = np.fromstring(p.read_text(), sep=" ", dtype=np.float32)
        if data.size != self.expect:
            raise RuntimeError(
                f"{p} has {data.size} values, expected {self.expect} for shape {self.shape}"
            )
        arr = data.reshape(self.shape)
        return {self.input_name: arr}


def main():
    ap = argparse.ArgumentParser(description="Generate int8 QDQ ONNX for ImageNet100 model.")
    ap.add_argument("--model-in", required=True, help="Input f32 ONNX")
    ap.add_argument("--model-out", required=True, help="Output QDQ ONNX")
    ap.add_argument("--txt-dir", required=True, help="Calibration txt folder")
    ap.add_argument("--input-name", default="input")
    ap.add_argument("--shape", default="1x3x224x224")
    ap.add_argument("--limit", type=int, default=256, help="Calibration sample count")
    ap.add_argument(
        "--activation-type",
        default="int8",
        choices=["int8", "uint8"],
        help="Activation quant type",
    )
    ap.add_argument(
        "--method",
        default="minmax",
        choices=["minmax", "entropy", "percentile"],
        help="Calibration method",
    )
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument(
        "--strip-output-qdq",
        action="store_true",
        default=True,
        help="Bypass trailing output QuantizeLinear/DequantizeLinear if present",
    )
    args = ap.parse_args()

    shape = parse_shape(args.shape)
    reader = TxtDirReader(Path(args.txt_dir), args.input_name, shape, args.limit)
    act_qt = QuantType.QInt8 if args.activation_type == "int8" else QuantType.QUInt8
    method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[args.method]

    quantize_static(
        model_input=args.model_in,
        model_output=args.model_out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=act_qt,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        per_channel=args.per_channel,
        extra_options={
            "ActivationSymmetric": args.activation_type == "int8",
            "WeightSymmetric": True,
        },
    )

    if args.strip_output_qdq:
        m = onnx.load(args.model_out)
        out2node = {}
        for n in m.graph.node:
            for o in n.output:
                out2node[o] = n
        changed = 0
        for out in m.graph.output:
            dq = out2node.get(out.name)
            if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 1:
                continue
            q = out2node.get(dq.input[0])
            if q is None or q.op_type != "QuantizeLinear" or len(q.input) < 1:
                continue
            out.name = q.input[0]
            changed += 1
        if changed:
            onnx.save(m, args.model_out)
            print(f"[ok] strip trailing output qdq: {changed} output(s)")
    print(f"[ok] wrote qdq model: {args.model_out}")


if __name__ == "__main__":
    main()
