from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


def quant_encoder():
    fp32_path = Path("weights/encoder.onnx")
    int32_path = Path("weights/encoder_quant.onnx")

    quantize_dynamic(fp32_path, int32_path, weight_type=QuantType.QUInt8)


def quant_decoder():
    fp32_path = Path("weights/decoder.onnx")
    int32_path = Path("weights/decoder_quant.onnx")
    quantize_dynamic(fp32_path, int32_path, weight_type=QuantType.QUInt8)


def main():
    quant_encoder()
    quant_decoder()


if __name__ == "__main__":
    main()
