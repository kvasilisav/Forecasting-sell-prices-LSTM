echo "Pulling ONNX model from DVC remote..."
dvc pull -r model

if [ ! -f "model/checkpoints/model.onnx" ]; then
    echo "Error: Failed to download model.onnx from DVC remote!"
    exit 1
fi

echo "Converting ONNX model to TensorRT..."
trtexec --onnx=model/checkpoints/model.onnx \
        --saveEngine=model/checkpoints/model.plan \
        --explicitBatch \
        --inputIOFormats=fp32:chw \
        --outputIOFormats=fp32:chw \
        --workspace=2048
