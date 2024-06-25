import onnx
from onnx import ModelProto

def load_onnx_model(file_path):
    # Load the ONNX model
    model = onnx.load(file_path)
    return model

def print_model_info(model):
    # Print the model's basic information
    print("Model IR Version: ", model.ir_version)
    print("Model Producer Name: ", model.producer_name)
    print("Model Producer Version: ", model.producer_version)
    print("Model Domain: ", model.domain)
    print("Model Model Version: ", model.model_version)
    print("Model Doc String: ", model.doc_string)
    
    # Print input and output information
    print("\nInputs:")
    for input_tensor in model.graph.input:
        print(f"Name: {input_tensor.name}, Shape: {input_tensor.type.tensor_type.shape}")

    print("\nOutputs:")
    for output_tensor in model.graph.output:
        print(f"Name: {output_tensor.name}, Shape: {output_tensor.type.tensor_type.shape}")
    
    # Print the nodes in the model
    print("\nNodes:")
    for node in model.graph.node:
        print(f"OpType: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")

def main(file_path):
    model = load_onnx_model(file_path)
    print_model_info(model)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX Model Info Extractor")
    parser.add_argument("onnx_file", type=str, help="Path to the ONNX file")
    args = parser.parse_args()

    main(args.onnx_file)
