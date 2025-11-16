"""
Create a dummy ONNX model for backbone inference.
This simulates the xT-Grid + Temporal ConvNet model.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_dummy_backbone_model():
    """
    Create a simple ONNX model that takes tracking data and outputs xT grid.

    Input: [batch_size, features, 105, 68] - batched frames with features
    Output: [batch_size, 105, 68] - xT probability grid
    """

    # Define input
    input_tensor = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        [None, 8, 105, 68]  # batch_size, features (x, y, vel, dir, acc, etc.), height, width
    )

    # Define output
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        [None, 105, 68]  # batch_size, height, width
    )

    # Create a simple conv layer followed by reduction
    # Conv weights: (out_channels=16, in_channels=8, kernel_h=3, kernel_w=3)
    conv1_weights = np.random.randn(16, 8, 3, 3).astype(np.float32) * 0.1
    conv1_bias = np.zeros(16, dtype=np.float32)

    # Conv2 weights: (out_channels=1, in_channels=16, kernel_h=1, kernel_w=1)
    conv2_weights = np.random.randn(1, 16, 1, 1).astype(np.float32) * 0.1
    conv2_bias = np.zeros(1, dtype=np.float32)

    # Create initializers
    conv1_w_init = helper.make_tensor('conv1_weight', TensorProto.FLOAT,
                                       conv1_weights.shape, conv1_weights.flatten())
    conv1_b_init = helper.make_tensor('conv1_bias', TensorProto.FLOAT,
                                       conv1_bias.shape, conv1_bias.flatten())
    conv2_w_init = helper.make_tensor('conv2_weight', TensorProto.FLOAT,
                                       conv2_weights.shape, conv2_weights.flatten())
    conv2_b_init = helper.make_tensor('conv2_bias', TensorProto.FLOAT,
                                       conv2_bias.shape, conv2_bias.flatten())

    # Create nodes
    conv1_node = helper.make_node(
        'Conv',
        inputs=['input', 'conv1_weight', 'conv1_bias'],
        outputs=['conv1_output'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],  # same padding
        strides=[1, 1]
    )

    relu1_node = helper.make_node(
        'Relu',
        inputs=['conv1_output'],
        outputs=['relu1_output']
    )

    conv2_node = helper.make_node(
        'Conv',
        inputs=['relu1_output', 'conv2_weight', 'conv2_bias'],
        outputs=['conv2_output'],
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1]
    )

    # Sigmoid to get probabilities
    sigmoid_node = helper.make_node(
        'Sigmoid',
        inputs=['conv2_output'],
        outputs=['sigmoid_output']
    )

    # Reshape to remove channel dimension [batch, 1, 105, 68] -> [batch, 105, 68]
    shape_tensor = helper.make_tensor('reshape_shape', TensorProto.INT64, [3], [-1, 105, 68])

    reshape_node = helper.make_node(
        'Reshape',
        inputs=['sigmoid_output', 'reshape_shape'],
        outputs=['output']
    )

    # Create graph
    graph_def = helper.make_graph(
        nodes=[conv1_node, relu1_node, conv2_node, sigmoid_node, reshape_node],
        name='BackboneModel',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[conv1_w_init, conv1_b_init, conv2_w_init, conv2_b_init, shape_tensor]
    )

    # Create model with compatible IR version
    model_def = helper.make_model(graph_def, producer_name='fieldsense-ai')
    model_def.opset_import[0].version = 11  # Use opset 11 for compatibility
    model_def.ir_version = 8  # Use IR version 8 for broader compatibility

    # Check model
    onnx.checker.check_model(model_def)

    return model_def


if __name__ == '__main__':
    model = create_dummy_backbone_model()
    onnx.save(model, '/home/user/fieldsense-ai/assets/backbone.onnx')
    print("Dummy ONNX model created at /home/user/fieldsense-ai/assets/backbone.onnx")

    # Print model info
    print(f"Model size: {len(onnx._serialize(model)) / 1024:.2f} KB")
