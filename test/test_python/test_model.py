import simpleinfer as infer
import numpy as np
import os

if __name__ == '__main__':
    infer.InitializeContext()

    engine = infer.Engine()

    cur_file_path = os.path.dirname(os.path.realpath(__file__))
    model_path = cur_file_path + '/../../3rdparty/tmp/yolo/demo/'
    rc = engine.LoadModel(model_path + 'yolov5n_small.pnnx.param',
                          model_path + 'yolov5n_small.pnnx.bin')
    print('LoadModel', rc)

    input_names = engine.InputNames()
    output_names = engine.OutputNames()
    print('input_names: ', input_names)
    print('output_names: ', output_names)

    # input
    input_shape = [4, 320, 320, 3] # NHWC
    input_np = np.ones(input_shape, dtype = np.float32) * 42.0
    input_tensor = infer.Tensor(infer.DataType.Float32, input_shape)
    rc = input_tensor.SetTensorDim4(input_np)
    print('SetTensorDim4', rc)

    rc = engine.Input(input_names[0], input_tensor)
    print('Input', rc)

    rc = engine.Forward()
    print('Forward', rc)

    output_tensor = infer.Tensor()
    rc = engine.Extract(output_names[0], output_tensor)
    print('Extract', rc)

    output_np = output_tensor.GetTensorDim4()
    print(output_np.dtype, output_np.shape)

    print(output_np[0, 0, 0, :])
