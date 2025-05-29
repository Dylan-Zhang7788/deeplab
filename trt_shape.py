import tensorrt as trt

def load_engine_and_print_io(engine_path):
    # 初始化Logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 创建Runtime对象（仅传入logger）
    runtime = trt.Runtime(logger)
    
    # 加载引擎文件
    with open(engine_path, "rb") as f:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    
    # 获取输入/输出绑定索引
    input_binding = engine.get_binding_index("input")  # 根据实际名称修改
    output_binding = engine.get_binding_index("output")  # 根据实际名称修改
    
    # 获取动态形状信息（支持动态输入）
    context = engine.create_execution_context()
    input_shape = context.get_binding_shape(input_binding)
    output_shape = context.get_binding_shape(output_binding)
    
    # 转换为可读格式（去除批次维度）
    input_dims = [input_shape[1], input_shape[2], input_shape[3]]  # NCHW → CHW
    output_dims = list(output_shape[1:])  # 假设输出为类别+边界框
    
    print(f"[INFO] Input dimensions: Channels={input_dims[0]}, Height={input_dims[1]}, Width={input_dims[2]}")
    print(f"[INFO] Output dimensions: Channels={output_dims[0]}, Height={output_dims[1]}, Width={output_dims[2]}")

# 示例调用
load_engine_and_print_io("rdrnet_0529.engine")