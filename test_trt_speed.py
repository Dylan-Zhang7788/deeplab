import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

def load_engine(engine_file_path):
    """加载TensorRT引擎文件"""
    with open(engine_file_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, batch_size=1):
    """分配输入输出缓冲区"""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # 分配主机和设备内存
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    return inputs, outputs, bindings, stream

def inference(context, bindings, inputs, outputs, stream):
    """执行推理"""
    # 传输输入数据到GPU
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    # 执行推理
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 传输输出数据回主机
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()
    return outputs

def benchmark(engine_file, input_shape, num_runs=100, warmup=10):
    """性能基准测试"""
    # 加载引擎
    engine = load_engine(engine_file)
    context = engine.create_execution_context()
    
    # 分配缓冲区
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    # 生成随机输入数据
    np.random.seed(42)
    input_data = np.random.random(input_shape).astype(np.float32)
    inputs[0]['host'] = np.ravel(input_data)
    
    # 预热
    print("Warming up...")
    for _ in range(warmup):
        inference(context, bindings, inputs, outputs, stream)
    
    # 正式测试
    print("Benchmarking...")
    times = []
    for _ in range(num_runs):
        start = time.time()
        inference(context, bindings, inputs, outputs, stream)
        times.append(time.time() - start)
    
    # 计算统计信息
    avg_time = np.mean(times) * 1000  # 转换为毫秒
    fps = 1 / np.mean(times)
    print(f"Average inference time: {avg_time:.2f}ms")
    print(f"Throughput: {fps:.2f}FPS")
    print(f"Min time: {np.min(times)*1000:.2f}ms")
    print(f"Max time: {np.max(times)*1000:.2f}ms")
    
    return times

if __name__ == "__main__":
    # 参数配置
    ENGINE_FILE = "seg_model_resnet_0529.trt"  # 替换为你的引擎文件路径
    INPUT_SHAPE = (1, 3, 768,1024)  # 替换为你的模型输入形状
    
    # 运行基准测试
    benchmark(ENGINE_FILE, INPUT_SHAPE, num_runs=100, warmup=10)