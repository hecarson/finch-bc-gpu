using CUDA
using NVTX
println("modules loaded")

function f_cuda(a_d, b_d)
    thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    a_d[thread_id] += b_d[thread_id]
    return
end

a = [i for i=1:1024]
b = [i for i=1024:-1:1]

a_d = CuArray(a)
b_d = CuArray(b)

num_threads = 256
num_blocks = 4
println("compiling kernel")
kernel = @cuda launch=false f_cuda(a_d, b_d)
println("launching kernel")
CUDA.@profile begin
    NVTX.@mark "kernel launch"
    kernel(blocks=num_blocks, threads=num_threads, a_d, b_d)
    NVTX.@mark "kernel done"
end
# println(a_d)