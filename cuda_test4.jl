using CUDA
using BenchmarkTools
println("modules loaded")

function f_cuda(res_d)
    thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    i = (thread_id - 1) ÷ 32 + 1
    # i = (thread_id - 1) % 16 + 1
    # res_d[i] += 1
    CUDA.@atomic res_d[i] += 1
    return
end

res_d = CUDA.zeros(Int, 16)

num_threads = 32
num_blocks = 16
println("compiling kernel")
kernel = @cuda launch=false f_cuda(res_d)
println("launching kernel")
# CUDA.@sync kernel(blocks=num_blocks, threads=num_threads, res_d)
# kernel(blocks=num_blocks, threads=num_threads, res_d)
CUDA.@profile kernel(blocks=num_blocks, threads=num_threads, res_d)
res = zeros(Int, 16)
copyto!(res, res_d)
println(res)