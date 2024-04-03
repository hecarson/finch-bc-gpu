using CUDA
using BenchmarkTools
println("modules loaded")

function f_cuda(res_d)
    thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    i1 = ((thread_id - 1) ÷ (3 * 3)) % 3 + 1
    i2 = ((thread_id - 1) ÷ 3) % 3 + 1
    i3 = (thread_id - 1) % 3 + 1
    res_d[thread_id] = i1 + i2 + i3
    return
end

res_d = CuArray{Int}(undef, 3 * 3 * 3)
num_threads = 27
num_blocks = 1
println("compiling kernel")
kernel = @cuda launch=false f_cuda(res_d)
println("launching kernel")
@btime @sync kernel(blocks=num_blocks, threads=num_threads, res_d)

res = Array{Int}(undef, 3 * 3 * 3)
copyto!(res, res_d)
println(res)