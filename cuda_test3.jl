using CUDA
using BenchmarkTools
println("modules loaded")

function f_cuda(mat_d, res_d)
    thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    res_d[thread_id] = 0
    for i::Int64=1:16
        for j::Int64=1:16
            res_d[thread_id] += mat_d[i, j]
        end
    end
    return
end

mat_d = CUDA.ones(Int, 16, 16)
res_d = CUDA.zeros(Int, 64)

num_threads = 64
num_blocks = 1
println("compiling kernel")
kernel = @cuda launch=false f_cuda(mat_d, res_d)
println("launching kernel")
@btime @sync kernel(blocks=num_blocks, threads=num_threads, mat_d, res_d)

res = zeros(Int, 64)
copyto!(res, res_d)
println(res)