using CUDA
using BenchmarkTools
println("modules loaded")

function f_cuda(a_max, b_max, res)
    thread_idx = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    a = (thread_idx - 1) % a_max + 1
    b = (thread_idx - 1) ÷ a_max + 1

    if b > b_max
        return
    end
    
    res[a_max * (b - 1) + a] = a + b
    return
end

a_max = 100
b_max = 100
res_d = CuArray{Int}(undef, a_max * b_max)
num_threads = 256
num_blocks = ceil(Int, a_max * b_max / num_threads)
println("compiling kernel")
kernel = @cuda launch=false f_cuda(a_max, b_max, res_d)
println("launching kernel")
@btime @sync kernel(threads=num_threads, blocks=num_blocks, a_max, b_max, res_d)

# res = Array{Int}(undef, a_max * b_max)
# res = copyto!(res, res_d)
# for i in eachindex(res)
#     a = (i - 1) % a_max + 1
#     b = (i - 1) ÷ a_max + 1
#     if res[i] != a + b
#         println("error")
#         break
#     end
# end