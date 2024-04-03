include("gen_cuda.jl")

input_path = "/uufs/chpc.utah.edu/common/home/u1309986/bc-gpu/input_code1.jl"
input_code = read(input_path, String)
cuda_code = gen_bc_cuda(input_code, bc_index_vars=["a", "b", "c"], add_finch_code=false)
println(cuda_code)

#=

println("Importing")
using CUDA
println("done")

b = 2
max_a = 10
max_c = 15
max_d = 20

println("\n---\nSerial:")
i = 1
res1 = zeros(Int, max_a * max_c * max_d)
for a = 1:max_a
    for c = 1:max_c
        for d = 1:max_d
            r = a + b + c + d
            for j = 1:a
                r += 1
            end
            res1[i] = r
            global i += 1
        end
    end
end
println(res1)

println("\n---\nParallel:")
res2_d = CuArray{Int}(undef, max_a * max_c * max_d)
num_threads = 32
num_blocks = ceil(Int, max_a * max_c * max_d / num_threads)
eval(Meta.parse(cuda_code))
println("Launching")
@cuda blocks=num_blocks threads=num_threads f(max_a, b, max_c, max_d, res2_d)
println("done")
res2 = zeros(Int, max_a * max_c * max_d)
copyto!(res2, res2_d)
println(res2)

println("\n---\nEqual:")
println(res1 == res2)

=#