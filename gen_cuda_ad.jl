include("gen_cuda.jl")

input_path1 = "/uufs/chpc.utah.edu/common/home/u1309986/bc-gpu/input_code_ad1.jl"
input_path2 = "/uufs/chpc.utah.edu/common/home/u1309986/bc-gpu/input_code_ad2.jl"
input_code1 = read(input_path1, String)
input_code2 = read(input_path2, String)

cuda_code1 = gen_bc_cuda(input_code1, add_finch_code=true, bi=1)
cuda_code2 = gen_bc_cuda(input_code2, add_finch_code=true, bi=2)

println(cuda_code1)
println()
println(cuda_code2)
println()