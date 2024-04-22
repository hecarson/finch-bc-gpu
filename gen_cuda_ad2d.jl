include("gen_cuda.jl")

input_path = "/uufs/chpc.utah.edu/common/home/u1309986/bc-gpu/input_code_ad2d.jl"
input_code = read(input_path, String)

cuda_code = gen_bc_cuda(input_code, add_finch_code=true, bi=1)

println(cuda_code)