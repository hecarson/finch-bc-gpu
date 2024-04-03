include("gen_cuda.jl")

input_path = "/uufs/chpc.utah.edu/common/home/u1309986/bc-gpu/input_code2.jl"
input_code = read(input_path, String)
bc_expr_args_map=[
    Dict("temp" => "300"),
    Dict("temp" => "300"),
    Dict("temp" => "300"),
    Dict("temp" =>"300 + 50*exp(-(x-262e-6)*(x-262e-6)/(5e-9))")
]
cuda_code = gen_bc_cuda(input_code, index_vars=["band", "dir"], add_finch_code=true, bc_expr_args_map=bc_expr_args_map,
finch_variables=["intensity"])
println(cuda_code)