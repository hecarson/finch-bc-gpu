include("gen_cuda.jl")

input_path = "/uufs/chpc.utah.edu/common/home/u1309986/bc-gpu/input_code_isothermal.jl"
input_code = read(input_path, String)

bc_expr_args1 = Dict("temp" => "300")
bc_expr_args2 = Dict("temp" => "300 + 50*exp(-(x-262e-6)*(x-262e-6)/(5e-9))")

cuda_code1 = gen_bc_cuda(input_code, index_vars=["dir", "band"], add_finch_code=true, bi=1,
    bc_expr_args=bc_expr_args1, finch_pde_variables=["intensity"])
cuda_code2 = gen_bc_cuda(input_code, index_vars=["dir", "band"], add_finch_code=true, bi=2,
    bc_expr_args=bc_expr_args1, finch_pde_variables=["intensity"])
cuda_code3 = gen_bc_cuda(input_code, index_vars=["dir", "band"], add_finch_code=true, bi=3,
    bc_expr_args=bc_expr_args1, finch_pde_variables=["intensity"])
cuda_code4 = gen_bc_cuda(input_code, index_vars=["dir", "band"], add_finch_code=true, bi=4,
    bc_expr_args=bc_expr_args2, finch_pde_variables=["intensity"])

println(cuda_code1)
println()
println(cuda_code2)
println()
println(cuda_code3)
println()
println(cuda_code4)
println()