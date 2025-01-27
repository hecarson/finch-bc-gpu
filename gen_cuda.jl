const INDENT = ' ' ^ 4

"""
Generates a CUDA kernel for one boundary.

index_vars: Parameters that should be treated as index vars, in loop order
add_finch_code: Whether Finch code should be added
bc_expr_args: Mapping from parameter names to expressions
finch_variables: Parameters that are Finch variables
"""
function gen_bc_cuda(input_code::String; index_vars::Vector{String} = Vector{String}(), add_finch_code::Bool = true,
bi::Int = 0, bc_expr_args::Dict{String, String} = Dict{String, String}(),
finch_variables::Vector{String} = Vector{String}())::String
    func_expr = Meta.parse(input_code)
    Base.remove_linenums!(func_expr)
    code_buffer = IOBuffer()

    gen_kernel_header(code_buffer, func_expr, index_vars, add_finch_code, bi, bc_expr_args)
    println(code_buffer, "@inbounds begin")

    # All index vars including Finch index vars, in loop order
    all_index_vars = copy(index_vars)
    if add_finch_code
        pushfirst!(all_index_vars, "fi")
    end
    gen_index_vars(code_buffer, bi, all_index_vars)

    println(code_buffer)
    if add_finch_code
        gen_bc_body_finch(code_buffer, func_expr, index_vars, bc_expr_args, finch_variables)
    else
        func_body = func_expr.args[2]
        gen_main_body(code_buffer, func_body)
        println(code_buffer)
        println(code_buffer, "result_vector[thread_id] = result")
    end
    
    println(code_buffer)
    println(code_buffer, "end # inbounds\nreturn\nend")

    return String(take!(code_buffer))
end

function gen_kernel_header(code_buffer::IOBuffer, func_expr::Expr, index_args::Vector{String},
add_finch_code::Bool, bi::Int, bc_expr_args::Dict{String, String})
    call_expr = func_expr.args[1]

    if add_finch_code
        print(code_buffer, "function $(call_expr.args[1])_bi_$(bi)_gpu(")
    else
        print(code_buffer, "function $(call_expr.args[1])_gpu(")
    end

    # Set of parameters that have a provided expression argument
    bc_expr_arg_params::Set{String} = Set()
    for param in keys(bc_expr_args)
        push!(bc_expr_arg_params, param)
    end
    
    kernel_params::Vector{String} = []

    for i = 2 : length(call_expr.args)
        param_str = get_param_str(call_expr.args[i])
        
        if param_str in index_args
            # Exclude index vars from kernel params, but add param for max value
            push!(kernel_params, "max_$(param_str)")
        elseif param_str in bc_expr_arg_params
            # Params with expression args will be given a value in the kernel body
            continue
        else
            push!(kernel_params, param_str)
        end
    end

    if add_finch_code
        push!(kernel_params, "max_fi")
        push!(kernel_params, "mesh_bdryface", "mesh_face2element", "mesh_bids", "geometric_factors_volume",
        "geometric_factors_area", "dofs_per_node", "faceCenters", "dim_faceCenters", "facex")
        if "t" ∉ kernel_params
            push!(kernel_params, "t")
        end
        push!(kernel_params, "boundary_flux", "boundary_dof_index", "global_vector")
    else
        push!(kernel_params, "result_vector")
    end

    print(code_buffer, join(kernel_params, ", "))
    println(code_buffer, ")")
    return
end

function get_param_str(param)::String
    if param isa Symbol
        return string(param)
    elseif param isa Expr
        # Assuming that arg is just a symbol and type, e.g. :(sym::Type)
        return string(param.args[1])
    else
        error("Unexpected arg node")
    end
end

"Generates index vars with a loop order that is the same as the order of the given index vars"
function gen_index_vars(code_buffer::IOBuffer, bi::Int, index_vars::Vector{String})
    # Map thread ID to (index1, index2, ...) tuple
    
    println(code_buffer, "thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)")

    index_var_maxes = ["max_$(index_var)" for index_var in index_vars]
    total_thread_ids = join(index_var_maxes, " * ")

    println(code_buffer, "if thread_id > $(total_thread_ids)")
    println(code_buffer, "$(INDENT)return")
    println(code_buffer, "end")

    println(code_buffer)
    println(code_buffer, "# All index vars")
    println(code_buffer, "bi = $(bi)")
    for i in eachindex(index_vars)
        index_var = index_vars[i]

        if i == length(index_vars)
            thread_ids_per_index = "1"
        else
            thread_ids_per_index = join(index_var_maxes[i + 1 : end], " * ")
        end

        if i == 1
            println(code_buffer, "$(index_var) = (thread_id - 1) ÷ ($(thread_ids_per_index)) + 1")
        else
            println(code_buffer, "$(index_var) = ((thread_id - 1) ÷ ($(thread_ids_per_index))) % max_$(index_var) + 1")
        end
    end

    return
end

function gen_main_body(code_buffer::IOBuffer, func_body::Expr)
    println(code_buffer, "# Main function body")
    for part in func_body.args
        println(code_buffer, part)
    end
    println(code_buffer)
end

function gen_bc_body_finch(code_buffer::IOBuffer, func_expr::Expr, bc_index_vars::Vector{String},
bc_expr_args::Dict{String, String}, finch_variables::Vector{String})
    println(code_buffer, "fid = mesh_bdryface[fi]")
    println(code_buffer, "eid = mesh_face2element[1,fid]");
    println(code_buffer, "fbid = mesh_bids[bi]");
    println(code_buffer, "volume = geometric_factors_volume[eid]");
    println(code_buffer, "area = geometric_factors_area[fid]");
    println(code_buffer, "area_over_volume = (area / volume)");

    # Index offset. Logic is from Finch src/generate_code_layer_julia_gpu.jl (search for "index_offset").
    index_offset_parts::Vector{String} = []
    index_var_maxes::Vector{String} = ["max_$(index_var)" for index_var in bc_index_vars]
    for i in eachindex(bc_index_vars)
        if i == 1
            multiplier = "1"
        else
            multiplier = join(index_var_maxes[begin : i - 1], " * ")
        end
        push!(index_offset_parts, "($(bc_index_vars[i]) - 1) * ($(multiplier))")
    end
    index_offset = join(index_offset_parts, " + ")
    if index_offset != ""
        println(code_buffer, "index_offset = $(index_offset)")
    end

    # Row index
    if index_offset != ""
        println(code_buffer, "row_index = index_offset + 1 + dofs_per_node * (eid - 1)")
    else
        println(code_buffer, "row_index = 1 + dofs_per_node * (eid - 1)")
    end

    # Finch-provided BC function args. This is incomplete and does not provide all values provided by Finch.
    println(code_buffer, "for i in 1:dim_faceCenters")
    println(code_buffer, "$(INDENT)facex[thread_id, i] = faceCenters[i, fid]")
    println(code_buffer, "end")
    println(code_buffer, "x = facex[thread_id, 1]")
    println(code_buffer, "y = dim_faceCenters >= 2 ? facex[thread_id, 2] : 0.0")
    println(code_buffer, "z = dim_faceCenters >= 2 ? facex[thread_id, 3] : 0.0")
    
    # Output expression arguments to BC
    for (param, arg) in bc_expr_args
        println(code_buffer, "$(param) = $(arg)")
    end
    
    println(code_buffer)
    func_body = deepcopy(func_expr.args[2])
    update_body_ast!(func_body, finch_variables)
    gen_main_body(code_buffer, func_body)

    println(code_buffer)
    println(code_buffer, "# Output")
    println(code_buffer, "boundary_flux[thread_id] = result * area_over_volume")
    println(code_buffer, "boundary_dof_index[thread_id] = row_index")
    println(code_buffer, "global_vector[row_index] = global_vector[row_index] + boundary_flux[thread_id]")

    return
end

"""
Update the main BC body AST for use in the GPU kernel

variable_params: list of parameters that are Finch variables
"""
function update_body_ast!(expr::Expr, finch_variables::Vector{String})
    # Some types of vector accesses need to have an additional index
    if expr.head == :ref
        var = expr.args[1]
        
        if string(var) in finch_variables
            push!(expr.args, :eid)
        elseif var == :normal
            push!(expr.args, :fid)
        end
    end

    # Remove "return result" statement
    filter!(part -> part != :(return result), expr.args)

    for i in eachindex(expr.args)
        part = expr.args[i]
        if typeof(part) == Expr
            update_body_ast!(part, finch_variables)
        end
    end
end