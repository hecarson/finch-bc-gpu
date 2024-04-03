#=
Generated functions for FVbte2dgpu
=#

#=
Auxilliary code that will be included in Finch
=#

function gpu_assembly_kernel(mesh_elemental_order_gpu, variables_3_values_gpu, variables_2_values_gpu, variables_1_values_gpu, geometric_factors_volume_gpu,
    mesh_element2face_gpu, mesh_facebid_gpu, mesh_face2element_gpu, mesh_facenormals_gpu, fv_info_faceCenters_gpu,
    coefficients_3_value_gpu, coefficients_1_value_gpu, coefficients_2_value_gpu, geometric_factors_area_gpu, global_vector_gpu,
    dofs_global, faces_per_element, index_ranges)

    # The dofs handled by this thread are determined by the global thread ID
    # max_threads = blockDim().x * gridDim().x * blockDim().y * gridDim().y * blockDim().z * gridDim().z;
    # block_id = blockIdx().x + (blockIdx().y - 1) * gridDim().x + (blockIdx().z - 1) * gridDim().x * gridDim().y;
    # thread_id = threadIdx().x + (block_id - 1) * blockDim().x * blockDim().y * blockDim().z;

    # simplified version with 1D grid and blocks
    max_threads = blockDim().x * gridDim().x
    block_id = blockIdx().x
    thread_id = threadIdx().x + (block_id - 1) * blockDim().x

    # Index counts
    num_direction = index_ranges[3]
    num_band = index_ranges[2]
    num_elements = index_ranges[1]


    # Loop over all dofs assigned to this thread
    # strided by max_threads
    current_dof = thread_id
    while current_dof <= dofs_global

        # extract index values
        INDEX_VAL_direction = Int(mod(current_dof - 1, num_direction) + 1)
        INDEX_VAL_band = Int(floor(mod(current_dof - 1, num_band * num_direction) / (num_direction)) + 1)
        eid = Int(floor(mod(current_dof - 1, num_elements * num_direction * num_band) / (num_direction * num_band)) + 1)


        #= Begin assembly code =#
        source_gpu = 0.0
        flux_gpu = 0.0
        #= Evaluate volume coefficients. =#
        value__beta_band_gpu = variables_3_values_gpu[INDEX_VAL_band, eid]
        value__Io_band_gpu = variables_2_values_gpu[INDEX_VAL_band, eid]
        value__I_directionband_gpu = variables_1_values_gpu[(INDEX_VAL_direction+(4*(INDEX_VAL_band-1))), eid]
        volume = geometric_factors_volume_gpu[eid]
        #= Compute source terms (volume integral) =#
        source_gpu = ((value__beta_band_gpu * value__Io_band_gpu) + ((-value__beta_band_gpu) * value__I_directionband_gpu))

        #= Compute flux terms (surface integral) in a face loop =#
        for fi = 1:faces_per_element
            flux_tmp_gpu = 0.0
            #=  =#
            fid = mesh_element2face_gpu[fi, eid]
            fbid_gpu = mesh_facebid_gpu[fid]
            left_el_gpu = mesh_face2element_gpu[1, fid]
            right_el_gpu = mesh_face2element_gpu[2, fid]
            neighbor = left_el_gpu
            out_side_gpu = 1
            if ((eid == left_el_gpu) && (right_el_gpu > 0))
                neighbor = right_el_gpu
                out_side_gpu = 2
            end

            #= Evaluate surface coefficients. =#
            if (eid == left_el_gpu)
                normal_sign_gpu = 1
                in_side_gpu = 1

            else
                normal_sign_gpu = -1
                in_side_gpu = 2
            end

            FACENORMAL1_1_gpu = (normal_sign_gpu * mesh_facenormals_gpu[1, fid])
            FACENORMAL2_1_gpu = (-FACENORMAL1_1_gpu)
            FACENORMAL1_2_gpu = (normal_sign_gpu * mesh_facenormals_gpu[2, fid])
            FACENORMAL2_2_gpu = (-FACENORMAL1_2_gpu)
            x = fv_info_faceCenters_gpu[1, fid]
            y = fv_info_faceCenters_gpu[2, fid]
            z = 0.0
            value__vg_band_gpu = coefficients_3_value_gpu[INDEX_VAL_band]
            value__Sx_direction_gpu = coefficients_1_value_gpu[INDEX_VAL_direction]
            value__Sy_direction_gpu = coefficients_2_value_gpu[INDEX_VAL_direction]
            value_CELL1_I_directionband_gpu = variables_1_values_gpu[(INDEX_VAL_direction+(4*(INDEX_VAL_band-1))), eid]
            value_CELL2_I_directionband_gpu = variables_1_values_gpu[(INDEX_VAL_direction+(4*(INDEX_VAL_band-1))), neighbor]
            area = geometric_factors_area_gpu[fid]
            area_over_volume = (area / volume)
            #= Compute flux terms (surface integral) =#
            flux_tmp_gpu = ((-value__vg_band_gpu) * (((((value__Sx_direction_gpu * FACENORMAL1_1_gpu) + (value__Sy_direction_gpu * FACENORMAL1_2_gpu)) > 0)) ? ((((value__Sx_direction_gpu * FACENORMAL1_1_gpu) + (value__Sy_direction_gpu * FACENORMAL1_2_gpu)) * value_CELL1_I_directionband_gpu)) : ((((value__Sx_direction_gpu * FACENORMAL1_1_gpu) + (value__Sy_direction_gpu * FACENORMAL1_2_gpu)) * value_CELL2_I_directionband_gpu))))

            # boundary conditions handled on cpu side
            flux_tmp_gpu = (fbid_gpu == 1 || fbid_gpu == 2 || fbid_gpu == 3 || fbid_gpu == 4) ? 0.0 : flux_tmp_gpu

            flux_gpu = (flux_gpu + (flux_tmp_gpu * area_over_volume))
        end

        # Row index is current_dof
        global_vector_gpu[current_dof] = source_gpu + flux_gpu


        # go to the next assigned dof
        current_dof = current_dof + max_threads

    end # dof loop


    return nothing
end # GPU kernel

function gpu_bdry_vals_kernel(intensity, vg, sx, sy, center_freq, polarizations, delta_freq, g20xi, g20wi, max_band,
max_dir, normal, max_bi, max_fi, mesh_bdryface, mesh_face2element, mesh_bids, geometric_factors_volume,
geometric_factors_area, dofs_per_node, faceCenters, dim_faceCenters, facex, t, boundary_flux, boundary_dof_index,
global_vector)
    @inbounds begin
    thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if thread_id > max_bi * max_fi * max_band * max_dir
        return
    end
    
    # All index vars
    bi = (thread_id - 1) ÷ (max_fi * max_band * max_dir) + 1
    fi = ((thread_id - 1) ÷ (max_band * max_dir)) % max_fi + 1
    band = ((thread_id - 1) ÷ (max_dir)) % max_band + 1
    dir = ((thread_id - 1) ÷ (1)) % max_dir + 1
    
    fid = mesh_bdryface[fi,bi]
    eid = mesh_face2element[1,fid]
    fbid = mesh_bids[bi]
    volume = geometric_factors_volume[eid]
    area = geometric_factors_area[fid]
    area_over_volume = (area / volume)
    index_offset = (band - 1) * (max_dir) + (dir - 1) * (1)
    row_index = index_offset + 1 + dofs_per_node * (eid - 1)
    for i in 1:dim_faceCenters
        facex[thread_id, i] = faceCenters[i, fid]
    end
    x = facex[thread_id, 1]
    y = dim_faceCenters >= 2 ? facex[thread_id, 2] : 0.0
    z = dim_faceCenters >= 2 ? facex[thread_id, 3] : 0.0
    if fbid == 1
        temp = 300
    elseif fbid == 2
        temp = 300
    elseif fbid == 3
        temp = 300
    elseif fbid == 4
        temp = 300 + 50*exp(-(x-262e-6)*(x-262e-6)/(5e-9))
    end
    
    # Main function body
    ndir::Int = max_dir
    sdotn::Float64 = sx[dir] * normal[1, fid] + sy[dir] * normal[2, fid]
    if sdotn > 0
        interior_intensity::Float64 = intensity[dir + (band - 1) * ndir, eid]
        result = -(vg[band]) * interior_intensity * sdotn
    else
        center_f::Float64 = center_freq[band]
        polarization::Int = polarizations[band]
        delta_f::Float64 = delta_freq[band]
        temp::Float64 = Float64(temp)
        hobol = 7.63822401661014e-12
        vs_TAS = 5230.0
        c_TAS = -2.26e-7
        c_LAS = -2.0e-7
        vs_LAS = 9010.0
        freq = center_f
        dw = delta_f
        vs::Float64 = 0.0
        c::Float64 = 0.0
        if polarization == 0
            vs = vs_TAS
            c = c_TAS
            extra_factor = 2
        else
            vs = vs_LAS
            c = c_LAS
            extra_factor = 1
        end
        const_part = ((1.062861036647414e-37dw) / 2) / (c * c)
        iso_intensity = 0.0
        for gi = 1:20
            fi2 = freq + (dw / 2) * g20xi[gi]
            K2 = (-vs + sqrt(vs * vs + 4 * fi2 * c)) ^ 2
            iso_intensity += ((fi2 * K2) / (exp((hobol * fi2) / temp) - 1)) * g20wi[gi] * extra_factor
        end
        iso_intensity *= const_part
        result = -(vg[band]) * iso_intensity * sdotn
    end
    
    
    # Output
    boundary_flux[thread_id] = result * area_over_volume
    boundary_dof_index[thread_id] = row_index
    CUDA.@atomic global_vector[row_index] = global_vector[row_index] + boundary_flux[thread_id]

    # DEBUG
    # t = Float64(t)
    # if t == 0
    #     @cuprintln("bi $bi fi $fi band $band dir $dir : $result")
    # end
    
    end # inbounds
    return
end
    
function gpu_update_sol_kernel(solution, global_vector, dt, fv_dofs_partition, values_per_dof, components, variable1_values)
    @inbounds begin
        block_id = blockIdx().x
        thread_id = threadIdx().x + (block_id - 1) * blockDim().x
        if thread_id < fv_dofs_partition
            solution[thread_id] = solution[thread_id] + (dt * global_vector[thread_id])
            if thread_id < values_per_dof * components
                dofi = (thread_id - 1) ÷ components
                compi = thread_id - (dofi - 1) * components
                variable1_values[compi, dofi] = solution[thread_id]
            end
        end
    end
    return nothing
end

# begin solve function for I

function generated_solve_function_for_I(var::Vector{Variable{FT}}, mesh::Grid, refel::Refel, geometric_factors::GeometricFactors, fv_info::FVInfo, config::FinchConfig, coefficients::Vector{Coefficient}, variables::Vector{Variable{FT}}, test_functions::Vector{Coefficient}, ordered_indexers::Vector{Indexer}, prob::FinchProblem, time_stepper::Stepper, buffers::ParallelBuffers, timer_output::TimerOutput, nl_var=nothing) where {FT<:AbstractFloat}

    # User specified data types for int and float
    # int type is Int64
    # float type is Float64

    # pre/post step functions if defined
    pre_step_function = prob.pre_step_function
    post_step_function = prob.post_step_function

    num_elements = mesh.nel_owned
    num_band_indices = 13
    num_direction_indices = 4
    tmp_index_ranges = [num_elements, num_band_indices, num_direction_indices,]


    # Prepare some useful numbers
    # dofs_per_node = 880;
    # dofs_per_loop = 1;
    # dof_offsets = [0];

    varcount = length(var)
    dofs_per_node = var[1].total_components
    dofs_per_loop = length(var[1].symvar)
    dof_offsets = zeros(Int, varcount)
    for i = 2:varcount
        dof_offsets[i] = dofs_per_node
        dofs_per_node += var[i].total_components
        dofs_per_loop += length(var[i].symvar)
    end


    nnodes_partition = size(mesh.allnodes, 2)
    nnodes_global = nnodes_partition
    num_elements = mesh.nel_owned
    num_elements_global = mesh.nel_global
    num_elements_ghost = mesh.nel_ghost
    num_faces = mesh.nface_owned + mesh.nface_ghost

    dofs_global = dofs_per_node * nnodes_global
    fv_dofs_global = dofs_per_node * num_elements_global
    dofs_partition = dofs_per_node * nnodes_partition
    fv_dofs_partition = dofs_per_node * (num_elements + num_elements_ghost)
    num_partitions = config.num_partitions
    proc_rank = config.proc_rank

    nodes_per_element = refel.Np
    qnodes_per_element = refel.Nqp
    faces_per_element = refel.Nfaces
    nodes_per_face = refel.Nfp[1]
    dofs_per_element = dofs_per_node * nodes_per_element
    local_system_size = dofs_per_loop * nodes_per_element

    num_bdry_faces = 0
    nbids = length(mesh.bids)
    for bi = 1:nbids
        num_bdry_faces += length(mesh.bdryface[bi])
    end



    # FVM specific pieces
    dofs_global = fv_dofs_global
    # boundary values for flux on each boundary face
    boundary_flux = zeros(Float64, num_bdry_faces * dofs_per_node)
    boundary_dof_index = zeros(Int64, num_bdry_faces * dofs_per_node)

    @timeit timer_output "allocate" begin
        #= Allocate global vectors. =#
        global_vector::Vector{Float64} = zeros(Float64, fv_dofs_partition)
        global_solution::Vector{Float64} = zeros(Float64, fv_dofs_global)
        solution::Vector{Float64} = zeros(Float64, fv_dofs_partition)
        #= Allocate elemental source and flux. =#
        source::Vector{Float64} = zeros(Float64, dofs_per_loop)
        flux::Vector{Float64} = zeros(Float64, dofs_per_loop)
        flux_tmp::Vector{Float64} = zeros(Float64, dofs_per_loop)
        #= Boundary done flag for each face. =#
        bdry_done::Vector{Int64} = zeros(Int64, num_faces)
        #= Flux done flag for each face so that it is not done twice. =#
        face_flux_done::Vector{Bool} = zeros(Bool, num_faces)
        #= index values to be passed to BCs if needed =#
        index_values::Vector{Int64} = zeros(Int64, 2)

    end # timer:allocate

    @timeit timer_output "GPU alloc" begin
        # Allocate and transfer things to the GPU
        mesh_elemental_order_gpu = CuArray(mesh.elemental_order)
        variables_3_values_gpu = CuArray(variables[3].values)
        variables_2_values_gpu = CuArray(variables[2].values)
        variables_1_values_gpu = CuArray(variables[1].values)
        geometric_factors_volume_gpu = CuArray(geometric_factors.volume)
        mesh_element2face_gpu = CuArray(mesh.element2face)
        mesh_facebid_gpu = CuArray(mesh.facebid)
        mesh_face2element_gpu = CuArray(mesh.face2element)
        mesh_facenormals_gpu = CuArray(mesh.facenormals)
        fv_info_faceCenters_gpu = CuArray(fv_info.faceCenters)
        coefficients_3_value_gpu = CuArray(Array{Float64}(coefficients[3].value))
        coefficients_1_value_gpu = CuArray(Array{Float64}(coefficients[1].value))
        coefficients_2_value_gpu = CuArray(Array{Float64}(coefficients[2].value))
        geometric_factors_area_gpu = CuArray(geometric_factors.area)
        global_vector_gpu = CuArray(global_vector)

        index_ranges = CuArray(tmp_index_ranges)

        # ====================== bdry alloc ===========================
        nfaces = length(mesh.bdryface[1])

        total_iters = nbids * nfaces * num_direction_indices * num_band_indices
        bdryface_gpu = CUDA.zeros(Int, nfaces, nbids)
        for bi = 1:nbids
            bdryface_gpu[:, bi] = CuArray(mesh.bdryface[bi])
        end
        index_values_gpu = CUDA.zeros(Int, 2)
        mesh_bids_gpu = CuArray(mesh.bids)
        dim_faceCenters = size(fv_info.faceCenters, 1)
        facex_gpu = CUDA.zeros(total_iters, dim_faceCenters)
        dim_cellCenters = size(fv_info.cellCenters, 1)
        center_freq_gpu = CuArray(Array{Float64}(coefficients[4].value))
        delta_freq_gpu = CuArray(Array{Float64}(coefficients[5].value))
        polarizations_gpu = CuArray(Array{Float64}(coefficients[6].value))
        g20xi = [0.076526521133497, 0.227785851141645, 0.373706088715419, 0.510867001950827, 0.636053680726515,
            0.74633190646015, 0.839116971822218, 0.912234428251325, 0.963971927277913, 0.993128599185094,
            -0.076526521133497, -0.227785851141645, -0.373706088715419, -0.510867001950827, -0.636053680726515,
            -0.74633190646015, -0.839116971822218, -0.912234428251325, -0.963971927277913, -0.993128599185094]
        g20xi_gpu = CuArray(g20xi)

        g20wi = [0.152753387130725, 0.149172986472603, 0.142096109318382, 0.131688638449176, 0.118194531961518,
            0.10193011981724, 0.083276741576704, 0.062672048334109, 0.040601429800386, 0.017614007139152,
            0.152753387130725, 0.149172986472603, 0.142096109318382, 0.131688638449176, 0.118194531961518,
            0.10193011981724, 0.083276741576704, 0.062672048334109, 0.040601429800386, 0.017614007139152]
        g20wi_gpu = CuArray(g20wi)

        boundary_flux_gpu = CUDA.zeros(Float64, num_bdry_faces * dofs_per_node)
        boundary_dof_index_gpu = CUDA.zeros(Int64, num_bdry_faces * dofs_per_node)

    end

    #= No parent-child mesh needed =#
    if (num_partitions > 1)
        exchange_ghosts_fv(var, mesh, dofs_per_node, 0, config)
    end

    solution = get_var_vals(var, solution,)
    t = 0.0
    dt = time_stepper.dt

    # === Extra Addition =======
    nthreads = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    solution_gpu = CuArray(solution)
    Mem.pin(variables[3].values)
    Mem.pin(variables[2].values)
    Mem.pin(variables[1].values)
    Mem.pin(solution)
    values_per_dof = size(var[1].values, 2)
    totalcomponents = var[1].total_components

    global_vector_file = open("bte2d-gv-gpu-newkernel.txt", "w")

    #= ############################################### =#
    #= Time stepping loop =#
    @timeit timer_output "time_steps" begin
        last_minor_progress = 0
        last_major_progress = 0
        if (proc_rank == 0)
            print("Time step progress(%) 0")
        end

        for ti = 1:time_stepper.Nsteps
            bdry_done .= 0
            face_flux_done .= false
            next_nonzero_index = 1
            if (num_partitions > 1)
                exchange_ghosts_fv(var, mesh, dofs_per_node, ti, config)
            end

            #= No pre-step function specified =#
            @timeit timer_output "step_assembly" begin


                # Send needed values back to gpu
                copyto!(variables_3_values_gpu, variables[3].values)
                copyto!(variables_2_values_gpu, variables[2].values)
                # copyto!(variables_1_values_gpu, variables[1].values);
                CUDA.synchronize()


                # This is done on gpu
                CUDA.@sync @cuda threads = 256 blocks = min(4096, ceil(Int, dofs_global / 256)) gpu_assembly_kernel(mesh_elemental_order_gpu, variables_3_values_gpu, variables_2_values_gpu, variables_1_values_gpu, geometric_factors_volume_gpu,
                    mesh_element2face_gpu, mesh_facebid_gpu, mesh_face2element_gpu, mesh_facenormals_gpu, fv_info_faceCenters_gpu,
                    coefficients_3_value_gpu, coefficients_1_value_gpu, coefficients_2_value_gpu, geometric_factors_area_gpu, global_vector_gpu,
                    dofs_global, faces_per_element, index_ranges)


                # Asynchronously compute boundary values on cpu
                # @timeit timer_output "bdry_vals" begin
                #     CUDA.@sync @cuda threads = nthreads blocks = Int(cld(total_iters, nthreads)) gpu_bdry_vals_kernel(nbids, nfaces, num_direction_indices, num_band_indices,
                #         bdryface_gpu, mesh_face2element_gpu, mesh_bids_gpu, geometric_factors_volume_gpu,
                #         geometric_factors_area_gpu, index_values_gpu, dofs_per_node,
                #         fv_info_faceCenters_gpu, dim_faceCenters, facex_gpu, dim_cellCenters,
                #         t, variables_1_values_gpu, coefficients_3_value_gpu, coefficients_1_value_gpu, coefficients_2_value_gpu, mesh_facenormals_gpu,
                #         center_freq_gpu, polarizations_gpu, delta_freq_gpu, g20xi_gpu, g20wi_gpu, boundary_flux_gpu, boundary_dof_index_gpu, global_vector_gpu)
                # end # timer bdry_vals
                @timeit timer_output "bdry_vals" begin
                    CUDA.@sync @cuda threads = nthreads blocks = Int(cld(total_iters, nthreads)) gpu_bdry_vals_kernel(
                    variables_1_values_gpu, coefficients_3_value_gpu, coefficients_1_value_gpu, coefficients_2_value_gpu,
                    center_freq_gpu, polarizations_gpu, delta_freq_gpu, g20xi_gpu, g20wi_gpu, num_band_indices,
                    num_direction_indices, mesh_facenormals_gpu, nbids, nfaces, bdryface_gpu, mesh_face2element_gpu,
                    mesh_bids_gpu, geometric_factors_volume_gpu, geometric_factors_area_gpu, dofs_per_node,
                    fv_info_faceCenters_gpu, dim_faceCenters, facex_gpu, t, boundary_flux_gpu, boundary_dof_index_gpu,
                    global_vector_gpu)
                end # timer bdry_vals

            end # timer:step_assembly

            println(global_vector_file, "ti=$(ti), t=$(t)")
            println(global_vector_file, global_vector_gpu)
            println(global_vector_file, "=" ^ 30)
            println(global_vector_file)

            @timeit timer_output "update_sol" begin
                CUDA.@sync @cuda threads = nthreads blocks = Int(cld(fv_dofs_partition, nthreads)) gpu_update_sol_kernel(solution_gpu, global_vector_gpu, dt, fv_dofs_partition, values_per_dof, totalcomponents, variables_1_values_gpu)
                # copyto!(solution, solution_gpu)
            end # timer:update_sol


            # copy_bdry_vals_to_vector(var, solution, mesh, dofs_per_node, prob);
            # place_vector_in_vars(var, solution);

            copyto!(var[1].values, variables_1_values_gpu)
            post_step_function()
            t = (t + dt)
            if ((100.0 * (ti / time_stepper.Nsteps)) >= (last_major_progress + 10))
                last_major_progress = (last_major_progress + 10)
                last_minor_progress = (last_minor_progress + 2)
                if (proc_rank == 0)
                    print(string(last_major_progress))
                end


            else
                if ((100.0 * (ti / time_stepper.Nsteps)) >= (last_minor_progress + 2))
                    last_minor_progress = (last_minor_progress + 2)
                    if (proc_rank == 0)
                        print(".")
                    end

                end

            end

        end


    end # timer:time_steps

    close(global_vector_file)

    return nothing

end # function



# end solve function for I

# No code set for Io

# No code set for beta

# No code set for temperature

# No code set for temperatureLast

# No code set for G_last

# No code set for G_next
