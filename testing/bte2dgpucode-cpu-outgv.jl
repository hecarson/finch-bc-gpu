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
    max_threads = blockDim().x * gridDim().x;
    block_id = blockIdx().x;
    thread_id = threadIdx().x + (block_id - 1) * blockDim().x;
    
    # Index counts
    num_direction = index_ranges[3];
    num_band = index_ranges[2];
    num_elements = index_ranges[1];
    
    
    # Loop over all dofs assigned to this thread
    # strided by max_threads
    current_dof = thread_id;
    while current_dof <= dofs_global
        
        # extract index values
        INDEX_VAL_direction = Int(mod(current_dof-1, num_direction) + 1);
        INDEX_VAL_band = Int(floor(mod(current_dof-1, num_band * num_direction) / (num_direction)) + 1);
        eid = Int(floor(mod(current_dof-1, num_elements * num_direction * num_band) / (num_direction * num_band)) + 1);
        
        
        #= Begin assembly code =#
        source_gpu = 0.0
        flux_gpu = 0.0
        #= Evaluate volume coefficients. =#
        value__beta_band_gpu = variables_3_values_gpu[INDEX_VAL_band, eid]
        value__Io_band_gpu = variables_2_values_gpu[INDEX_VAL_band, eid]
        value__I_directionband_gpu = variables_1_values_gpu[(INDEX_VAL_direction + (4*(INDEX_VAL_band-1))), eid]
        volume = geometric_factors_volume_gpu[eid]
        #= Compute source terms (volume integral) =#
        source_gpu = ((value__beta_band_gpu * value__Io_band_gpu) + ((-value__beta_band_gpu) * value__I_directionband_gpu));
        
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
            value_CELL1_I_directionband_gpu = variables_1_values_gpu[(INDEX_VAL_direction + (4*(INDEX_VAL_band-1))), eid]
            value_CELL2_I_directionband_gpu = variables_1_values_gpu[(INDEX_VAL_direction + (4*(INDEX_VAL_band-1))), neighbor]
            area = geometric_factors_area_gpu[fid]
            area_over_volume = (area / volume)
            #= Compute flux terms (surface integral) =#
            flux_tmp_gpu = ((-value__vg_band_gpu) * (((((value__Sx_direction_gpu * FACENORMAL1_1_gpu) + (value__Sy_direction_gpu * FACENORMAL1_2_gpu)) > 0)) ? ((((value__Sx_direction_gpu * FACENORMAL1_1_gpu) + (value__Sy_direction_gpu * FACENORMAL1_2_gpu)) * value_CELL1_I_directionband_gpu)) : ((((value__Sx_direction_gpu * FACENORMAL1_1_gpu) + (value__Sy_direction_gpu * FACENORMAL1_2_gpu)) * value_CELL2_I_directionband_gpu))));
            
            # boundary conditions handled on cpu side
            flux_tmp_gpu = (fbid_gpu==1 || fbid_gpu==2 || fbid_gpu==3 || fbid_gpu==4) ? 0.0 : flux_tmp_gpu
            
            flux_gpu = (flux_gpu + (flux_tmp_gpu * area_over_volume))
        end

        # Row index is current_dof
        global_vector_gpu[current_dof] = source_gpu + flux_gpu
        
        
        # go to the next assigned dof
        current_dof = current_dof + max_threads;
        
    end # dof loop

    
    return nothing;
end # GPU kernel



# begin solve function for I

function generated_solve_function_for_I(var::Vector{Variable{FT}}, mesh::Grid, refel::Refel, geometric_factors::GeometricFactors, fv_info::FVInfo, config::FinchConfig, coefficients::Vector{Coefficient}, variables::Vector{Variable{FT}}, test_functions::Vector{Coefficient}, ordered_indexers::Vector{Indexer}, prob::FinchProblem, time_stepper::Stepper, buffers::ParallelBuffers, timer_output::TimerOutput, nl_var=nothing) where FT<:AbstractFloat
    
    # User specified data types for int and float
    # int type is Int64
    # float type is Float64
    
    # pre/post step functions if defined
    pre_step_function = prob.pre_step_function;
    post_step_function = prob.post_step_function;
    
    num_elements = mesh.nel_owned;
    num_band_indices = 13;
    num_direction_indices = 4;
    tmp_index_ranges = [num_elements, num_band_indices, num_direction_indices, ];
    
    
    # Prepare some useful numbers
    # dofs_per_node = 52;
    # dofs_per_loop = 1;
    # dof_offsets = [0];
    
    varcount = length(var);
    dofs_per_node = var[1].total_components;
    dofs_per_loop = length(var[1].symvar);
    dof_offsets = zeros(Int, varcount);
    for i=2:varcount
        dof_offsets[i] = dofs_per_node;
        dofs_per_node += var[i].total_components;
        dofs_per_loop += length(var[i].symvar);
    end

    
    nnodes_partition = size(mesh.allnodes,2);
    nnodes_global = nnodes_partition;
    num_elements = mesh.nel_owned;
    num_elements_global = mesh.nel_global;
    num_elements_ghost = mesh.nel_ghost;
    num_faces = mesh.nface_owned + mesh.nface_ghost;
    
    dofs_global = dofs_per_node * nnodes_global;
    fv_dofs_global = dofs_per_node * num_elements_global;
    dofs_partition = dofs_per_node * nnodes_partition;
    fv_dofs_partition = dofs_per_node * (num_elements + num_elements_ghost);
    num_partitions = config.num_partitions;
    proc_rank = config.proc_rank;
    
    nodes_per_element = refel.Np;
    qnodes_per_element = refel.Nqp;
    faces_per_element = refel.Nfaces;
    nodes_per_face = refel.Nfp[1];
    dofs_per_element = dofs_per_node * nodes_per_element;
    local_system_size = dofs_per_loop * nodes_per_element;
    
    num_bdry_faces = 0;
    nbids = length(mesh.bids);
    for bi=1:nbids
        num_bdry_faces += length(mesh.bdryface[bi]);
    end

    
    
    # FVM specific pieces
    dofs_global = fv_dofs_global;
    # boundary values for flux on each boundary face
    boundary_flux = zeros(Float64, num_bdry_faces * dofs_per_node);
    boundary_dof_index = zeros(Int64, num_bdry_faces * dofs_per_node);
    
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
        mesh_elemental_order_gpu = CuArray(mesh.elemental_order);
        variables_3_values_gpu = CuArray(variables[3].values);
        variables_2_values_gpu = CuArray(variables[2].values);
        variables_1_values_gpu = CuArray(variables[1].values);
        geometric_factors_volume_gpu = CuArray(geometric_factors.volume);
        mesh_element2face_gpu = CuArray(mesh.element2face);
        mesh_facebid_gpu = CuArray(mesh.facebid);
        mesh_face2element_gpu = CuArray(mesh.face2element);
        mesh_facenormals_gpu = CuArray(mesh.facenormals);
        fv_info_faceCenters_gpu = CuArray(fv_info.faceCenters);
        coefficients_3_value_gpu = CuArray(Array{Float64}(coefficients[3].value));
        coefficients_1_value_gpu = CuArray(Array{Float64}(coefficients[1].value));
        coefficients_2_value_gpu = CuArray(Array{Float64}(coefficients[2].value));
        geometric_factors_area_gpu = CuArray(geometric_factors.area);
        global_vector_gpu = CuArray(global_vector);
        
        index_ranges = CuArray(tmp_index_ranges);
    end

    #= No parent-child mesh needed =#
    if (num_partitions > 1)
        exchange_ghosts_fv(var, mesh, dofs_per_node, 0, config);
    end

    solution = get_var_vals(var, solution, );
    t = 0.0
    dt = time_stepper.dt

    global_vector_file = open("bte2d-gv-cpu.txt", "w")

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
                exchange_ghosts_fv(var, mesh, dofs_per_node, ti, config);
            end

            #= No pre-step function specified =#
            @timeit timer_output "step_assembly" begin
                
                
                # Send needed values back to gpu
                copyto!(variables_3_values_gpu, variables[3].values);
                copyto!(variables_2_values_gpu, variables[2].values);
                copyto!(variables_1_values_gpu, variables[1].values);
                CUDA.synchronize();
                
                
                # This is done on gpu
                @cuda threads=256 blocks=min(4096,ceil(Int, dofs_global/256)) gpu_assembly_kernel(mesh_elemental_order_gpu, variables_3_values_gpu, variables_2_values_gpu, variables_1_values_gpu, geometric_factors_volume_gpu, 
                            mesh_element2face_gpu, mesh_facebid_gpu, mesh_face2element_gpu, mesh_facenormals_gpu, fv_info_faceCenters_gpu, 
                            coefficients_3_value_gpu, coefficients_1_value_gpu, coefficients_2_value_gpu, geometric_factors_area_gpu, global_vector_gpu, 
                            dofs_global, faces_per_element, index_ranges)
                
                
                # Asynchronously compute boundary values on cpu
                @timeit timer_output "bdry_vals" begin
                    next_bdry_index = 1;
                    for bi=1:nbids
                        nfaces = length(mesh.bdryface[bi]);
                        for fi=1:nfaces
                            fid = mesh.bdryface[bi][fi];
                            eid = mesh.face2element[1,fid];
                            fbid = mesh.bids[bi];
                            volume = geometric_factors.volume[eid]
                            area = geometric_factors.area[fid]
                            area_over_volume = (area / volume)
                            
                            for INDEX_VAL_direction = 1:num_direction_indices
                                for INDEX_VAL_band = 1:num_band_indices
                                    
                                    index_values[1] = INDEX_VAL_direction
                                    index_values[2] = INDEX_VAL_band
                                    
                                    index_offset = INDEX_VAL_direction + num_direction_indices * (INDEX_VAL_band - 1) - 1
                                    
                                    row_index = index_offset + 1 + dofs_per_node * (eid - 1);
                                    
                                    apply_boundary_conditions_face_rhs(var, eid, fid, fbid, mesh, refel, geometric_factors, fv_info, prob, 
                                                                        t, dt, flux_tmp, bdry_done, index_offset, index_values)
                                    #
                                    # store it
                                    boundary_flux[next_bdry_index] = flux_tmp[1] * area_over_volume;
                                    boundary_dof_index[next_bdry_index] = row_index;
                                    next_bdry_index += 1;

                                    # DEBUG
                                    if t == 0
                                        println("bi $bi fi $fi band $INDEX_VAL_band dir $INDEX_VAL_direction : $(flux_tmp[1])")
                                    end
                                end

                            end

                            
                        end

                    end

                end # timer bdry_vals

                
                # Then get global_vector from gpu
                CUDA.synchronize()
                copyto!(global_vector, global_vector_gpu)
                CUDA.synchronize()
                
                # And add BCs to global vector
                for update_i = 1:(num_bdry_faces * dofs_per_node)
                    row_index = boundary_dof_index[update_i]
                    global_vector[row_index] = global_vector[row_index] + boundary_flux[update_i];
                end

                
                
            end # timer:step_assembly


            println(global_vector_file, "ti=$(ti), t=$(t)")
            println(global_vector_file, global_vector)
            println(global_vector_file, "=" ^ 30)
            println(global_vector_file)
            
            @timeit timer_output "update_sol" begin
                for update_i = 1:fv_dofs_partition
                    solution[update_i] = (solution[update_i] + (dt * global_vector[update_i]))
                end

                
            end # timer:update_sol

            
            copy_bdry_vals_to_vector(var, solution, mesh, dofs_per_node, prob);
            place_vector_in_vars(var, solution);
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


        close(global_vector_file)
        
    end # timer:time_steps

    
    
    return nothing;
    
end # function



# end solve function for I

# No code set for Io

# No code set for beta

# No code set for temperature

# No code set for temperatureLast

# No code set for G_last

# No code set for G_next
