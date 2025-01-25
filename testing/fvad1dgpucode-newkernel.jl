#=
Generated functions for advection1d
=#

#=
Auxilliary code that will be included in Finch
=#

function gpu_assembly_kernel(mesh_elemental_order_gpu, geometric_factors_volume_gpu, mesh_element2face_gpu, mesh_facebid_gpu, mesh_face2element_gpu, 
                mesh_facenormals_gpu, variables_1_values_gpu, geometric_factors_area_gpu, global_vector_gpu, dofs_global, 
                faces_per_element, index_ranges)
    
    # The dofs handled by this thread are determined by the global thread ID
    # max_threads = blockDim().x * gridDim().x * blockDim().y * gridDim().y * blockDim().z * gridDim().z;
    # block_id = blockIdx().x + (blockIdx().y - 1) * gridDim().x + (blockIdx().z - 1) * gridDim().x * gridDim().y;
    # thread_id = threadIdx().x + (block_id - 1) * blockDim().x * blockDim().y * blockDim().z;
    
    # simplified version with 1D grid and blocks
    max_threads = blockDim().x * gridDim().x;
    block_id = blockIdx().x;
    thread_id = threadIdx().x + (block_id - 1) * blockDim().x;
    
    # Index counts
    num_elements = index_ranges[1];
    
    
    # Loop over all dofs assigned to this thread
    # strided by max_threads
    current_dof = thread_id;
    while current_dof <= dofs_global
        
        # extract index values
        eid = current_dof;
        
        
        #= Begin assembly code =#
        source_gpu = 0.0
        flux_gpu = 0.0
        #= Evaluate volume coefficients. =#
        volume = geometric_factors_volume_gpu[eid]
        #= Compute source terms (volume integral) =#
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
            value_CELL1_u_1_gpu = variables_1_values_gpu[1, eid]
            value_CELL2_u_1_gpu = variables_1_values_gpu[1, neighbor]
            area = geometric_factors_area_gpu[fid]
            area_over_volume = (area / volume)
            #= Compute flux terms (surface integral) =#
            flux_tmp_gpu = (-((((1.0 * FACENORMAL1_1_gpu) > 0)) ? ((1.0 * value_CELL1_u_1_gpu * FACENORMAL1_1_gpu)) : ((1.0 * value_CELL2_u_1_gpu * FACENORMAL1_1_gpu))))
            # boundary conditions handled on cpu side
            # flux_tmp_gpu = (fbid_gpu==1 || fbid_gpu==2) ? 0.0 : flux_tmp_gpu
            # FIX
            flux_tmp_gpu = (fbid_gpu==1) ? 0.0 : flux_tmp_gpu
            
            flux_gpu = (flux_gpu + (flux_tmp_gpu * area_over_volume))
        end

        # Row index is current_dof
        global_vector_gpu[current_dof] = source_gpu + flux_gpu
        
        
        # go to the next assigned dof
        current_dof = current_dof + max_threads;
        
    end # dof loop

    
    return nothing;
end # GPU kernel

include("ad1d_bdry_kernels.jl")

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

# begin solve function for u

function generated_solve_function_for_u(var::Vector{Variable{FT}}, mesh::Grid, refel::Refel, geometric_factors::GeometricFactors, fv_info::FVInfo, config::FinchConfig, coefficients::Vector{Coefficient}, variables::Vector{Variable{FT}}, test_functions::Vector{Coefficient}, ordered_indexers::Vector{Indexer}, prob::FinchProblem, time_stepper::Stepper, buffers::ParallelBuffers, timer_output::TimerOutput, nl_var=nothing) where FT<:AbstractFloat
    
    # User specified data types for int and float
    # int type is Int64
    # float type is Float64
    
    # pre/post step functions if defined
    pre_step_function = prob.pre_step_function;
    post_step_function = prob.post_step_function;
    
    # FIX
    num_elements = mesh.nel_owned
    tmp_index_ranges = [num_elements]
    
    # Prepare some useful numbers
    # dofs_per_node = 1;
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
        #= No indexed variables =#
        index_values::Vector{Int64} = zeros(Int64, 0)
        
    end # timer:allocate

    @timeit timer_output "GPU alloc" begin
        # Allocate and transfer things to the GPU
        mesh_elemental_order_gpu = CuArray(mesh.elemental_order);
        geometric_factors_volume_gpu = CuArray(geometric_factors.volume);
        mesh_element2face_gpu = CuArray(mesh.element2face);
        mesh_facebid_gpu = CuArray(mesh.facebid);
        mesh_face2element_gpu = CuArray(mesh.face2element);
        mesh_facenormals_gpu = CuArray(mesh.facenormals);
        variables_1_values_gpu = CuArray(variables[1].values);
        geometric_factors_area_gpu = CuArray(geometric_factors.area);
        global_vector_gpu = CuArray(global_vector);
        
        index_ranges = CuArray(tmp_index_ranges);

        # bdry alloc
        bdryface_bi_1_gpu = CuArray(mesh.bdryface[1])
        mesh_bids_gpu = CuArray(mesh.bids)
        fv_info_faceCenters_gpu = CuArray(fv_info.faceCenters)
        dim_faceCenters = size(fv_info.faceCenters, 1)
        max_nfaces = maximum([length(mesh.bdryface[bi]) for bi=1:nbids])
        max_total_iters = max_nfaces
        facex_gpu = CUDA.zeros(max_total_iters, dim_faceCenters)
        boundary_flux_gpu = CUDA.zeros(Float64, num_bdry_faces * dofs_per_node)
        boundary_dof_index_gpu = CUDA.zeros(Int64, num_bdry_faces * dofs_per_node)
    end

    #= No parent-child mesh needed =#
    if (num_partitions > 1)
        exchange_ghosts_fv(var, mesh, dofs_per_node, 0, config);
    end

    solution = get_var_vals(var, solution, );
    t = 0.0
    dt = time_stepper.dt

    # Extra addition
    nthreads = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    # nthreads = 2
    solution_gpu = CuArray(solution)
    Mem.pin(variables[1].values)
    Mem.pin(solution)
    values_per_dof = size(var[1].values, 2)
    totalcomponents = var[1].total_components

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
                # copyto!(variables_1_values_gpu, variables[1].values); # CHANGED
                CUDA.synchronize();
                
                
                # This is done on gpu
                @cuda threads=256 blocks=min(4096,ceil(Int, dofs_global/256)) gpu_assembly_kernel(mesh_elemental_order_gpu, geometric_factors_volume_gpu, mesh_element2face_gpu, mesh_facebid_gpu, mesh_face2element_gpu, 
                            mesh_facenormals_gpu, variables_1_values_gpu, geometric_factors_area_gpu, global_vector_gpu, dofs_global, 
                            faces_per_element, index_ranges)
                
                
                # Asynchronously compute boundary values on cpu
                @timeit timer_output "bdry_vals" begin
                    nfaces = length(mesh.bdryface[1])
                    total_iters = nfaces
                    CUDA.@sync @cuda threads = nthreads blocks = Int(cld(total_iters, nthreads)) ad_bc_bi_1_gpu(
                    t, nfaces, bdryface_bi_1_gpu, mesh_face2element_gpu, mesh_bids_gpu, geometric_factors_volume_gpu,
                    geometric_factors_area_gpu, dofs_per_node, fv_info_faceCenters_gpu, dim_faceCenters, facex_gpu,
                    boundary_flux_gpu, boundary_dof_index_gpu, global_vector_gpu)

                end # timer bdry_vals

            end # timer:step_assembly

            
            @timeit timer_output "update_sol" begin
                CUDA.@sync @cuda threads = nthreads blocks = Int(cld(fv_dofs_partition, nthreads)) gpu_update_sol_kernel(solution_gpu, global_vector_gpu, dt, fv_dofs_partition, values_per_dof, totalcomponents, variables_1_values_gpu)
                # copyto!(solution, solution_gpu)
            end # timer:update_sol

            
            # copy_bdry_vals_to_vector(var, solution, mesh, dofs_per_node, prob);
            # place_vector_in_vars(var, solution);

            # copyto!(var[1].values, variables_1_values_gpu) # CHANGED

            #= No post-step function specified =#
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

    copyto!(var[1].values, variables_1_values_gpu) # CHANGED
    
    return nothing;
    
end # function



# end solve function for u

