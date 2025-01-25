function ad_bc_bi_1_gpu(t, max_fi, mesh_bdryface, mesh_face2element, mesh_bids, geometric_factors_volume, geometric_factors_area, dofs_per_node, faceCenters, dim_faceCenters, facex, boundary_flux, boundary_dof_index, global_vector)
@inbounds begin
thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
if thread_id > max_fi
    return
end

# All index vars
bi = 1
fi = (thread_id - 1) ÷ (1) + 1

fid = mesh_bdryface[fi]
eid = mesh_face2element[1,fid]
fbid = mesh_bids[bi]
volume = geometric_factors_volume[eid]
area = geometric_factors_area[fid]
area_over_volume = (area / volume)
row_index = 1 + dofs_per_node * (eid - 1)
for i in 1:dim_faceCenters
    facex[thread_id, i] = faceCenters[i, fid]
end
x = facex[thread_id, 1]
y = dim_faceCenters >= 2 ? facex[thread_id, 2] : 0.0
z = dim_faceCenters >= 2 ? facex[thread_id, 3] : 0.0

# Main function body
if t < 0.2
    result = sin((pi * t) / 0.2) ^ 2
else
    result = 0
end


# Output
boundary_flux[thread_id] = result * area_over_volume
boundary_dof_index[thread_id] = row_index
global_vector[row_index] = global_vector[row_index] + boundary_flux[thread_id]

end # inbounds
return
end

