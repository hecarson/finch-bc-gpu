function isothermal_bdry_bi_1_gpu(intensity, vg, sx, sy, center_freq, polarizations, delta_freq, g20xi, g20wi, max_band, max_dir, normal, max_fi, mesh_bdryface, mesh_face2element, mesh_bids, geometric_factors_volume, geometric_factors_area, dofs_per_node, faceCenters, dim_faceCenters, facex, t, boundary_flux, boundary_dof_index, global_vector)
@inbounds begin
thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
if thread_id > max_fi * max_dir * max_band
    return
end

# All index vars
bi = 1
fi = (thread_id - 1) ÷ (max_dir * max_band) + 1
dir = ((thread_id - 1) ÷ (max_band)) % max_dir + 1
band = ((thread_id - 1) ÷ (1)) % max_band + 1

fid = mesh_bdryface[fi]
eid = mesh_face2element[1,fid]
fbid = mesh_bids[bi]
volume = geometric_factors_volume[eid]
area = geometric_factors_area[fid]
area_over_volume = (area / volume)
index_offset = (dir - 1) * (1) + (band - 1) * (max_dir)
row_index = index_offset + 1 + dofs_per_node * (eid - 1)
for i in 1:dim_faceCenters
    facex[thread_id, i] = faceCenters[i, fid]
end
x = facex[thread_id, 1]
y = dim_faceCenters >= 2 ? facex[thread_id, 2] : 0.0
z = dim_faceCenters >= 2 ? facex[thread_id, 3] : 0.0
temp = 300

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
global_vector[row_index] = global_vector[row_index] + boundary_flux[thread_id]

end # inbounds
return
end


function isothermal_bdry_bi_2_gpu(intensity, vg, sx, sy, center_freq, polarizations, delta_freq, g20xi, g20wi, max_band, max_dir, normal, max_fi, mesh_bdryface, mesh_face2element, mesh_bids, geometric_factors_volume, geometric_factors_area, dofs_per_node, faceCenters, dim_faceCenters, facex, t, boundary_flux, boundary_dof_index, global_vector)
@inbounds begin
thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
if thread_id > max_fi * max_dir * max_band
    return
end

# All index vars
bi = 2
fi = (thread_id - 1) ÷ (max_dir * max_band) + 1
dir = ((thread_id - 1) ÷ (max_band)) % max_dir + 1
band = ((thread_id - 1) ÷ (1)) % max_band + 1

fid = mesh_bdryface[fi]
eid = mesh_face2element[1,fid]
fbid = mesh_bids[bi]
volume = geometric_factors_volume[eid]
area = geometric_factors_area[fid]
area_over_volume = (area / volume)
index_offset = (dir - 1) * (1) + (band - 1) * (max_dir)
row_index = index_offset + 1 + dofs_per_node * (eid - 1)
for i in 1:dim_faceCenters
    facex[thread_id, i] = faceCenters[i, fid]
end
x = facex[thread_id, 1]
y = dim_faceCenters >= 2 ? facex[thread_id, 2] : 0.0
z = dim_faceCenters >= 2 ? facex[thread_id, 3] : 0.0
temp = 300

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
global_vector[row_index] = global_vector[row_index] + boundary_flux[thread_id]

end # inbounds
return
end


function isothermal_bdry_bi_3_gpu(intensity, vg, sx, sy, center_freq, polarizations, delta_freq, g20xi, g20wi, max_band, max_dir, normal, max_fi, mesh_bdryface, mesh_face2element, mesh_bids, geometric_factors_volume, geometric_factors_area, dofs_per_node, faceCenters, dim_faceCenters, facex, t, boundary_flux, boundary_dof_index, global_vector)
@inbounds begin
thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
if thread_id > max_fi * max_dir * max_band
    return
end

# All index vars
bi = 3
fi = (thread_id - 1) ÷ (max_dir * max_band) + 1
dir = ((thread_id - 1) ÷ (max_band)) % max_dir + 1
band = ((thread_id - 1) ÷ (1)) % max_band + 1

fid = mesh_bdryface[fi]
eid = mesh_face2element[1,fid]
fbid = mesh_bids[bi]
volume = geometric_factors_volume[eid]
area = geometric_factors_area[fid]
area_over_volume = (area / volume)
index_offset = (dir - 1) * (1) + (band - 1) * (max_dir)
row_index = index_offset + 1 + dofs_per_node * (eid - 1)
for i in 1:dim_faceCenters
    facex[thread_id, i] = faceCenters[i, fid]
end
x = facex[thread_id, 1]
y = dim_faceCenters >= 2 ? facex[thread_id, 2] : 0.0
z = dim_faceCenters >= 2 ? facex[thread_id, 3] : 0.0
temp = 300

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
global_vector[row_index] = global_vector[row_index] + boundary_flux[thread_id]

end # inbounds
return
end


function isothermal_bdry_bi_4_gpu(intensity, vg, sx, sy, center_freq, polarizations, delta_freq, g20xi, g20wi, max_band, max_dir, normal, max_fi, mesh_bdryface, mesh_face2element, mesh_bids, geometric_factors_volume, geometric_factors_area, dofs_per_node, faceCenters, dim_faceCenters, facex, t, boundary_flux, boundary_dof_index, global_vector)
@inbounds begin
thread_id = threadIdx().x + blockDim().x * (blockIdx().x - 1)
if thread_id > max_fi * max_dir * max_band
    return
end

# All index vars
bi = 4
fi = (thread_id - 1) ÷ (max_dir * max_band) + 1
dir = ((thread_id - 1) ÷ (max_band)) % max_dir + 1
band = ((thread_id - 1) ÷ (1)) % max_band + 1

fid = mesh_bdryface[fi]
eid = mesh_face2element[1,fid]
fbid = mesh_bids[bi]
volume = geometric_factors_volume[eid]
area = geometric_factors_area[fid]
area_over_volume = (area / volume)
index_offset = (dir - 1) * (1) + (band - 1) * (max_dir)
row_index = index_offset + 1 + dofs_per_node * (eid - 1)
for i in 1:dim_faceCenters
    facex[thread_id, i] = faceCenters[i, fid]
end
x = facex[thread_id, 1]
y = dim_faceCenters >= 2 ? facex[thread_id, 2] : 0.0
z = dim_faceCenters >= 2 ? facex[thread_id, 3] : 0.0
temp = 300 + 50*exp(-(x-262e-6)*(x-262e-6)/(5e-9))

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
global_vector[row_index] = global_vector[row_index] + boundary_flux[thread_id]

end # inbounds
return
end


