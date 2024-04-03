function isothermal_bdry(intensity, vg::Vector, sx::Vector, sy::Vector, center_freq, polarizations, delta_freq,
g20xi::Vector{Float64}, g20wi::Vector{Float64}, band::Int, dir::Int, normal::Vector{Float64}, temp)
    ndir::Int = max_dir
    sdotn::Float64 = sx[dir]*normal[1] + sy[dir]*normal[2]

    if sdotn > 0 # outward
        interior_intensity::Float64 = intensity[dir + (band-1)*ndir]
        result = -vg[band] * interior_intensity * sdotn
    else # inward gains from equilibrium
        center_f::Float64 = center_freq[band]
        polarization::Int = polarizations[band]
        delta_f::Float64 = delta_freq[band]
        temp::Float64 = Float64(temp)

        # --- equilibrium_intensity ---
        hobol = 7.63822401661014e-12
        vs_TAS= 5230.0
        c_TAS = -2.26e-7
        c_LAS = -2.0e-7;
        vs_LAS= 9010.0;
        freq = center_f
        dw = delta_f

        vs::Float64 = 0.0;
        c::Float64 = 0.0;
        if polarization == 0 # T
            vs = vs_TAS;
            c = c_TAS;
            extra_factor = 2;
        else # L
            vs = vs_LAS;
            c = c_LAS;
            extra_factor = 1;
        end
        # dirac/(32*pi^3) = 1.062861036647414e-37
        const_part = 1.062861036647414e-37 * dw/2 / (c*c); # constants to pull out of integral
        iso_intensity = 0.0;
        for gi=1:20
            fi2 = freq + dw/2 * g20xi[gi]; # frequency at gauss point
            K2 = (-vs + sqrt(vs*vs + 4*fi2*c))^2; # K^2 * (2*c)^2   the (2*c)^2 is put in the const_part
            iso_intensity += (fi2 * K2 / (exp(hobol*fi2/temp) - 1)) * g20wi[gi] * extra_factor;
        end
        iso_intensity *= const_part;
        # ---

        result = -vg[band] * iso_intensity * sdotn
    end

    return result
end