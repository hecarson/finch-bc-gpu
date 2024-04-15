function ad_bc(t)
    if t < 0.2
        result = sin(pi*t/0.2)^2
    else
        result = 0
    end
    return result
end