# Functions must have "return result" at end and only one "return result"

function f(a::Integer, b, c, d)
    result = a + b + c + d

    for i = 1:a
        result += 1
    end

    return result
end