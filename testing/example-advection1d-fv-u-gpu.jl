#=
1D advection using FV
=#

### If the Finch package has already been added, use this line #########
using Finch # Note: to add the package, first do: ]add "https://github.com/paralab/Finch.git"

### If not, use these four lines (working from the examples directory) ###
# if !@isdefined(Finch)
#     include("../Finch.jl");
#     using .Finch
# end
##########################################################################

initFinch("advection1d");

useLog("advection1dlog", level=3)

# Configuration setup
domain(1)
solverType(FV)
timeStepper(EULER_EXPLICIT)

Finch.finch_state.config.use_gpu = true

# Mesh
# n = 100 # number of elements
n = 10
mesh(LINEMESH, elsperdim=n, bids=2)

# Variables and BCs
u = variable("u", location=CELL)
# v = variable("v", location=CELL)
# w = variable("w", location=CELL)

@callbackFunction(
    function bc1_f(t)
        return t<0.2 ? sin(pi*t/0.2)^2 : 0
    end
)

# left boundary
boundary(u, 1, FLUX, "bc1_f(t)")
# boundary(v, 1, DIRICHLET, "bc_f(t)")
# boundary(w, 1, DIRICHLET, 1)

# right boundary
boundary(u, 2, NO_BC)
# boundary(u, 2, FLUX, "bc2_f()")
# boundary(v, 2, NO_BC)
# boundary(w, 2, DIRICHLET, 0)

# Time interval and initial condition
T = 0.5;
timeInterval(T)
initial(u, 0)
# initial(v, 0)
# initial(w, 0)

# Advection velocity is the same for all
coefficient("a", 1) 

# The conservation type equation
# The "upwind" function applies upwinding to the term (a.n)*f with flow velocity a.
# The optional third parameter is for tuning. Default upwind = 0, central = 1. Choose something between these.
# conservationForm([u, v, w], ["surface(upwind(a,u))", "surface(upwind(a,v))", "surface(upwind(a,w))"]) 
conservationForm(u, "surface(upwind(a,u))") 

# exportCode("fvad1dgpucode") # uncomment to export generated code to a file
# importCode("finch-gpu-test/fvad1dgpucode-fix")
importCode("finch-gpu-test/fvad1dgpucode-newkernel")

# solve([u,v,w])
solve(u)
out_file = open("fvad1d-u-gpu-sol.txt", "w")
println(out_file, "u: $(u.values)")
# println(out_file, "v: $(v.values)")
# println(out_file, "w: $(w.values)")
close(out_file)

finalizeFinch()

##### Uncomment below to compare to exact solution

# # The exact solution for u with constant velocity a=1
# a = 1;
# n = size(Finch.finch_state.fv_info.cellCenters,2);
# exact = zeros(n);
# x = Finch.finch_state.fv_info.cellCenters[:]
# for i=1:n
#     xt = x[i] - a*T;
#     if xt < 0
#         exact[i] = xt > -0.2 ? sin(pi*-xt/0.2)^2 : 0
#     end
# end

# using Plots
# pyplot();
# display(plot([x x x x], [exact u.values[:] v.values[:] w.values[:]], markershape=:circle, label=["exact u" "u" "v" "w"]))