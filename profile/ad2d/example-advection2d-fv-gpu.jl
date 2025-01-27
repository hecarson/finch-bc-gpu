#=
Note from Carson:
Copy of testing/ad2d/example-advection2d-fv-gpu.jl, with minor changes for profiling
=#

#=
2D advection using structured or unstructured mesh
=#

### If the Finch package has already been added, use this line #########
# using Finch # Note: to add the package, first do: ]add "https://github.com/paralab/Finch.git"

### If not, use these four lines (working from the examples directory) ###
# if !@isdefined(Finch)
#     include("../../Finch/src/Finch.jl");
#     using .Finch
# end
include("../../../Finch/src/Finch.jl")
using .Finch
using CUDA
##########################################################################

initFinch("advection2d");

useLog("advection2dlog", level=3)

# Configuration setup
domain(2)
solverType(FV)

# timeStepper(EULER_IMPLICIT)
# FIX
timeStepper(EULER_EXPLICIT)

Finch.finch_state.config.use_gpu = true

use_unstructured=false;
if use_unstructured
    # Using an unstructured mesh of triangles or irregular quads
    # This is a 0.1 x 0.3 rectangle domain
    mesh("src/examples/utriangle.msh")
    # mesh("src/examples/uquad.msh")
    
    addBoundaryID(2, (x,y) -> (x >= 0.1));
    addBoundaryID(3, (x,y) -> (y <= 0));
    addBoundaryID(4, (x,y) -> (y >= 0.3));
    
else
    # a uniform grid of quads on a 0.1 x 0.3 rectangle domain
    # mesh(QUADMESH, elsperdim=[15, 45], bids=4, interval=[0, 0.1, 0, 0.3])
    # mesh(QUADMESH, elsperdim=[10, 10], bids=4, interval=[0, 0.1, 0, 0.1])
    # mesh(QUADMESH, elsperdim=[100, 100], bids=4, interval=[0, 0.1, 0, 0.1])
    mesh(QUADMESH, elsperdim=[300, 300], bids=4, interval=[0, 0.1, 0, 0.1])
end

# Variables and BCs
u = variable("u", location=CELL)

@callbackFunction(
    function bc_f(t, y)
        return (abs(y-0.06) < 0.033 && sin(3*pi*t)>0) ? 1 : 0
    end
)

boundary(u, 1, FLUX, "bc_f(t, y)") # x=0
boundary(u, 2, NO_BC) # x=0.1
boundary(u, 3, NO_BC) # y=0
boundary(u, 4, NO_BC) # y=0.3

# Time interval and initial condition
# T = 1.3;
# timeInterval(T)
nsteps = 100
dt = 1.3 / nsteps
setSteps(dt, nsteps)
initial(u, "0")

# Coefficients
coefficient("a", ["0.1*cos(pi*x/2/0.1)","0.3*sin(pi*x/2/0.1)"], type=VECTOR) # advection velocity
coefficient("s", ["0.1 * sin(pi*x)^4 * sin(pi*y)^4"]) # source

# The "upwind" function applies upwinding to the term (a.n)*u with flow velocity a.
# The optional third parameter is for tuning. Default upwind = 0, central = 1. Choose something between these.
conservationForm(u, "s + surface(upwind(a,u))");

# exportCode("fvad2dgpucode") # uncomment to export generated code to a file
# importCode("profile/ad2d/fvad2dgpucode-fix")
importCode("profile/ad2d/fvad2dgpucode-fix-newkernel")

# solve(u)
@CUDA.profile solve(u)

outputValues(u, "fvad2dgpu-vals", format="vtk");

finalizeFinch()

##### Uncomment below to plot

# xy = Finch.fv_info.cellCenters

# using Plots
# pyplot();
# display(plot(xy[1,:], xy[2,:], u.values[:], st=:surface))
