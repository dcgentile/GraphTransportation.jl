# Optimal Transport on Graphs
## Installation

To install this package, navigate in the commandline to your project directory, and start a Julia instance. Activate a virtual environment with

``` julia
]activate .
```

Now install the package with

``` julia
]add https://github.com/dcgentile/GraphTransportation.jl.git
```

Finally, you can test out the module with the following little example, which approximates the geodesic between two Dirac masses on a 2 point graph.

``` julia
using GraphTransportation
# establish your graph/markov kernel Q
Q = [0. 1.; 1. 0.];

# set measures
a = [2.0; 0];
b = [0.; 2];

# set number of steps (optional, will default to 128)
N = 100;

# call the Benamou Brenier Distances (BBD) f
v, dist = BBD(Q, a, b, N)
```

The variable v contains the vector information associated to the geodesic and all of its slack variables, while dist is the approximate Benamou-Brenier distance between the measures (i.e., the action of the computed geodesic, which is stored in v).
