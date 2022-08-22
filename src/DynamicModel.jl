#= Solve the dynamic EGSS model (first pass). Annual calibration 
Package published at https://github.com/meghanagaur/DynamicModel. =#
module DynamicModel # begin module

export  rouwenhorst, model, solveModel, unemploymentValue # , simulateProd, simulateWages,
#solveModelSavings, simulateWagesSavings,unemploymentValueSavings,

using DataStructures, Distributions, ForwardDiff, Interpolations,
 LinearAlgebra, Parameters, Random, Roots, StatsBase

include("dep/rouwenhorst.jl")
#include("dep/EGSS-FiniteHorizon.jl")
include("dep/EGSS-InfHorizon.jl")
#include("dep/EGSS-savings.jl")

end # module
