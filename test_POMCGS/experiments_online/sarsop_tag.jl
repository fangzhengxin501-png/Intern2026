using POMDPs
using SARSOP
using POMDPModels
using Random 

Random.seed!(1)


pomdp = TagPOMDP()
solver = SARSOPSolver()
policy = solve(solver, pomdp)