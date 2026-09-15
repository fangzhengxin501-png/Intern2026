using POMCGraphSearch
using POMDPs
using Random
using POMDPTools
using POMDPModels

Random.seed!(1)

pomdp = LightDark1D()
pomcgs = SolverPOMCGS(pomdp;
    max_b_gap=0.2,
    max_planning_secs=3600.0, 
    max_search_depth=40,
    num_fixed_observations=20,
    state_grid = [1.0, 1.0],
    num_sim_per_sa=2000,
)

solve(pomcgs, pomdp)
ExportLogData(pomcgs, "Lightdark")
SavePrunedPolicyJSON(pomcgs.fsc, outfile_name="LightDark_policy")
SavePrunedPolicyDOT(pomcgs.fsc, outfile_name="LightDark_policy")