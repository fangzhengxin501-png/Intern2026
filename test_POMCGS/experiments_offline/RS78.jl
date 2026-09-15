using POMCGraphSearch
using POMDPs
using Random
using POMDPTools
using RockSample

Random.seed!(1)

pomdp = RockSamplePOMDP(7, 8)  # Small problem for fast testing
pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.3,
    max_planning_secs=3600.0,  # Short time for tests
    max_search_depth=30,
    num_sim_per_sa=20
)

solve(pomcgs, pomdp)

ExportLogData(pomcgs, "RS78_log")
SavePrunedPolicyJSON(pomcgs.fsc, outfile_name="RS78_policy")
SavePrunedPolicyDOT(pomcgs.fsc, outfile_name="RS78_policy")