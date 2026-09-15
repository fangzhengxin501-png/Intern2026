using POMCGraphSearch
using POMDPs
using Random
using POMDPTools
using RockSample

Random.seed!(1)

pomdp = RockSamplePOMDP(11, 11)  

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.3,
    max_search_depth = 50,
    nb_particles = 10000, #original 50000
    epsilon_VMDP = 0.01,
    num_sim_per_sa = 20
)


solve(pomcgs, pomdp)

#ExportLogData(pomcgs, "RS11_log")
#SavePrunedPolicyJSON(pomcgs.fsc, outfile_name="RS1111_policy")
#SavePrunedPolicyDOT(pomcgs.fsc, outfile_name="RS1111_policy")