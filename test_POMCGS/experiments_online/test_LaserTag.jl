using POMCGraphSearch
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using LaserTag
using CSV, DataFrames
using Random 
using Dates

Random.seed!(1)

rng = MersenneTwister(7)
pomdp = gen_lasertag(rng=rng, robot_position_known=false)
p_1 = 30 #offset: 20


pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.03, # 这个也调一下，主要范围[0.03, 0.3]之间
    max_search_depth = 50, # 这个最多可以调到100，但最好固定
    num_fixed_observations = p_1, # 主要调这个，比如试试[3,5,10,20,50,100]这种, this is setting of k-cluster
    num_sim_per_sa = 1000,
)

# deal with different number of clulsters with offset settings first
 
planning_time = 10.0 # 这个视情况可以稍微增大，但要保证每个实验用的时间一致 in range [1,10]
max_depth = 100
nb_runs = 100


results = Float64[]

for i in 1:nb_runs
    run_return = POMCGraphSearch.SolveOnline(pomcgs, max_depth, planning_time; verbose=true)
    push!(results, run_return)
    println("Run $i: $run_return")
end

# Save results to CSV
df = DataFrame(run_id = 1:nb_runs, return_value = results)
#CSV.write("pomcgs_results_LaserTag.csv", df)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
CSV.write("0.03_10_nobs$(p_1)_$(timestamp).csv", df)
println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results))")

