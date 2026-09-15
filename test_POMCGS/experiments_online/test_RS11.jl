using POMCGraphSearch
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using RockSample
using CSV, DataFrames
using Random
using Dates

Random.seed!(1)


pomdp = RockSamplePOMDP(11,11)

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.2, #offset 0.2, 
    max_search_depth = 80, #original 40 in offline
    num_sim_per_sa = 20,
    nb_particles = 10000,  #given in table 2, offset 10000
    epsilon_VMDP = 0.05 #offset is 0.1
)

planning_time = 6.0 #offset 2.0
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
#CSV.write("pomcgs_results_RS1111.csv", df)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
CSV.write("on_off_compare_data/RS11_6s_0.2_80_$(timestamp)_$(mean(results)).csv", df)


println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results))")
println("Results saved to RS11_6s_0.2_80_$(timestamp).csv")