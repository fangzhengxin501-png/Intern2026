using POMCGraphSearch
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using CSV, DataFrames
using Random 
using Dates

Random.seed!(1)

pomdp = LightDark1D()
pomcgs = SolverPOMCGS(pomdp;
    max_b_gap=0.1,   #0.05 original, 0.1 given in table 2
    max_search_depth=30,
    num_fixed_observations=10, # 20 original
    state_grid = [1.0, 1.0],
    num_sim_per_sa=500,
    nb_particles=5000  #offset 10000, 5000 given in table 2
)

planning_time = 3.0 #2.0 original
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
#CSV.write("pomcgs_results_LD.csv", df)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
CSV.write("on_off_compare_data/LD_table2_$(timestamp).csv", df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results))")
println("Results saved to LD_table2_$(timestamp).csv")