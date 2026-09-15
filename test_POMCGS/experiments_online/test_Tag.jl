using POMCGraphSearch
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using TagPOMDPProblem
using CSV, DataFrames
using Random 

Random.seed!(1)
pomdp = TagPOMDP()

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.02,
    max_search_depth = 50,
    num_sim_per_sa = 1000
)

planning_time = 3.0
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
CSV.write("pomcgs_results_Tag.csv", df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results))")
