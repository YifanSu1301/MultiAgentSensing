using PyPlot
using HDF5, JLD
using Statistics

using SubmodularMaximization

pygui(false)
name = "compare_solvers"
fig_path = "./fig/$name"
data_path = "./data/$name"
mkpath(fig_path)
mkpath(data_path)

num_trials = 100

num_agents = 5
num_sensors = 5
nominal_area = 2.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

f(x) = mean_area_coverage(x, 50)

solvers = Any[]

#push!(solvers, (solve_optimal, "Optimal"))
#push!(solvers, (solve_worst, "Worst-case"))
push!(solvers, (solve_random, "Random"))
push!(solvers, (solve_myopic, "Myopic"))

partitions = [2, 4, 8]
for num_partitions in partitions
  solve_n(p) = solve_n_partitions(num_partitions, p)
  push!(solvers, (solve_n, "Partition-$num_partitions"))
end

push!(solvers, (solve_sequential, "Sequential"))
push!(solvers, (solve_continuous, "Continuous"))
# solve_multi(p) = solve_sequential_multiround(p,3)
# push!(solvers, (solve_multi, "Multi-sequential"))

results = zeros(num_trials, length(solvers))

problems = map(1:num_trials) do unused
  agents = generate_agents(agent_specification, num_agents)
  problem = ExplicitPartitionProblem(f, agents)
end
@save compress = false "$data_path/partition_matroids" (map(x->x.partition_matroid, problems))

for trial_num in 1:length(problems)
  println("Trial: $trial_num")

  for solver_num in 1:length(solvers)
    solver = solvers[solver_num][1]
    name = solvers[solver_num][2]

    solution = solver(problems[trial_num])

    results[trial_num, solver_num] = solution.value
  end
end
@save compress = false "$data_path/results" results

# now analyze weights
weight_matrices = map(problems, 1:length(problems)) do problem, ii
  println("Weight matrix $ii")
  compute_weight_matrix(problem)
end

@save compress = false "$data_path/weights" weight_matrices

total_weights = map(weight_matrices, 1:length(problems)) do weight_matrix, ii
  println("Total weights $ii")
  total_weight(weight_matrix)
end

edge_sets = map(weight_matrices, 1:length(problems)) do weight_matrix, ii
  println("Triangle $ii")
  extract_triangle(weight_matrix)
end


figure()
boxplot(results, notch=false, vert=false)
yticks(1:length(solvers), map(x->x[2], solvers))
xlabel("Area coverage")
tight_layout()

save_fig(fig_path, "results")

# plot histograms of edge weights

figure()
PyPlot.plt.hist(total_weights, 20)
tt = "Total Graph Weight Frequency"
ylabel("Frequency")
xlabel("Redundancy graph weight")
tight_layout()

save_fig(fig_path, tt)
title(tt)

figure()
PyPlot.plt.hist(vcat(edge_sets...), 20)
tt = "Edge Weight Frequency"
ylabel("Frequency")
xlabel("Edge weight (redundancy)")
tight_layout()

save_fig(fig_path, tt)
title(tt)

# sequential solutions and edge weights
sequential_mean = mean(results[:,end][:])

partition_values = mean(results[:,3:end-1]; dims=1)[:]

mean_weight = mean(total_weights)
println("Mean weight: $mean_weight")
bounds = map(x->mean_weight/x, partitions)

#figure()
#plot([partitions[1], partitions[end]], [sequential_mean, sequential_mean])
#plot(partitions, partition_values)
#plot(partitions, partition_values + bounds)
#legend(["Sequential", "Partition-n"])

nothing