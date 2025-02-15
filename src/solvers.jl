# A collection of solvers for submodular maximization problems with partition
# matroid constraints
#
# Some solvers are only applicable to certain problem types. Note type
# constraints.

# Utilities

# Optionally threaded loop
function optionally_threaded(f, collection; threaded = false)
  if threaded
    Threads.@threads for x in collection
      f(x)
    end
  else
    for x in collection
      f(x)
    end
  end
end
#######################################
# Multiround Sequential solvers (Test)
#######################################
function solve_sequential_multiround(p::PartitionProblem, num_rounds::Int)
  selection = empty(p)

  for iter = 1:num_rounds
    for ii = 1:get_num_agents(p)
      # filter returns element values, findall returns element indices
      prev_sol = findall(t -> t[1]==ii,selection)
      
      # check if ii in selection
      if !isempty(prev_sol) 
        # remove ii from selection
        deleteat!(selection, prev_sol)
      end

      solution_element = solve_block(p, ii, selection)

      push!(selection, solution_element)
    end
  end

  evaluate_solution(p, selection)
end

#######################################
# Continuous solvers (Test)
#######################################
# Define a function to estimate the expected marginal profit of player i from item j
function expected_marginal_profit(i, j, y, p, my_dict)
  # Generate a random set Ri containing each item j independently with probability yij(t)
  num_agents = length(p.partition_matroid)
  num_sensors = length(p.partition_matroid[1].sensors)
  R =[]
  keyM = zeros(num_sensors, num_agents)
  
  for agents = 1 : num_agents
    if agents == j
      continue
    end
    for sensors = 1 : num_sensors
      if rand() < y[sensors, agents]
        keyM[sensors, agents] = 1
        push!(R, p.partition_matroid[agents].sensors[sensors])
      end
    end
    
  end

  if haskey(my_dict, keyM)
    wi_Ri =  my_dict[keyM]
  else
    # println(keyM)
    wi_Ri = p.objective(R)
    my_dict[keyM] = wi_Ri
  end

  # Compute the expected marginal profit of player i from item j
  
  push!(R, p.partition_matroid[j].sensors[i])
 
  keyMj = copy(keyM)
  keyMj[i,j] = 1
  if haskey(my_dict, keyMj)
    wi_Ri_j =  my_dict[keyMj]
  else
    wi_Ri_j =  p.objective(R)
    my_dict[keyMj] = wi_Ri_j
  end
  
  return wi_Ri_j - wi_Ri
end

function solve_continuous(p::PartitionProblem)
  num_agents = length(p.partition_matroid)
  num_sensors = length(p.partition_matroid[1].sensors)
  # Initialize y matrix
  y = zeros(num_sensors, num_agents)

  # Initialize t and delta
  t = 0.0
  delta = 1.0 / ((num_agents*num_sensors)^2)

  my_dict = Dict()
  # Run the algorithm
  while t < 1
      # Estimate the expected marginal profits for all players and items
      ω = zeros(num_sensors, num_agents)
      for i = 1:num_sensors
         for j = 1:num_agents
              for k = 1:(num_agents*num_sensors)^3
                  ω[i, j] += expected_marginal_profit(i, j, y, p, my_dict)
              end
              ω[i, j] /= (num_agents*num_sensors)^3
              
          end
      end
      # Update y matrix
      # println(y)
      # println(ω)
      for j = 1:num_agents
          i_star = argmax(ω[:, j])
          y[i_star, j] += delta
      end
      # Increment t
      t += delta
      # println(t)
      # println(t)
  end
  # println(y)
  selection = empty(p)
  for j = 1:length(p.partition_matroid)
      y_j = copy(y[:, j])
      y_j = sort(y_j)
      # println(y_j)
      idx = findfirst(x -> x > rand(),y_j)
      # println(idx)
      if(idx == nothing)
        _, idx = findmax(y_j)
        # println(idx)
      end
      for val = 1:num_sensors
        if y[:,j][val] == y_j[idx]
          idx = val
          break
        end
      end
      # println(idx)
      push!(selection, (j, idx))
  end
  # println(selection)
  evaluate_solution(p, selection)
end

#######################################
# Basic solvers (sequential and myopic)
#######################################

# Myopic solver
function solve_myopic(p::PartitionProblem; threaded=false)
  selection = ElementArray(p)(undef, get_num_agents(p))

  optionally_threaded(1:get_num_agents(p), threaded=threaded) do ii
    # Solve given no knowledge of prior decisions
    solution_element = solve_block(p, ii, empty(p))

    selection[ii] = solution_element
  end

  evaluate_solution(p, selection)
end

# sequential solver
function solve_sequential(p::PartitionProblem)
  selection = empty(p)

  for ii = 1:get_num_agents(p)
    solution_element = solve_block(p, ii, selection)

    push!(selection, solution_element)
  end
  # println(selection)
  evaluate_solution(p, selection)
 
end

#############################################################
# Solvers that require direct access to the partition matroid
#
# Optimal, worst-case, and randomized solvers
#
# These solvers require explicit representations
#############################################################

# optimal solver
function solve_optimal(p::ExplicitPartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  v0 = evaluate_solution(p, empty())

  # collect because product returns a tuple
  op(s::Solution, b) = partial_max(s, evaluate_solution(p, collect(b)))

  foldl(op, product(indices...); init=v0)
end

# anti-optimal solver
function solve_worst(p::ExplicitPartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  v0 = Solution(Inf, empty())

  # collect because product returns a tuple
  op(s::Solution, b) = partial_min(s, evaluate_solution(p, collect(b)))

  foldl(op, product(indices...), init=v0)
end

# random solver
function solve_random(p::ExplicitPartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  evaluate_solution(p, map(x->rand(x), indices))
end
# In general we require a method to produce random solutions
function sample_block(p::PartitionProblem, ind::Integer)
  error("Please define sample_block method to enable random sampling")
end
function solve_random(p::PartitionProblem)
  block_indices = 1:length(p.partition_matroid)

  evaluate_solution(p, map(x->sample_block(p, x), block_indices))
end

########################################################################
# DAG solver variants
#
# These solvers produce reduced versions of sequential greedy solvers by
# partitioning robots and pruning edges
########################################################################

export DAGSolver, solve_dag, sequence, in_neighbors, deleted_edge_weight,
solve_problem

abstract type DAGSolver end
# in_neighbors(d::DAGSolver, agent_index) = <agent in neighbors>
# sequence(d::DAGSolver) = <sequence of agent ids>

function solve_dag(d::DAGSolver, p::PartitionProblem; threaded=false)
  # The selections is an array of assignments on the matroid
  #
  # Generally, solution elements will line up with the blocks of the matroid,
  # but code should not rely on this fact
  selections = ElementArray(p)(undef, get_num_agents(p))

  for partition in partitions(d)
    function solve(agent_index)
      neighbor_selection::ElementArray(p) =
        map(x->selections[x], in_neighbors(d, agent_index))

      # Index of agent and its solution
      solution_element = solve_block(p, agent_index, neighbor_selection)

      # Store the index of agent's solution
      selections[agent_index] = solution_element
    end

    # Threaded execution is optional
    optionally_threaded(solve, partition; threaded=threaded)
  end

  evaluate_solution(p, selections)
end

solve_problem(d::DAGSolver, p::PartitionProblem; kwargs...) =
  solve_dag(d, p; kwargs...)

function deleted_edge_weight(d::DAGSolver, W::Array{Float64})
  weight = 0.0

  s = sequence(d)
  for ii in 1:length(s)
    agent_index = s[ii]

    nominal_neighbors = s[1:ii-1]
    neighbors = in_neighbors(d, agent_index)

    deleted_edges = setdiff(nominal_neighbors, neighbors)

    weight += reduce(deleted_edges; init=0.0) do w, edge
      w + W[agent_index, edge]
    end
  end

  weight
end

function deleted_edge_weight(d::DAGSolver, p::PartitionProblem)
  weights = compute_weight_matrix(p)

  deleted_edge_weight(d, weights)
end

###########################
# basic partitioned solvers
###########################

export PartitionSolver, generate_by_local_partition_size,
  generate_by_global_partition_size, solve_n_partitions

# helper function to construct the partitions for the solver
function construct_partitions(partition_numbers)
  num_partitions = maximum(partition_numbers)

  partitions = [Int64[] for x in 1:num_partitions]

  for agent_index in 1:length(partition_numbers)
    partition_index = partition_numbers[agent_index]

    push!(partitions[partition_index], agent_index)
  end

  partitions
end

# Generic partition solver
struct PartitionSolver <: DAGSolver
  # Array partitioning agents
  # Inner arrays are blocks and elements are agent ids
  partitions::Array{Array{Int64,1},1}
  # the index of the block in "partitions" containing a given agent_index
  agent_partition_numbers::Array{Int64,1}

  # Construct the dag using just the partition numbers
  PartitionSolver(x) = new(construct_partitions(x), x)
end

sequence(p::PartitionSolver) = vcat(p.partitions...)
partitions(p::PartitionSolver) = p.partitions

solver_rank(p::PartitionSolver) = length(p.agent_partition_numbers)
communication_span(p::PartitionSolver) = length(partitions(p))

function in_neighbors(p::PartitionSolver, agent_index)
  partition_index = p.agent_partition_numbers[agent_index]

  vcat(p.partitions[1:(partition_index-1)]...)
end

# general random partition solver framework from paper
function generate_by_local_partition_size(local_partition_sizes)
  partition_numbers = map(x->rand(1:x), local_partition_sizes)

  PartitionSolver(partition_numbers)
end

function generate_by_global_partition_size(num_agents, partition_size)
  generate_by_local_partition_size(fill(partition_size, num_agents))
end

# fixed number of partitions
function solve_n_partitions(num_partitions, p::PartitionProblem;
                            threaded=false)
  num_agents = length(p.partition_matroid)

  partition_solver = generate_by_global_partition_size(num_agents,
                                                       num_partitions)

  solve_dag(partition_solver, p, threaded=threaded)
end

#######################
# Adaptive partitioning
#######################

export compute_global_num_partitions, compute_local_num_partitions,
  generate_global_adaptive, generate_local_adaptive, solve_global_adaptive,
  solve_local_adaptive

# global adaptive number of partitions
function compute_global_num_partitions(desired_suboptimality,
                                       p::PartitionProblem)
  n = convert(Int64, ceil(total_weight(p)
                          / (get_num_agents(p)*desired_suboptimality)))
  max(1, n)
end

# compute nominal local number of partitions
function compute_local_num_partitions(desired_suboptimality,
                                      p::PartitionProblem)
  W  = compute_weight_matrix(p)

  map(1:length(p.partition_matroid)) do ii
    n = convert(Int64, ceil(sum(W[ii,:]) / (2 * desired_suboptimality)))
    max(1, n)
  end
end

# generate planner using global adaptive number of partitions
function generate_global_adaptive(desired_suboptimality, p::PartitionProblem)
  num_partitions = compute_global_num_partitions(desired_suboptimality, p)

  generate_by_global_partition_size(length(p.partition_matroid), num_partitions)
end

# generate planner using local adaptive number of partitions
function generate_local_adaptive(desired_suboptimality, p::PartitionProblem)
  local_partition_sizes = compute_local_num_partitions(desired_suboptimality, p)

  generate_by_local_partition_size(local_partition_sizes)
end

function solve_global_adaptive(desired_suboptimality, p::PartitionProblem)
  solver = generate_global_adaptive(desired_suboptimality, p)

  solve_dag(solver, p)
end

function solve_local_adaptive(desired_suboptimality, p::PartitionProblem)
  solver = generate_local_adaptive(desired_suboptimality, p)

  solve_dag(solver, p)
end

#######################
# range-limited solvers
#######################

export RangeSolver, solve_communication_range_limit

struct RangeSolver <: DAGSolver
  problem::PartitionProblem
  nominal_solver::DAGSolver
  communication_range::Float64
end

sequence(x::RangeSolver) = sequence(x.nominal_solver)
partitions(x::RangeSolver) = partitions(x.nominal_solver)

function in_neighbors(x::RangeSolver, agent_index)
  agents = x.problem.partition_matroid
  nominal_neighbors = in_neighbors(x.nominal_solver, agent_index)

  neighbors = Int64[]
  for neighbor in nominal_neighbors
    dist = norm(get_center(agents[neighbor]) - get_center(agents[agent_index]))

    if dist < x.communication_range
      push!(neighbors, neighbor)
    end
  end

  neighbors
end

function solve_communication_range_limit(p::PartitionProblem;
                                         num_partitions,
                                         communication_range,
                                         threaded=false)
  num_agents = length(p.partition_matroid)

  partition_solver = generate_by_global_partition_size(num_agents,
                                                       num_partitions)

  range_solver = RangeSolver(p, partition_solver, communication_range)

  solve_dag(range_solver, p, threaded=threaded)
end
