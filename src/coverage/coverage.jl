using PyPlot
using Colors

export Circle, CircleAgentSpecification,
  get_block,
  mean_coverage,
  mean_area_coverage

# evaluation of the area coverage objective
mutable struct Circle
  center::Array{Float64, 1}
  radius::Float64
end

struct CircleAgentSpecification
  sensor_radius
  station_radius
  num_sensors
end

# returns a vector of vectors of element indices abstracting the partition
# matroid

point_in_circle(center, radius, point) = norm(point - center) < radius
point_in_circle(circle::Circle, point) =
  point_in_circle(circle.center, circle.radius, point)

function point_covered(circles, x, y)
  #optimized point_in_circle code runs about ten times faster
  @inbounds for circle in circles
    if (x - circle.center[1])^2 + (y - circle.center[2])^2 < circle.radius^2
      return true
    end
  end

  false
end

mean_coverage(circles, points) =
  mean(map(point->point_covered(circles, point), points))

function mean_area_coverage(circles, discretization)
  num_covered = 0
  for x in range(0, stop=1, length=discretization),
    y in range(0, stop=1, length=discretization)

    num_covered += point_covered(circles, x, y)
  end

  num_covered / (discretization * discretization)
end


# agent generation
function rand_in_circle()
  r = sqrt(rand())
  theta = 2*pi*rand()
  r * [cos(theta);sin(theta)]
end

function make_agent(agent_specification::CircleAgentSpecification)
  agent_center = rand(2)
  make_agent(agent_specification, agent_center)
end
function make_agent(agent_specification::CircleAgentSpecification,
                    agent_center::Vector{Float64})
  make_sensor() = Circle(agent_center + agent_specification.station_radius *
                        rand_in_circle(),
                        agent_specification.sensor_radius)

  sensors = [make_sensor() for agent in 1:agent_specification.num_sensors]

  Agent(agent_center, agent_specification.station_radius, sensors)
end

function plot_circle(circle::Circle; x...)
  plot_circle(circle.center; radius = circle.radius, x...)
end

function plot_element(circle::Circle; color = "k", alpha = 0.3)
  center = circle.center
  scatter([center[1]], [center[2]], color = color, s = 4*agent_scale,
          marker = sensor_center, alpha = alpha)
end

function plot_agent(agent; color = "k")
  center = agent.center
  scatter([center[1]], [center[2]], color = color, s = 9*agent_scale,
          marker = "o", alpha = 0.3)

  plot_circle(Circle(agent.center, agent.radius);
              linestyle=station_style, color = color,
              linewidth = 3.0)
end

function plot_filled_circle(circle::Circle; color = "k", alpha = 0.3)
  ps = circle_points(circle.center, scale = 1, radius = circle.radius)
  full_circle = [ps ps[:,1]]

  ret = []

  append!(ret, fill(ps[1,:][:], ps[2,:][:], color = color, alpha=alpha, linewidth=0.0))

  append!(ret, plot_circle(circle, color="k", linewidth = 1.0))

  ret
end

function visualize_solution(circles::Array{Circle}, colors)
  map(circles, colors) do circle, color
    plot_filled_circle(circle, color = rgb_tuple(color), alpha=0.2)

    center = circle.center
    scatter([center[1]], [center[2]], color = rgb_tuple(color), s = 4*agent_scale,
            marker = selected_sensor, edgecolors="k")
  end
  nothing
end
