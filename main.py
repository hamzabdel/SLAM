from slam.particle_filter import ParticleFilter
from simulation.robot import Robot
from simulation.environment import Environment
import matplotlib.pyplot as plt

# Creates environment with landmarks
env = Environment()
env.add_landmarks([(2,3), (5,1), (6,6)])

# Initializes robot
robot = Robot(x=1, y=1, theta=0)

# Initialize particle filter
pf = ParticleFilter(num_particles=100, env=env)

# Run a few steps
for _ in range(20):
    robot.move(1, 0.1)      # move forward + small rotation
    measurements = robot.sense()
    
    pf.predict(1, 0.1)      # motion update
    pf.update(measurements)  # weight update
    pf.resample()            # resample particles
    
    pf.plot(robot)           # visualize particles & robot
