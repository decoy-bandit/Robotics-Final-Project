"""
Monte Carlo Localization in Custom Simulator

Robotics Project 2020

Group V
"""

import numpy as np
import turtle
import bisect
import argparse
import time
import random
from math import radians


class Environment(object):

    def __init__(self, cell_height, cell_width, num_rows = None,
                num_cols = None, random_seed = None):

        self.cell_height = cell_height
        self.cell_width = cell_width

        self.random_env(num_rows = num_rows, num_cols = num_cols, random_seed = random_seed)

        self.height = self.num_rows * self.cell_height
        self.width = self.num_cols * self.cell_width

        self.turtle_registration()

    def turtle_registration(self):

        turtle.register_shape('tri', ((-3, -2), (0, 3), (3, -2), (0, 0)))


    def random_env(self, num_rows, num_cols, random_seed = None):

        if random_seed is not None:
            np.random.seed(random_seed)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.env = np.zeros((num_rows, num_cols), dtype = np.int8)
        for i in range(self.num_rows):
            for j in range(self.num_cols-1):
                if np.random.rand() < 0.25:
                    self.env[i,j] |= 2
        for i in range(self.num_rows-1):
            for j in range(self.num_cols):
                if np.random.rand() < 0.25:
                    self.env[i,j] |= 4

        for i in range(self.num_rows):
            self.env[i,0] |= 8
            self.env[i,-1] |= 2
        for j in range(self.num_cols):
            self.env[0,j] |= 1
            self.env[-1,j] |= 4

        wall_errors = []

        for i in range(self.num_rows):
            for j in range(self.num_cols-1):
                if (self.env[i,j] & 2 != 0) != (self.env[i,j+1] & 8 != 0):
                    wall_errors.append(((i,j), 'v'))
        for i in range(self.num_rows-1):
            for j in range(self.num_cols):
                if (self.env[i,j] & 4 != 0) != (self.env[i+1,j] & 1 != 0):
                    wall_errors.append(((i,j), 'h'))

        for (i,j), error in wall_errors:
            if error == 'v':
                self.env[i,j] |= 2
                self.env[i,j+1] |= 8
            elif error == 'h':
                self.env[i,j] |= 4
                self.env[i+1,j] |= 1

    def check_boundaries(self, cell):

        cell_value = self.env[cell[0], cell[1]]
        return (cell_value & 1 == 0, cell_value & 2 == 0, cell_value & 4 == 0, cell_value & 8 == 0)

    def obstacle_distance(self, coordinates, theta):

        x, y = coordinates
        theta = theta % 360
        i = int(x // self.cell_width)
        j = int(y // self.cell_height)

        if theta >= 0 and theta < 90:
            d_h = np.inf
            d_v = np.inf
            for p in range(i, num_cols):
                x_t = (p+1)*cell_width
                y_t = y - 1/np.tan(radians(theta))*( x_t - x)
                j_t = min(int(y_t // cell_height), num_cols - 1)
                j_t = max(j_t, 0)
                i_t = p
                if self.check_boundaries(cell = (i_t, j_t))[1] == False:
                    d_h = np.sqrt((x_t - x)**2 + (y_t - y)**2)
                    break

            for q in range(j, 0, -1):
                y_t = (q - 1) * cell_height
                x_t = x + np.tan(radians(theta)) * (y - y_t)
                j_t = q
                i_t = min(int(x_t // cell_width), num_rows - 1)
                i_t = max(i_t, 0)
                if self.check_boundaries(cell = (i_t, j_t))[0] == False:
                    d_v = np.sqrt((x_t - x)**2 + (y_t - y)**2)
                    break

            d = min(d_h, d_v)

        elif theta >= 90 and theta < 180:

            theta_t = theta - 90
            d_h = np.inf
            d_v = np.inf
            for p in range(i, num_cols):
                x_t = (p + 1) * cell_width
                y_t = y + np.tan(radians(theta_t))*( x_t - x)
                j_t = min(int(y_t // cell_height), num_cols - 1)
                j_t = max(j_t, 0)
                i_t = p
                if self.check_boundaries(cell = (i_t, j_t))[1] == False:
                    d_h = np.sqrt((x_t - x)**2 + (y_t - y)**2)
                    break

            for q in range(j, num_rows):
                y_t = (q + 1) * cell_height
                x_t = x + 1/np.tan(radians(theta_t)) * (y_t - y)
                j_t = q
                i_t = min(int(x_t // cell_width), num_rows - 1)
                i_t = max(i_t, 0)
                if self.check_boundaries(cell = (i_t, j_t))[2] == False:
                    d_v = np.sqrt((x_t - x)**2 + (y_t - y)**2)
                    break

            d = min(d_h, d_v)

        elif theta >= 180 and theta < 270:

            theta_t = theta - 180
            d_h = np.inf
            d_v = np.inf
            for p in range(i, -1 , -1):
                x_t = p * cell_width
                y_t = y + 1/np.tan(radians(theta_t))*(x - x_t)
                j_t = min(int(y_t // cell_height), num_cols - 1)
                j_t = max(j_t, 0)
                i_t = p
                if self.check_boundaries(cell = (i_t, j_t))[3] == False:
                    d_h = np.sqrt((x_t - x)**2 + (y_t - y)**2)
                    break

            for q in range(j, num_rows):
                y_t = (q + 1) * cell_height
                x_t = x - np.tan(radians(theta_t)) * (y_t - y)
                j_t = q
                i_t = min(int(x_t // cell_width), num_rows - 1)
                i_t = max(i_t, 0)
                if self.check_boundaries(cell = (i_t, j_t))[2] == False:
                    d_v = np.sqrt((x_t - x)**2 + (y_t - y)**2)
                    break

            d = min(d_h, d_v)

        elif theta >= 270 and theta < 360:

            theta_t = theta - 270
            d_h = np.inf
            d_v = np.inf
            for p in range(i, -1, -1):
                x_t = p * cell_width
                y_t = y - np.tan(radians(theta_t))*( x - x_t)
                j_t = min(int(y_t // cell_height), num_cols - 1)
                j_t = max(j_t, 0)
                i_t = p
                if self.check_boundaries(cell = (i_t, j_t))[3] == False:
                    d_h = np.sqrt((x_t - x)**2 + (y_t - y)**2)
                    break

            for q in range(j, -1, -1):
                y_t = q  * cell_height
                x_t = x - 1/np.tan(radians(theta_t)) * (y - y_t)
                j_t = q
                i_t = min(int(x_t // cell_width), num_rows - 1)
                i_t = max(i_t, 0)
                if self.check_boundaries(cell = (i_t, j_t))[0] == False:
                    d_v = np.sqrt((x_t - x)**2 + (y_t - y)**2)
                    break

            d = min(d_h, d_v)

        return d


    def show_env(self):

        turtle.setworldcoordinates(0, 0, self.width * 1.005, self.height * 1.005)

        env_maker = turtle.Turtle()
        env_maker.speed(0)
        env_maker.width(1.5)
        env_maker.hideturtle()
        turtle.tracer(0, 0)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                check_boundaries = self.check_boundaries(cell = (i,j))
                turtle.up()
                env_maker.setposition((j * self.cell_width, i * self.cell_height))
                # Set turtle heading orientation
                # 0 - east, 90 - north, 180 - west, 270 - south
                env_maker.setheading(0)
                if not check_boundaries[0]:
                    env_maker.down()
                else:
                    env_maker.up()
                env_maker.forward(self.cell_width)
                env_maker.setheading(90)
                env_maker.up()
                if not check_boundaries[1]:
                    env_maker.down()
                else:
                    env_maker.up()
                env_maker.forward(self.cell_height)
                env_maker.setheading(180)
                env_maker.up()
                if not check_boundaries[2]:
                    env_maker.down()
                else:
                    env_maker.up()
                env_maker.forward(self.cell_width)
                env_maker.setheading(270)
                env_maker.up()
                if not check_boundaries[3]:
                    env_maker.down()
                else:
                    env_maker.up()
                env_maker.forward(self.cell_height)
                env_maker.up()

        turtle.update()


    def weight_to_color(self, weight):
        #print(weight)


        return '#%02x00%02x' % (int(weight * 255), int((1 - weight) * 255))


    def show_particles(self, particles, show_frequency = 10):

        turtle.shape('tri')

        for i, particle in enumerate(particles):
            if i % show_frequency == 0:
                turtle.setposition((particle.x, particle.y))
                turtle.setheading(90 - particle.theta)
                turtle.color(self.weight_to_color(particle.weight))
                turtle.stamp()

        turtle.update()

    def show_estimated_location(self, particles):


        x_accum = 0
        y_accum = 0
        theta_accum = 0
        weight_accum = 0

        num_particles = len(particles)

        for particle in particles:

            weight_accum += particle.weight
            x_accum += particle.x * particle.weight
            y_accum += particle.y * particle.weight
            theta_accum += particle.theta * particle.weight

        if weight_accum == 0:

            return False

        x_estimate = x_accum / weight_accum
        y_estimate = y_accum / weight_accum
        theta_estimate = theta_accum / weight_accum

        turtle.color('red')
        turtle.setposition(x_estimate, y_estimate)
        turtle.setheading(90 - theta_estimate)
        turtle.shape('arrow')
        turtle.stamp()
        turtle.update()

        self.estimated_location = np.array([x_estimate, y_estimate, theta_estimate])

    def show_robot(self, robot):

        turtle.color('green')
        turtle.shape('arrow')
        turtle.shapesize(0.7, 0.7)
        turtle.setposition((robot.x, robot.y))
        turtle.setheading(90 - robot.theta)
        turtle.stamp()
        turtle.update()

    def clear_objects(self):
        turtle.clearstamps()


class Particle(object):

    def __init__(self, x, y, env, theta = None, weight = 1.0,
                odometry_noise = False, sensor_noise = False):

        if theta is None:
            theta = np.random.uniform(0,360)

        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.env = env
        self.odometry_noise = odometry_noise
        self.sensor_noise = sensor_noise

        if self.odometry_noise:
            std = max(self.env.cell_height, self.env.cell_width) * 0.5
            self.x = self.add_noise(x = self.x, std = std)
            self.y = self.add_noise(x = self.y, std = std)
            self.theta = self.add_noise(x = self.theta, std = 360 * 0.05)

        self.fix_invalid_particles()


    def fix_invalid_particles(self):

        # Fix invalid particles
        if self.x < 0:
            self.x = 0
        if self.x > self.env.width:
            self.x = self.env.width * 0.9999
        if self.y < 0:
            self.y = 0
        if self.y > self.env.height:
            self.y = self.env.height * 0.9999

        self.theta = self.theta % 360

    @property
    def state(self):

        return [self.x, self.y, self.theta]

    def add_noise(self, x, std):

        return x + np.random.normal(0, std)

    def read_sensor(self, env):

        reading_1 = env.obstacle_distance(coordinates = (self.x, self.y), theta = self.theta)
        reading_2 = env.obstacle_distance(coordinates = (self.x, self.y), theta = self.theta + 90)
        reading_3 = env.obstacle_distance(coordinates = (self.x, self.y), theta = self.theta + 180)
        reading_4 = env.obstacle_distance(coordinates = (self.x, self.y), theta = self.theta + 270)

        reading_1 = min(reading_1, np.sqrt(env.height**2 + env.width**2))
        reading_2 = min(reading_2, np.sqrt(env.height**2 + env.width**2))
        reading_3 = min(reading_3, np.sqrt(env.height**2 + env.width**2))
        reading_4 = min(reading_4, np.sqrt(env.height**2 + env.width**2))

        readings = np.array([reading_1, reading_2, reading_3, reading_4])

        return readings

    def try_move(self, speed, env):

        theta = self.theta
        theta_rad = np.radians(theta)

        dx = np.sin(theta_rad) * speed
        dy = np.cos(theta_rad) * speed

        x = self.x + dx
        y = self.y + dy

        gj1 = int(self.x // env.cell_width)
        gi1 = int(self.y // env.cell_height)
        gj2 = int(x // env.cell_width)
        gi2 = int(y // env.cell_height)

        # Check if the particle is still in the env
        if gi2 < 0 or gi2 >= env.num_rows or gj2 < 0 or gj2 >= env.num_cols:
            return False

        # Move in the same cell
        if gi1 == gi2 and gj1 == gj2:
            self.x = x
            self.y = y
            return True
        # Move across one cell vertically
        elif abs(gi1 - gi2) == 1 and abs(gj1 - gj2) == 0:
            if env.env[min(gi1, gi2), gj1] & 4 != 0:
                return False
            else:
                self.x = x
                self.y = y
                return True
        # Move across one cell horizonally
        elif abs(gi1 - gi2) == 0 and abs(gj1 - gj2) == 1:
            if env.env[gi1, min(gj1, gj2)] & 2 != 0:
                return False
            else:
                self.x = x
                self.y = y
                return True
        # Move across cells both vertically and horizonally
        elif abs(gi1 - gi2) == 1 and abs(gj1 - gj2) == 1:

            x0 = max(gj1, gj2) * env.cell_width
            y0 = (y - self.y) / (x - self.x) * (x0 - self.x) + self.y

            if env.env[int(y0 // env.cell_height), min(gj1, gj2)] & 2 != 0:
                return False

            y0 = max(gi1, gi2) * env.cell_height
            x0 = (x - self.x) / (y - self.y) * (y0 - self.y) + self.x

            if env.env[min(gi1, gi2), int(x0 // env.cell_width)] & 4 != 0:
                return False

            self.x = x
            self.y = y
            return True


class Robot(Particle):

    def __init__(self, x, y, env, theta = None, speed = 1.0,odometry_noise = False,
                sensor_noise = False):

        super(Robot, self).__init__(x = x, y = y, env = env,
                                    theta = theta, odometry_noise = odometry_noise,
                                    sensor_noise = sensor_noise)
        self.step_count = 0
        self.odometry_noise = odometry_noise
        self.sensor_noise = sensor_noise
        self.time_step = 0
        self.speed = speed

    def choose_random_direction(self):

        self.theta = np.random.uniform(0, 360)

    def kidnap(self, env):
        self.x = np.random.uniform(0, env.width)
        self.y = np.random.uniform(0, env.height)


    def add_sensor_noise(self, x, z = 0.05):

        readings = list(x)

        for i in range(len(readings)):
            std = readings[i] * z / 2
            readings[i] = readings[i] + np.random.normal(0, std)

        return readings

    def read_sensor(self, env):

        # Robot has error in reading the sensor while particles do not.

        readings = super(Robot, self).read_sensor(env = env)


        if self.sensor_noise == True:
            readings = self.add_sensor_noise(x = readings)

        return readings

    def move(self, env):

        while True:
            self.time_step += 1
            if self.try_move( speed = self.speed, env = env):
                break
            self.choose_random_direction()


class WeightedDistribution(object):

    def __init__(self, particles):

        accum = 0.0
        self.particles = particles
        self.distribution = []
        for particle in self.particles:
            accum += particle.weight
            self.distribution.append(accum)

    def random_select(self):

        try:
            particle = self.particles[bisect.bisect_left(self.distribution, np.random.uniform(0, 1))]
            #std = np.exp(-particle.weight) * 50
            #std = max(particle.env.cell_height, particle.env.cell_width) * 0.2
            #particle.x += np.random.normal(0, std)
            #particle.y += np.random.normal(0, std)

            return particle
        except IndexError:
            # When all particles have weights zero
            return None





def weight_gaussian_kernel(x1, x2, std = 10):

    distance = np.linalg.norm(np.asarray(x1) - np.asarray(x2))
    return np.exp(-distance ** 2 / (2 * std))


def main(window_width, window_height, num_particles, cell_height, cell_width,
        num_rows, num_cols, random_seed, kernel_sigma, robot_speed,
        particle_show_frequency, tolerance = 10):


    window = turtle.Screen()
    window.setup (width = window_width, height = window_height)

    world = Environment(cell_height = cell_height, cell_width = cell_width,
                        num_rows = num_rows, num_cols = num_cols,
                        random_seed = random_seed)

    x = np.random.uniform(0, world.width)
    y = np.random.uniform(0, world.height)
    burger = Robot(x = x, y = y, env = world, speed = robot_speed)

    particles = []
    for i in range(num_particles):
        x = np.random.uniform(0, world.width)
        y = np.random.uniform(0, world.height)
        particles.append(Particle(x = x, y = y, env = world))

    time.sleep(1)
    world.show_env()
    steps = 0

    while True:

        #if steps == 15:
            #burger.kidnap(world)

        readings_robot = burger.read_sensor(env = world)
        steps += 1

        particle_weight_total = 0
        for particle in particles:
            readings_particle = particle.read_sensor(env = world)
            particle.weight = weight_gaussian_kernel(x1 = readings_robot, x2 = readings_particle, std = kernel_sigma)
            particle_weight_total += particle.weight

        world.show_particles(particles = particles, show_frequency = particle_show_frequency)
        world.show_robot(robot = burger)
        world.show_estimated_location(particles = particles)
        world.clear_objects()

        # Make sure normalization is not divided by zero
        if particle_weight_total == 0:
            particle_weight_total = 1e-8

        # Normalize particle weights
        for particle in particles:
            particle.weight /= particle_weight_total

        # Resampling particles
        distribution = WeightedDistribution(particles = particles)
        particles_new = []

        for i in range(num_particles):

            particle = distribution.random_select()

            #particle = Gaussian_Mixture_sample(particles = particles)

            if particle is None:
                x = np.random.uniform(0, world.width)
                y = np.random.uniform(0, world.height)
                particles_new.append(Particle(x = x, y = y, env = world))

            else:
                particles_new.append(Particle(x = particle.x, y = particle.y, env = world, theta = particle.theta, odometry_noise = True))#True

        particles = particles_new

        theta_old = burger.theta
        burger.move(env = world)
        theta_new = burger.theta
        dh = theta_new - theta_old

        for particle in particles:
            particle.theta = (particle.theta + dh) % 360
            particle.try_move(env = world, speed = burger.speed)

        robot_state = np.array(burger.state)
        estimated_state = world.estimated_location
        localization_error = np.linalg.norm(robot_state - estimated_state)
        #print(tolerance)
        print(localization_error)
        print(robot_state - estimated_state)
        if localization_error < tolerance:
            print(localization_error)
            break

    print(steps)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Particle filter in env.')

    window_width_default = 800
    window_height_default = 800
    num_particles_default = 5000
    cell_height_default = 100
    cell_width_default = 100
    num_rows_default = 25
    num_cols_default = 25
    robot_speed_default = 10
    random_seed_default = 100
    kernel_sigma_default = 500
    particle_show_frequency_default = 5

    parser.add_argument('--window_width', type = int, help = 'Window width.', default = window_width_default)
    parser.add_argument('--window_height', type = int, help = 'Window height.', default = window_height_default)
    parser.add_argument('--num_particles', type = int, help = 'Number of particles used in particle filter.', default = num_particles_default)
    parser.add_argument('--cell_height', type = int, help = 'Height for each cell of env.', default = cell_height_default)
    parser.add_argument('--cell_width', type = int, help = 'Width for each cell of env.', default = cell_width_default)
    parser.add_argument('--num_rows', type = int, help = 'Number of rows in env', default = num_rows_default)
    parser.add_argument('--num_cols', type = int, help = 'Number of columns in env', default = num_cols_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for random env and particle filter.', default = random_seed_default)
    parser.add_argument('--robot_speed', type = int, help = 'Robot movement speed in maze.', default = robot_speed_default)
    parser.add_argument('--kernel_sigma', type = int, help = 'Standard deviation for Gaussian distance kernel.', default = kernel_sigma_default)
    parser.add_argument('--particle_show_frequency', type = int, help = 'Frequency of showing particles on env.', default = particle_show_frequency_default)

    argv = parser.parse_args()

    window_width = argv.window_width
    window_height = argv.window_height
    num_particles = argv.num_particles
    cell_height = argv.cell_height
    cell_width = argv.cell_width
    num_rows = argv.num_rows
    num_cols = argv.num_cols
    random_seed = argv.random_seed
    robot_speed = argv.robot_speed
    kernel_sigma = argv.kernel_sigma
    particle_show_frequency = argv.particle_show_frequency

    main(window_width = window_width, window_height = window_height, num_particles = num_particles,
        cell_height = cell_height, cell_width = cell_width, num_rows = num_rows, num_cols = num_cols,
        random_seed = random_seed, robot_speed = robot_speed, kernel_sigma = kernel_sigma,
         particle_show_frequency = particle_show_frequency)
