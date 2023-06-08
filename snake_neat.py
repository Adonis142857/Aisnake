import pygame
import numpy as np
import random
import os
import neat
import pickle

class SnakeGame:
    def __init__(self, width=480, height=480, snake_size=20, fps=30, max_steps=3000):
        self.width = width
        self.height = height
        self.snake_size = snake_size
        self.snake_pos = [[100, 100], [90, 100], [80, 100]]
        self.food_pos = self.generate_food()
        self.score = 0
        self.direction = 'RIGHT'
        self.is_over_flag = False
        self.steps = 0
        self.max_steps = max_steps

    def generate_food(self):
        return [random.randrange(1, self.width // self.snake_size) * self.snake_size,
                random.randrange(1, self.height // self.snake_size) * self.snake_size]

    def is_over(self):
        return self.is_over_flag

    def move(self, move):
        if self.steps >= self.max_steps:
            self.is_over_flag = True
            return

        if move == 0 and self.direction != 'DOWN':
            self.direction = 'UP'
        elif move == 1 and self.direction != 'UP':
            self.direction = 'DOWN'
        elif move == 2 and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        elif move == 3 and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        if self.direction == 'UP':
            new_head = [self.snake_pos[0][0], self.snake_pos[0][1] - self.snake_size]
        elif self.direction == 'DOWN':
            new_head = [self.snake_pos[0][0], self.snake_pos[0][1] + self.snake_size]
        elif self.direction == 'LEFT':
            new_head = [self.snake_pos[0][0] - self.snake_size, self.snake_pos[0][1]]
        else:
            new_head = [self.snake_pos[0][0] + self.snake_size, self.snake_pos[0][1]]

        if (new_head[0] < 0 or
            new_head[0] >= self.width or
            new_head[1] < 0 or
            new_head[1] >= self.height or
            new_head in self.snake_pos):
            self.is_over_flag = True
            return

        if new_head == self.food_pos:
            self.food_pos = self.generate_food()
            self.score += 1
        else:
            self.snake_pos.pop()

        self.snake_pos.insert(0, new_head)
        self.steps += 1

def extract_features(game):
    head = game.snake_pos[0]
    food = game.food_pos
    body = game.snake_pos[1:]
    tail = game.snake_pos[-1]

    features = [
        head[0] < food[0],  # food is on the right
        head[0] > food[0],  # food is on the left
        head[1] < food[1],  # food is below
        head[1] > food[1],  # food is above
        head[0] < game.snake_size,  # snake is near the left boundary
        head[0] > game.width - game.snake_size * 2,  # snake is near the right boundary
        head[1] < game.snake_size,  # snake is near the top boundary
        head[1] > game.height - game.snake_size * 2,  # snake is near the bottom boundary
        min([abs(head[0] - x[0]) + abs(head[1] - x[1]) for x in body]),  # min distance to the body in x-axis
        max([abs(head[0] - x[0]) + abs(head[1] - x[1]) for x in body]),  # max distance to the body in x-axis
        sum([1 for x in body if x[0] == head[0] and x[1] < head[1]]),  # number of body segments above head
        sum([1 for x in body if x[0] == head[0] and x[1] > head[1]]),  # number of body segments below head
        sum([1 for x in body if x[1] == head[1] and x[0] < head[0]]),  # number of body segments left of head
        sum([1 for x in body if x[1] == head[1] and x[0] > head[0]]),  # number of body segments right of head
        abs(head[0] - food[0]),  # horizontal distance between head and food
        abs(head[1] - food[1])   # vertical distance between head and food
    ]

    return np.array(features, dtype=float)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = SnakeGame()

        while not game.is_over():
            inputs = extract_features(game)
            output = net.activate(inputs)
            move = np.argmax(output)
            game.move(move)

        genome.fitness = (game.score * game.score * 2) - (game.steps * 1)  # 修改适应度计算

class CustomCheckpointer(neat.Checkpointer):
    def __init__(self, generation_interval, prefix):
        super().__init__(generation_interval)
        self.prefix = prefix

    def save_checkpoint(self, config, population, species_set, generation):
        """Save the current simulation state."""
        filename = '{0}neat-checkpoint-{1}'.format(self.prefix, generation)
        print("Saving checkpoint to {0}".format(filename))

        with open(filename, 'wb') as f:
            pickle.dump((generation, config, population, species_set, random.getstate()), f)

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(CustomCheckpointer(generation_interval=10, prefix='snake_neat_checkpoint_'))  # 修改 generation_interval 为 10

    winner = p.run(eval_genomes, 3000)  # 将训练代数增加到3000

    print('\nBest genome:\n{!s}'.format(winner))

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
