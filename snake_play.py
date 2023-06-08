import pygame
import numpy as np
import pickle
import neat
from snake_neat import SnakeGame, extract_features

def play_game(winner_file):
    # Load the trained genome
    with open(winner_file, 'rb') as f:
        winner_genome = pickle.load(f)

    # Set up the NEAT configuration
    config_path = 'config-feedforward.txt'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the neural network from the genome
    winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    # Initialize the game and screen
    pygame.init()
    game = SnakeGame()
    screen = pygame.display.set_mode((game.width, game.height))
    pygame.display.set_caption('Snake Game')
    clock = pygame.time.Clock()

    while not game.is_over():
        # Get the current game state and extract features
        inputs = extract_features(game)

        # Use the neural network to get the move
        outputs = winner_net.activate(inputs)
        move = np.argmax(outputs)

        # Perform the move in the game
        game.move(move)

        # Render the game screen
        screen.fill((0, 0, 0))
        for pos in game.snake_pos:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pos[0], pos[1], game.snake_size, game.snake_size))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(game.food_pos[0], game.food_pos[1], game.snake_size, game.snake_size))
        pygame.display.flip()

        # Control the game speed
        clock.tick(10)

if __name__ == '__main__':
    winner_file = 'winner.pkl'
    play_game(winner_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the neural network from the genome
    winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    # (Rest of the function remains unchanged)
