import pygame
import neat
import os
import random
pygame.font.init()

WIN_WIDTH = 400
WIN_HEIGHT = 600

GEN = 0

SPACESHIP_IMAGE = pygame.image.load("assets/agent/PlayerBlue_Frame_01_png_processed.png")

ASTEROID_IMAGE = pygame.image.load("assets/asteroids/Asteroid 01_png_processed.png")

BACKGROUND_IMAGE = pygame.image.load("assets/background/Space-Background.jpg")

STAT_FONT = pygame.font.SysFont("comicsans", 50)


class SpaceShip:
    IMG = SPACESHIP_IMAGE
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.img = self.IMG

    def slide_right(self):
        self.x += 2

    def slide_left(self):
        self.x -= 2

    def draw(self, win):
        spaceship_rect = self.img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(self.img, spaceship_rect)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Asteroid:
    VEL = 5

    def __init__(self):
        self.y = 0
        self.x = random.randint(0, WIN_WIDTH - 50)
        self.passed = False
        self.img = ASTEROID_IMAGE

    def move(self):
        self.y += self.VEL

    def draw(self, win):
        asteroid_rect = self.img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(self.img, asteroid_rect)

    def collide(self, spaceship):
        spaceship_mask = spaceship.get_mask()
        asteroid_mask = pygame.mask.from_surface(self.img)
        offset = (self.x - round(spaceship.x), self.y - round(spaceship.y))
        point = spaceship_mask.overlap(asteroid_mask, offset)
        if point:
            return True
        return False


def draw_window(win, spaceships, asteroids, score, gen):
    win.fill((0, 0, 0))

    win.blit(BACKGROUND_IMAGE, (0, 0))
    for asteroid in asteroids:
        asteroid.draw(win)

    text = STAT_FONT.render("Score:" + str(score), True, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen:" + str(gen), True, (255, 255, 255))
    win.blit(text, (10, 10))

    for spaceship in spaceships:
        spaceship.draw(win)

    pygame.display.update()


def main(genomes, config):
    global GEN
    GEN += 1

    nets = []
    ge = []
    spaceships = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        spaceships.append(SpaceShip(200, 450))
        g.fitness = 0
        ge.append(g)

    asteroids = [Asteroid()]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        asteroid_ind = 0
        if len(asteroids) > 0 and len(spaceships) > 0:
            if len(asteroids) > 1 and spaceships[0].y > asteroids[asteroid_ind].y+100:
                asteroid_ind += 1
        else:
            run = False
            break

        for i, spaceship in enumerate(spaceships):
            ge[i].fitness += 0.1
            output = nets[i].activate((spaceship.x, spaceship.x - asteroids[asteroid_ind].x,
                                       WIN_WIDTH - spaceship.x))
            if output[0] >= 0.5:
                spaceship.slide_right()
            elif output[0] <= -0.5:
                spaceship.slide_left()
            else:
                pass
        add_asteroid = False
        rem = []
        for asteroid in asteroids:
            for i, spaceship in enumerate(spaceships):
                if asteroid.collide(spaceship):
                    spaceships.remove(spaceship)
                    nets.pop(i)
                    ge.pop(i)
                if not asteroid.passed and asteroid.y > spaceship.y+100:
                    asteroid.passed = True
                    add_asteroid = True
            if asteroid.y > WIN_HEIGHT:
                rem.append(asteroid)

            asteroid.move()

        if add_asteroid:
            score += 1
            for g in ge:
                g.fitness += 5
            asteroids.append(Asteroid())

        for asteroid in rem:
            asteroids.remove(asteroid)
        for i, spaceship in enumerate(spaceships):
            if spaceship.x < 0 or spaceship.x > WIN_WIDTH-60:
                ge[i].fitness -= 5
                spaceships.remove(spaceship)
                nets.pop(i)
                ge.pop(i)

        draw_window(win, spaceships, asteroids, score, GEN)


def run_game(config):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(main, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config")
    run_game(config_path)
