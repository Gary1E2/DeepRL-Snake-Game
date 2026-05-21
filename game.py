import pygame
from random import randint
import numpy as np
import math

# game loop
class CarGame:
    def __init__(self):
        # init pygame
        pygame.init()

        # screen params
        self.cell_size = 32
        self.cell_rows = 12
        self.cell_cols = 16

        # player data
        self.player_body = [[4, 6], [3, 6], [2, 6]]
        self.player_direction = (1, 0)
        self.score = 0
        self.gameover_state = False

        # apple data
        self.apple_pos = [-33, -33]
        self.eaten = False

        # Solve state
        self.solving = False
        self.previous_move = (0, 1)

        # game window set up, extra row for score board
        self.screen = pygame.display.set_mode((self.cell_cols * self.cell_size, 
                                               (self.cell_rows + 1) * self.cell_size))
        pygame.display.set_caption("Car Game")

        # font setting
        self.font = pygame.font.Font('freesansbold.ttf', 32)


    def cycle(self):
        """ Continuous game cycle to be called each game loop """

        # draw environment
        self.draw_board()

        # draw player
        self.move()
        self.draw_player()

        # draw apple
        self.apple()

        # get player input
        self.events()
        self.solver()
        
        # draw score count
        score = self.font.render(f'score: {self.score}', True, (0, 0, 0), (255, 255, 255))
        scoreRect = score.get_rect()
        scoreRect.center = (3 * self.cell_size, 12.5 * self.cell_size)
        self.screen.blit(score, scoreRect)

        # check player obstacle collision
        self.collision_check()
        
        # update game display
        pygame.time.delay(100)
        pygame.display.update()

        # solve if action performed or game started NOTE: Solver disabled
        # if not self.solving:
        #     self.solver()

    def solver(self):
        """ Neural Network reinforcement learning solver handler """
        
        # control player snake
        # if actions[0] == (0, 1):
        #     self.direction('d')
        # elif actions[0] == (0, -1):
        #     self.direction('a')
        # elif actions[0] == (-1, 0): # reverse y axis direction due to flipped simulated environment
        #     self.direction('w')
        # elif actions[0] == (1, 0):  # reverse y axis direction due to flipped simulated environment
        #     self.direction('s')


    def draw_board(self):
        """ Game environment display with checkerboard pattern """
        self.screen.fill((120, 200, 80))

        for row in range(self.cell_rows):
            for col in range(self.cell_cols):

                # alternate by row and col
                if (row % 2 == 0 and col % 2 != 0) or (row % 2 != 0 and col % 2 == 0):
                    pygame.draw.rect(
                        self.screen, (150, 250, 100), 
                        pygame.Rect(
                            col * self.cell_size, 
                            row * self.cell_size, 
                            self.cell_size, self.cell_size
                        )
                    )


    def draw_player(self):
        """ Player body display with alternating pattern """
        for i, part in enumerate(self.player_body):
        
            # alternate
            if i == 0:
                colour = (150, 0, 150)
            elif i % 2 == 0:
                colour = (0, 0, 180)
            else:
                colour = (0, 0, 100)
            
            pygame.draw.rect(
                    self.screen, colour, 
                    pygame.Rect(
                        part[0] * self.cell_size, 
                        part[1] * self.cell_size, 
                        self.cell_size, self.cell_size
                    )
                )

        self.drawing_player = False


    def events(self):
        """ Keyboard event handler """

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                # allow quitting at all times
                if event.key == pygame.K_SPACE:
                    print('p pressed')
                    pygame.quit()
                    return True
                
                # no movement action when game over
                if not self.gameover_state:
                    if event.key == pygame.K_w:
                        self.direction('w')
                    elif event.key == pygame.K_s:
                        self.direction('s')
                    elif event.key == pygame.K_a:
                        self.direction('a')
                    elif event.key == pygame.K_d:
                        self.direction('d')
                    elif event.key == pygame.K_l:
                        while True:
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_k:
                                        break
                # restart game
                if self.gameover_state:
                    if event.key == pygame.K_f:
                        self.gameover_state = False
                        self.player_body = [[4, 6], [3, 6], [2, 6]]


    def direction(self, direction):
        """ Simple player direction handler """

        # prevent change of direction while player is being drawn
        if not self.drawing_player:
            if direction == 'w' and self.player_direction != (0, 1):
                self.player_direction = (0, -1)
            elif direction == 's' and self.player_direction != (0, -1):
                self.player_direction = (0, 1)
            elif direction == 'a' and self.player_direction != (1, 0):
                self.player_direction = (-1, 0)
            elif direction == 'd' and self.player_direction != (-1, 0):
                self.player_direction = (1, 0)
        
            self.drawing_player = True
    

    def move(self):
        """ Player snake movement and follow handler """

        # shift the player body part pos backwards
        tail_body = self.player_body.copy()
        tail_body = tail_body[:-1]

        # insert new player head position at the start
        tail_body.insert(0, [self.player_body[0][0] + self.player_direction[0], 
                             self.player_body[0][1] + self.player_direction[1]])
        
        self.player_body = tail_body


    def apple(self):
        """ Simple apple creation """
        def valid_player_body(axis):
            """ Validity check for adding body part to player """
            
            # find valid body adding positions in corresponding axis before appending to player body
            if axis == 0:  
                for dir in [-1, 1]:
                    if not ((0 > self.player_body[-1][0] + dir) or 
                        (self.cell_cols - 1 < self.player_body[-1][0] + dir)):
                        self.player_body.append([self.player_body[-1][0] + dir, 
                                                 self.player_body[-1][1]])

            if axis == 1:
                for dir in [-1, 1]:
                    if not ((0 > self.player_body[-1][0] + dir) or 
                        (self.cell_rows - 1 < self.player_body[-1][0] + dir)):
                        self.player_body.append([self.player_body[-1][0], 
                                                 self.player_body[-1][1] + dir])

        
        # if player head and apple overlap and apple not eaten yet: add 1 to score
        if self.apple_pos == self.player_body[0] and not self.eaten:
            self.score += 1
            self.eaten = True

        # check if appending normally will result in body part being out of bounds
        if self.eaten:
            if ((1 > self.player_body[-1][0] and self.player_direction == (1, 0)) or 
                (self.cell_cols - 2 < self.player_body[-1][0] and self.player_direction == (-1, 0))):
                valid_player_body(1)

            elif ((1 > self.player_body[-1][1] and self.player_direction == (0, 1)) or 
                (self.cell_rows - 2 < self.player_body[-1][1]) and self.player_direction == (0, -1)):
                valid_player_body(0)
            
            # appending normally if safe to do so
            else:
                self.player_body.append([self.player_body[-1][0] + self.player_direction[0], 
                                         self.player_body[-1][1] + self.player_direction[1]])

        # if game start or apple is eaten: create new apple
        if self.apple_pos == [-33, -33] or self.eaten:
            self.apple_pos = [randint(0, self.cell_cols - 1), randint(0, self.cell_rows - 1)]
            self.solving = False
            self.eaten = False
        
        pygame.draw.rect(
            self.screen, (180, 0, 0), 
            pygame.Rect(
                self.apple_pos[0] * self.cell_size, 
                self.apple_pos[1] * self.cell_size, 
                32, 32
            )
        )


    def collision_check(self):
        """ Player wall and body collision check """
        
        # player head out of bounds or in other player body parts: gameover
        if ((0 > self.player_body[0][0]) or (self.cell_cols - 1 < self.player_body[0][0]) or 
            (0 > self.player_body[0][1]) or (self.cell_rows -1 < self.player_body[0][1]) or 
            self.player_body[0] in self.player_body[1:]):
            self.score = 0
            self.apple_pos = [-33, -33]
            self.player_direction = (1, 0)
            self.gameover()


    def gameover(self):
        """ Game over state handler """

        self.gameover_state = True

        # freeze game state until restart
        while self.gameover_state:
            self.events()
                

game = CarGame()

# game loop
while True:
    if game.cycle():
        break