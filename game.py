import pygame
import random
from random import randint, sample
import numpy as np
import matplotlib.pyplot as plt
import math

import torch

from collections import deque

from model import RLNet, RLTrainer

# game loop
class SnakeGame:
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
        self.drawing_player = False
        self.score = 0
        self.record = 0
        self.gameover_state = False

        # apple data
        self.apple_pos = [-33, -33]
        self.eaten = False
        self.apple_reward = False

        # Solve state
        self.solving = False
        self.previous_move = (0, 1)

        # game window set up, extra row for score board
        self.screen = pygame.display.set_mode((self.cell_cols * self.cell_size, 
                                               (self.cell_rows + 1) * self.cell_size))
        pygame.display.set_caption("Car Game")

        # font setting
        self.font = pygame.font.Font('freesansbold.ttf', 16)

        # model init
        self.model = RLNet()
        self.trainer = RLTrainer(self.model)

        self.memory = deque(maxlen=100000)
        self.n_games = 0
        self.batch_size = 1000
        self.reward = 0
        self.move_count = 0
        self.scores = []
        self.avg_scores = []

        # plot init
        plt.ion()
        plt.figure(figsize=(6, 3))
        plt.tight_layout()


    def cycle(self):
        """ Continuous game cycle to be called each game loop """

        # draw environment
        self.draw_board()

        # draw player
        self.move()
        self.draw_player()

        # get player input
        self.events()

        # draw apple
        self.apple()
        
        # draw score count (with record and game count)
        score = self.font.render(f'score: {self.score}  record: {self.record}  n_games: {self.n_games}', True, (0, 0, 0), (255, 255, 255))
        scoreRect = score.get_rect(left=self.cell_size, top=(12 * self.cell_size))
        self.screen.blit(score, scoreRect)

        # check player obstacle collision
        self.collision_check()
        
        # update game display (tunable for training speed)
        pygame.time.delay(20)
        pygame.display.update()

    def solver(self):
        """ Neural Network reinforcement learning solver handler """        

        # ==============================
        # 1. Get the current/initial state
        # ==============================

        state_initial = self.get_state()

        # ==============================
        # 2. Get the next action via exploitation vs exploration
        # ==============================

        self.epsilon = 80 - self.n_games
        chosen_action = [0, 0, 0]
        
        # exploration (random)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            chosen_action[move] = 1
        # exploitation (model prediction)
        else:
            state0 = torch.tensor(state_initial, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            chosen_action[move] = 1

        # ==============================
        # 3. Perform the next action
        # ==============================

        if chosen_action == [1, 0, 0]:  # turn left
            if self.player_direction == (0, -1):
                self.direction('a')
            elif self.player_direction == (0, 1):
                self.direction('d')
            elif self.player_direction == (-1, 0):
                self.direction('s')
            elif self.player_direction == (1, 0):
                self.direction('w')
        elif chosen_action == [0, 1, 0]:  # turn right
            if self.player_direction == (0, -1):
                self.direction('d')
            elif self.player_direction == (0, 1):
                self.direction('a')
            elif self.player_direction == (-1, 0):
                self.direction('w')
            elif self.player_direction == (1, 0):
                self.direction('s')
        
        # do nothing if go straight

        # cycle the game to get next state with chosen action
        self.cycle()
        self.move_count += 1

        # ==============================
        # 4. Get reward/penalty of next state/chosen action (score and gameover state provided through game)
        # ==============================

        # NOTE: can be rewards or penalty, will affect design of downstream processes
        reward_obs = -10       # out of bounds penalty
        reward_apple = 10      # reach apple reward
        reward_step = 0        # action taking penalty
        
        done = False

        # NOTE: reward/penalty system design determined by how rewards/penalty are used

        # prevent repetitive motion by move limit gameover
        if self.gameover_state or self.move_count > 192:
            self.reward += reward_obs
            self.move_count = 0
            done = True
        elif self.apple_reward:
            self.reward += reward_apple
            self.apple_reward = False
            self.move_count = 0
        else:
            self.reward += reward_step

        # ==============================
        # 5. Get new state from chosen action
        # ==============================

        state_new = self.get_state()

        # ==============================
        # 6. Train on current (initial state, chosen_action, self.reward, state_new, done) information
        # ==============================

        self.trainer.train_step(state_initial, chosen_action, self.reward, state_new, done)

        # ==============================
        # 7. Remember/store current game information for bulk training
        # ==============================

        self.memory.append((state_initial, chosen_action, self.reward, state_new, done))  # pops left when maxlen is reached

        # ==============================
        # 8. Train on total collected game information memory in an n_game
        # ==============================
        
        if done:
            # train on self.memory
            if len(self.memory) > self.batch_size:
                mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
            else:
                mini_sample = self.memory

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

            # save best model
            if self.score > self.record:
                self.record = self.score
                self.model.save()

            # debug/tracking info
            # print('Game', self.n_games, 'Score', self.score, 'Record:', self.record)

            # reset the game (gameover)
            self.gameover_state = False
            self.player_body = [[4, 6], [3, 6], [2, 6]]
            self.n_games += 1
            self.scores.append(self.score)
            self.avg_scores.append(np.mean(self.scores))
            self.score = 0

            # print(self.scores, list(range(0, self.n_games)))
            plt.ylim(ymin=0)
            plt.title('Model Performance over n games')
            plt.xlabel('n games')
            plt.ylabel('Score')
            plt.plot(self.scores, color='blue')
            plt.plot(self.avg_scores, color='red')
            plt.show(block=False)

        # clear previous reward for next state/action
        self.reward = 0       


    def get_state(self):
        """ Get game state information for model training """
        # danger based on global snake head pos (no player direction)
        global_danger_left = ((0 > (self.player_body[0][0] - 1)) or 
                              (([self.player_body[0][0] - 1, self.player_body[0][1]]) in self.player_body) or 
                              ((self.player_body[0][0] - 1) in self.player_body[1:]))

        global_danger_right = (((self.cell_cols - 1) < (self.player_body[0][0] + 1)) or 
                               (([self.player_body[0][0] + 1, self.player_body[0][1]]) in self.player_body) or 
                               ((self.player_body[0][0] + 1) in self.player_body[1:]))

        global_danger_up = ((0 > (self.player_body[0][1] - 1)) or 
                            (([self.player_body[0][0], self.player_body[0][1] - 1]) in self.player_body) or 
                            ((self.player_body[0][1] - 1) in self.player_body[1:]))

        global_danger_down = (((self.cell_rows - 1) < (self.player_body[0][1] + 1)) or 
                              (([self.player_body[0][0], self.player_body[0][1] + 1]) in self.player_body) or 
                              ((self.player_body[0][1] + 1) in self.player_body[1:]))
        
        # danger based on relative snake head pos (considers player direction)
        if self.player_direction == (1, 0):
            self.danger_left = global_danger_up
            self.danger_right = global_danger_down
            self.danger_straight = global_danger_right

        elif self.player_direction == (-1, 0):
            self.danger_left = global_danger_down
            self.danger_right = global_danger_up
            self.danger_straight = global_danger_left

        elif self.player_direction == (0, 1):
            self.danger_left = global_danger_right
            self.danger_right = global_danger_left
            self.danger_straight = global_danger_down

        elif self.player_direction == (0, -1):
            self.danger_left = global_danger_left
            self.danger_right = global_danger_right
            self.danger_straight = global_danger_down

        # binary player facing direction info
        self.dir_left = (self.player_direction == (-1, 0))
        self.dir_right = (self.player_direction == (1, 0))
        self.dir_up = (self.player_direction == (0, -1))
        self.dir_down = (self.player_direction == (0, 1))

        # apple direction info 
        self.apple_left = (self.player_body[0][0] > self.apple_pos[0])
        self.apple_right = (self.player_body[0][0] < self.apple_pos[0])
        self.apple_up = (self.player_body[0][1] > self.apple_pos[1])
        self.apple_down = (self.player_body[0][1] < self.apple_pos[1])
        
        state = [
            self.danger_left,
            self.danger_right,
            self.danger_straight,

            self.dir_left,
            self.dir_right,
            self.dir_up,
            self.dir_down,

            self.apple_left,
            self.apple_right,
            self.apple_up,
            self.apple_down
        ]
        
        return np.array(state, dtype=int)
    

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
                    print('space pressed')
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
                    # debug freeze state
                    # elif event.key == pygame.K_l:
                    #     while True:
                    #         for event in pygame.event.get():
                    #             if event.type == pygame.KEYDOWN:
                    #                 if event.key == pygame.K_k:
                    #                     break

                # restart game (auto restart)
                # if self.gameover_state:
                #     if event.key == pygame.K_f:
                #         self.gameover_state = False
                #         self.player_body = [[4, 6], [3, 6], [2, 6]]


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
            self.apple_reward = True

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
            (0 > self.player_body[0][1]) or (self.cell_rows - 1 < self.player_body[0][1]) or 
            self.player_body[0] in self.player_body[1:]):
            # self.score = 0
            self.apple_pos = [-33, -33]
            self.player_direction = (1, 0)
            self.gameover()


    def gameover(self):
        """ Game over state handler """

        self.gameover_state = True

        # freeze game state until restart
        # while self.gameover_state:
        #     self.events()
                

game = SnakeGame()

# game loop
while True:
    if game.solver():
        break