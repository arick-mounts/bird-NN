
##
##import pygame, tensor flow, and keras
##



import sys
import pygame
from random import randint
from DQN import DQNAgent
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

width = 450
height = 380
pygame.font.init()

class Game:
    def __init__(self, game_width, game_height):
        pygame.display.set_caption('BirdGame')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.upper_pipe = Pipe(game_width-1, randint(-100,80))     
        
        self.lower_pipe = Pipe(game_width-1, self.upper_pipe.piperect.bottom + 230)
        self.player = Player(self)
        self.score = 0
    
    def game_display(self):
        self.gameDisplay.fill((255,255,255))
        
        
class Player(object):
    def __init__(self, game):
        self.x= 0.5 * game.game_width
        self.y = 0.5 * game.game_height
        self.position =[]
        self.position.append([self.x,self.y])
        self.image = pygame.image.load("img/birb.jpg")
        self.playerrect = self.image.get_rect(center = (self.x,self.y)) 
        self.speed = [0,-8]
    
    def Move(self, game, move):
        self.playerrect= self.playerrect.move(self.speed)
        if self.playerrect.top < 0+20 or self.playerrect.bottom > height-80:
            game.crash = True
        elif self.speed[1] < 12:
            self.speed[1] += 1 
            
        #print (move)
        if np.array_equal(move ,[0,1]):
            self.Jump()
            
        
        
    def Jump(self):
        self.speed[1] = -5
    
    def display_player(self, game):
        game.gameDisplay.blit(self.image,self.playerrect)


class Pipe:
    def __init__(self, x, y, ):
        self.x_pipe = x
        self.y_pipe = y
        self.pipe_speed = -5
        self.image = pygame.image.load("img/pipe.png")   
        self.piperect = self.image.get_rect(center = (self.x_pipe, self.y_pipe))
        self.has_scored = False
    
    
    def display_pipe(self, game):
        game.gameDisplay.blit(self.image, self.piperect)
    
    
def get_record(score, record):
        if score >= record:
            return score
        else:
            return record
        
def move_pipes(pipeU,pipeD, height_mod):
    pipeU.piperect.move_ip(pipeU.pipe_speed, 0)
    pipeD.piperect.move_ip(pipeD.pipe_speed, 0)
    
    if pipeU.piperect.right < 0:
           pipeU.piperect.left = width  + 35
           pipeU.piperect.bottom = randint(64, height-height_mod)
           pipeD.piperect.left = width +35
           pipeD.piperect.top = pipeU.piperect.bottom + 110
    

        
def check_collision(player,pipe1,pipe2,game):
    if player.playerrect.colliderect(pipe1.piperect) or player.playerrect.colliderect(pipe2.piperect) :
            game.crash =True
            
def check_score(pipeU,game):
    if  pipeU.piperect.left == 225 and pipeU.has_scored == False:
        game.score = game.score + 1
        pipeU.has_scored = True
    if pipeU.piperect.left < 225:    
           pipeU.has_scored = False
           
           
    

def display(game, player, pipeU, pipeD, record):
    game.game_display()
    game.gameDisplay.blit(game.bg, (10, 10))
    pipeU.display_pipe(game)
    pipeD.display_pipe(game)
    display_ui(game, game.score, record)
    player.display_player(game)
    pygame.display.update()

def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (15, 340))
    game.gameDisplay.blit(text_score_number, (120, 340))
    game.gameDisplay.blit(text_highest, (190, 340))
    game.gameDisplay.blit(text_highest_number, (350, 340))
    
    
def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()
    
def check_pipes(pipeU,pipeD, player):
    if player.playerrect.top > pipeU.piperect.bottom+20 and player.playerrect.bottom < pipeD.piperect.top-20:
        return True
    return False
    
def run():
    pygame.init()
    
    
    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot =[]
    record = 0
    
    while counter_games < 60:
        game = Game(width, height)
        pipeU = game.upper_pipe
        pipeD = game.lower_pipe
        player = game.player
        between_pipes = False
        while not game.crash:
            
            agent.epsilon = 20 - counter_games
            
            state_old = agent.get_state(game, player, pipeU, pipeD)
            
            if randint(0, 200) < agent.epsilon:
                final_move = to_categorical(randint(0, 1), num_classes=2)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1,8)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=2)
              
            player.Move(game,final_move)
            move_pipes(pipeU,pipeD,160)
            check_collision(player,pipeU,pipeD,game)
            
            check_score(pipeU,game)
            between_pipes = check_pipes(pipeU, pipeD, player)
            
            reward = agent.set_reward(pipeU, game.crash, between_pipes)
            
            state_new = agent.get_state(game, player, pipeU, pipeD)
            
            
            agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
            
             # store the new data into a long term memory
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            record = get_record(game.score, record)
            
         
            
            display(game,player,pipeU,pipeD,record)
            if game.score == 50:
                game.crash = True;
        agent.replay_new(agent.memory)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.score)
        score_plot.append(game.score)
        counter_plot.append(counter_games)
        
    agent.model.save_weights('weights.hdf5')
    plot_seaborn(counter_plot, score_plot)
    pygame.quit()
    
run()
