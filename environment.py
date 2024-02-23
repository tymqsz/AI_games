import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk

from objects import Ball, Wall, Player

class PongEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, render_mode=None, width=800, height=800):
        self.render_mode = render_mode
        self.FPS = 300
        self.platform_speed = 500
        self.width = width
        self.height = height
        self.ball_acceleration = 1.04
        self.P1_score = 0
        self.P2_score = 0
        self.paused = False
        if self.render_mode is not None:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.Font(None, 50)
            self.score_text = self.font.render(f'{self.P1_score}:{self.P2_score}', True, (255, 255, 255))
            self.display = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        self.space = pymunk.Space()

        self.reward = 0
        #objects
        self.player1 = Player((0, 350), (0, 450), 2, auto=False)
        self.player2 = Player((800, 350), (800, 450), 3, auto=True)

        self.ball = Ball(width//2, height//2, speed=400, acceleration=self.ball_acceleration)

        self.walls = [
            Wall((0, 0), (width, 0), 5),
            Wall((0, height), (width, height), 5),
        ]
        for wall in self.walls:
            self.add_objects([wall.body, wall.shape])

        self.add_objects([self.player1.body, self.player1.shape, self.ball.body,
                          self.player2.body, self.player2.shape, self.ball.shape])

        #collisions
        self.chandler_p1 = self.space.add_collision_handler(1, 2)
        self.chandler_p1.begin = self.touch_reward
        self.chandler_p1.pre_solve = lambda arbiter, space, data: self.ball.change_velocity(self.player1, -1, 1, angle=True,
                                                                                            arbiter=arbiter, space=space, data=data)
        self.chandler_p1.post_solve = lambda arbiter, space, data: self.ball.normalize_velocity(arbiter=arbiter, space=space, data=data)
        self.chandler_p2 = self.space.add_collision_handler(1, 3)
        self.chandler_p2.pre_solve = lambda arbiter, space, data: self.ball.change_velocity(self.player2, -1, 1, angle=True,
                                                                                            arbiter=arbiter, space=space, data=data)
        self.chandler_p2.post_solve = lambda arbiter, space, data: self.ball.normalize_velocity(arbiter=arbiter, space=space, data=data)
        
        self.chandler_hor = self.space.add_collision_handler(1, 5)
        self.chandler_hor.pre_solve = lambda arbiter, space, data: self.ball.change_velocity(None, 1, -1, angle=False,
                                                                                            arbiter=arbiter, space=space, data=data)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.agent = self.player1
    
    def touch_reward(self, arbiter, space, data):
        self.reward += 10
        
        return True
    
    def new_scene(self):
        self.ball.body.position = 400, 400
        Vx = np.random.choice([400, -400], 1, p=[0.5, 0.5])
        Vy = np.random.choice(np.linspace(-100, 100, num=200), 1)
        self.ball.body.velocity = (Vx, Vy)

        self.player1.body.position = 0, 0
        self.player2.body.position = 0, 0
    
    def reset(self, seed=None, options=None):
            
        super().reset(seed=seed)

        self.new_scene()
        self.agent = self.player1
        self.P1_score = 0
        self.P2_score = 0

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation


    def step(self, action):
        self.agent.move(dir=action, ball=self.ball)

        W = 0.01
        distance_to_ball = abs(self.agent.body.position.y-self.ball.body.position.y+400)
        reward = self.reward
            
        print(reward)
        self.reward = 0
        terminated = (self.P1_score == 3 or self.P2_score == 3)

        observation = self._get_obs()
        info = self._get_info()
        self._render_frame()

        return observation, reward, terminated, False, info
    
    def get_reward(self, my_prev, his_prev):
        return self.reward
        
    def _get_obs(self):
        agent_location = [self.agent.body.position.y]
        target_info = [self.ball.body.position.x, self.ball.body.position.y]
                       #self.ball.body.velocity.x, self.ball.body.velocity.y]
        
        obs = np.array(agent_location+target_info)
        return obs
    
    def _get_info(self):
        agent_location = [self.agent.body.position.x, self.agent.body.position.y]
        target_location = [self.ball.body.position.x, self.ball.body.position.y]

        return {
            "distance": 0
        }

    def _render_frame(self):
        scored = self.ball.scored(self.width)
        if scored == 1:
            self.P2_score += 1
            self.new_scene()
        if scored == 2:
            self.P1_score += 1
            #self.reward += 2
            self.new_scene()
        self.space.step(1 / self.FPS)
        
        if self.render_mode is not None:
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
            
            self.score_text = self.font.render(f'{self.P1_score}:{self.P2_score}', True, (255, 255, 255))

            self.player2.move(ball=self.ball)
            
            
            self.display.fill((0, 0, 0))
            self.player1.draw(self.display)
            self.player2.draw(self.display)
            self.ball.draw(self.display)

            self.display.blit(self.score_text, (370, 10))

            pygame.display.update()
            self.clock.tick(self.FPS)
        

    def render(self):
        return 0
    
    def add_objects(self, objects):
        for obj in objects:
            self.space.add(obj)

    def close(self):
        pygame.display.quit()
        pygame.quit()