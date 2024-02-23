import pymunk
import pygame
from objects import Ball, Player, Wall
import time

class Game():
    def __init__(self, height=800, width=800):
        #variables
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, 50)
        self.display = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.FPS = 300
        self.platform_speed = 500
        self.width = width
        self.height = height
        self.ball_acceleration = 1.04
        self.P1_score = 0
        self.P2_score = 0
        self.paused = False
        self.score_text = self.font.render(f'{self.P1_score}:{self.P2_score}', True, (255, 255, 255))
        self.space = pymunk.Space()

        #objects
        self.player1 = Player((0, 350), (0, 450), 2)
        self.player2 = Player((800, 350), (800, 450), 3)

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
    
    def add_objects(self, objects):
        for obj in objects:
            self.space.add(obj)


    def new_scene(self):
        self.space.remove(self.ball.body, self.ball.shape)
        self.space.remove(self.player1.body, self.player1.shape)
        self.space.remove(self.player2.body, self.player2.shape)

        self.ball = Ball(self.height//2, self.width//2, speed=400, acceleration=self.ball_acceleration)
        self.player1 = Player((0, 350), (0, 450), 2)
        self.player2 = Player((800, 350), (800, 450), 3, auto=True)

        self.add_objects([self.player1.body, self.player1.shape, self.ball.body,
                          self.player2.body, self.player2.shape, self.ball.shape])

    def run(self):
        while True:
            if self.paused:
                pass
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            scored = self.ball.scored(self.width)
            if scored == 1:
                self.P2_score += 1
                self.new_scene()
            if scored == 2:
                self.P1_score += 1
                self.new_scene()
            self.score_text = self.font.render(f'{self.P1_score}:{self.P2_score}', True, (255, 255, 255))

            keys = pygame.key.get_pressed()
            self.player1.move(keys=keys, height=self.height)
            self.player2.move(ball=self.ball)
            
            
            self.display.fill((0, 0, 0))
            self.player1.draw(self.display)
            self.player2.draw(self.display)
            self.ball.draw(self.display)
            self.display.blit(self.score_text, (370, 10))

            pygame.display.update()
            self.clock.tick(self.FPS)
            self.space.step(1 / self.FPS)

    def get_state(self):
        my_x, my_y = self.player1.body.position
        his_x, his_y = self.player2.body.position

        ball_x, ball_y = self.ball.body.position
        ball_Vx, ball_Vy = self.ball.body.velocity

        return my_x, my_y, his_x, his_y, ball_x, ball_y, ball_Vx, ball_Vy
    
    def get_reward(self, my_prev, his_prev):
        prev_diff = my_prev-his_prev
        crt_diff = self.P1_score - self.P2_score

        if crt_diff == prev_diff:
            return 0 
        if crt_diff > prev_diff:
            return -1
        else:
            return 1
            



game = Game()
game.run()
