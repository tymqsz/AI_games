import pymunk
import pygame
import numpy as np

class Ball():
    def __init__(self, x, y, speed, acceleration, radius=20, mass=10, moment=10):
        self.body = pymunk.Body(mass=mass, moment=moment)
        self.shape = pymunk.Circle(self.body, radius=radius)
        self.shape.elasticity = 1
        self.shape.friction = 0.33
        self.shape.collision_type = 1
        self.radius = radius

        self.speed = speed
        self.acceleration = acceleration
        self.body.position = (x, y)

        self.velocity_len = 0

        Vx = np.random.choice([speed, -speed], 1, p=[0.5, 0.5])
        Vy = np.random.choice(np.linspace(-100, 100, num=200), 1)
        self.body.velocity = (Vx, Vy)
    
    def change_velocity(self, player, Wx, Wy, angle, arbiter, space, data):
        self.velocity_len = (self.body.velocity.x**2 + self.body.velocity.y**2)**(1/2)

        return True
    
    def normalize_velocity(self, arbiter, space, data):
        crt_len = (self.body.velocity.x**2 + self.body.velocity.y**2)**(1/2)

        factor = self.velocity_len/crt_len*self.acceleration

        self.body.velocity = (self.body.velocity.x*factor, self.body.velocity.y*factor)

        return True

    def draw(self, display):
        pygame.draw.circle(display, (255, 255, 255), (int(self.body.position.x), int(self.body.position.y)), self.radius)
    
    def scored(self, width) -> int:
        if self.body.position.x < -100:
            return 1
        elif self.body.position.x > width + 100:
            return 2
        return 0          


class Player():
    def __init__(self, a: tuple, b:tuple, type:int, speed=500, auto=False, mass=10, moment=10):
        self.body = pymunk.Body(mass, moment, body_type=pymunk.Body.KINEMATIC)
        self.shape = pymunk.Segment(self.body, a, b, 5)
        self.shape.collision_type = type
        self.shape.elasticity = 1
        self.shape.friction = 1

        self.speed = speed
        self.auto = auto
    
    def draw(self, display):
        a = (int(self.shape.a[0] + self.body.position.x), int(self.shape.a[1] + self.body.position.y))
        b = (int(self.shape.b[0] + self.body.position.x), int(self.shape.b[1] + self.body.position.y))
        
        pygame.draw.line(display, (255, 255, 255), a, b, width=5)
    
    def in_bounds(self, height):
        if self.body.position.y+450 <= height and self.body.position.y+350 >=0:
            return 3
        elif self.body.position.y+450 <= height:
            return 1
        elif self.body.position.y+350 >=0:
            return 2
    
        return 0
    
    def move(self, keys=[], ball=None, height=800):
        if self.auto:
            rel_pos = (self.body.position.y - ball.body.position.y) + 400

            inbounds = self.in_bounds(height)
            if abs(rel_pos) < 20:
                self.body.velocity = (0, 0)
            elif rel_pos < 0 and (inbounds == 3 or inbounds == 1):
                self.body.velocity = (0, self.speed)
            elif (inbounds == 3 or inbounds == 2):
                self.body.velocity = (0, -self.speed)
        else:
            inbounds = self.in_bounds(height)
            if keys[pygame.K_UP] and (inbounds == 3 or inbounds == 2):
                self.body.velocity = (0, -self.speed)
            elif keys[pygame.K_DOWN] and (inbounds == 3 or inbounds == 1):
                self.body.velocity = (0, self.speed)
            else:
                self.body.velocity = 0, 0


class Wall():
    def __init__(self, a: tuple, b: tuple, type: int):
        self.body = pymunk.Body(10, 10, body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, a, b, 1)
        self.shape.collision_type = type

        self.shape.elasticity = 1
        self.shape.friction = 0
