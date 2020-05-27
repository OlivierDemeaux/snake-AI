import pygame
import random
import numpy as np
from copy import deepcopy
import math
from numpy import genfromtxt

direction = 'right'
available = []
gameOver = False
inputs = []

class Ai:
	def __init__(self, layers):
		self.layers = layers
		self.weights = self.initWeights()
		self.maxSteps  = 100
		self.stepSinceLastApple = 0
		self.steps = 0
		self.applesEaten = 0
		self.fitness = 0

	def initWeights(self):
		theta_1 = np.random.uniform(-1, 1, [self.layers[1], self.layers[0]])
		theta_2 = np.random.uniform(-1, 1, (self.layers[2], self.layers[1]))
		theta_3 = np.random.uniform(-1, 1, (self.layers[3], self.layers[2]))
		weights = []
		weights.append(theta_1)
		weights.append(theta_2)
		weights.append(theta_3)
		return (weights)

	def mutate(self):
		for i in range(len(self.weights)):
			for j in range(len(self.weights[i])):
				for k in range(len(self.weights[i][j])):
					if (random.randint(1, 100) <=  1):
						x = self.weights[i][j][k] + random.uniform(-1, 1)
						x = max(min(x, 1), -1)
						self.weights[i][j][k] =  x
	
	def reset(self):
		self.stepSinceLastApple = 0
		self.steps = 0
		self.applesEaten = 0
		self.fitness = 0

class Game:
	def __init__(self):
		self.DIM = 200
		self.itemSize = 20
		self.ratio = int(self.DIM / self.itemSize)
		self.snake = Snake(self)
		self.apple = Apple(self)
		self.apple.getNewPosition(self)
		self.gameOver = False
		self.points = 0

		self.WIN = pygame.display.set_mode((self.DIM, self.DIM))

	def draw(self):
		self.WIN.fill([0, 0, 0])
		pygame.draw.rect(self.WIN, [0, 255, 0], [self.apple.apple[0], self.apple.apple[1], self.itemSize, self.itemSize])
		for item in self.snake.snake:
			pygame.draw.rect(self.WIN, [255, 255, 255], [item[0], item[1], self.itemSize, self.itemSize])

	def collision(self):
		head = self.snake.snake[0]
		if (head in self.snake.snake[1:]):
			self.gameOver = True
			print('snake')
		if (head[0] < 0 or head[0] >= self.DIM or head[1] < 0  or head[1] >= self.DIM):
			self.gameOver =  True
			print('wall')
		if (head == self.apple.apple):
			self.points += 1
			self.snake.snake.append(self.snake.ghost)
			self.apple.getNewPosition(self)

class Apple:
	def __init__(self, game):
		self.size = game.itemSize
		self.apple = [0, 0]

	def getNewPosition(self, game):
		freePositions =  []
		for x in range(game.ratio):
			for y in range(game.ratio):
				position =  [x * game.itemSize, y * game.itemSize]
				if (position not in game.snake.snake):
					freePositions.append(position)
		number = random.randint(0, len(freePositions) - 1)
		self.apple  = freePositions[number]

class Snake:
	def __init__(self, game):
		self.direction = 'right'
		self.snake = [[int(game.DIM / 2), int(game.DIM / 2)], [int(game.DIM / 2) - game.itemSize, int(game.DIM / 2)], [int(game.DIM / 2) - 2 * game.itemSize, int(game.DIM / 2)] ]
		self.size = game.itemSize
		self.ghost = self.snake[0]

	def move(self):
		head = self.snake[0]
		new = []
		if (self.direction == 'left'):
			new = [head[0] -self.size, head[1]]
		elif (self.direction == 'right'):
			new = [head[0] + self.size, head[1]]
		elif (self.direction == 'up'):
			new = [head[0], head[1] - self.size]
		elif (self.direction == 'down'):
			new = [head[0], head[1] + self.size]
		self.snake.insert(0, new)
		self.ghost = self.snake.pop()


	def draw(self, game):
		if (game.snake.snake[0][0] >= game.DIM or game.snake.snake[0][0] <= 0 or game.snake.snake[0][1] >= game.DIM or game.snake.snake[0][1] <= 0):
			game.gameOver = True
		for item in game.snake.snake:
			pygame.draw.rect(game.WIN, (255, 255, 255), (item[0], item[1], 20, 20))

		# Draw eyes
		# pygame.draw.circle(WIN, (0,0,0), (snake[0][0] + 5, snake[0][1] + 5), 3)
		# pygame.draw.circle(WIN, (0,0,0), (snake[0][0] + 15, snake[0][1] + 5), 3)


def draw_window(game):
	game.WIN.fill((0, 0, 0))
	game.draw()

	pygame.display.update()

def getInputsAI(game):
	head = game.snake.snake[0]
	inputs = []

	views = [
		[0, -1],
		[1, -1],
		[1, 0],
		[1, 1],
		[0, 1],
		[-1, 1],
		[-1, 0],
		[-1, -1]
	]
	for view in views:
		tmp = head.copy()
		distanceToWall = 0
		distanceToSnake = math.inf
		distanceToApple = math.inf
		while (tmp[0] >= 0 and tmp[0] < game.DIM and tmp[1] >= 0 and tmp[1] < game.DIM):
			tmp[0] += view[0] * 20
			tmp[1] += view[1] * 20
			distanceToWall += 1
			if (distanceToSnake == math.inf and tmp in game.snake.snake):
				distanceToSnake = distanceToWall
			elif (distanceToApple == math.inf and tmp == game.apple.apple):
				distanceToApple = distanceToWall
		inputs += [(1 / distanceToWall) , (1 / distanceToSnake) , (1 / distanceToApple)]
	return inputs

def  checkDir(game):
	dir = []
	dirTail = []
	if (direction == 'up'):
		dir = [1, 0, 0, 0]
	if (direction == 'down'):
		dir = [0, 1, 0, 0]
	if (direction  == 'left'):
		dir = [0, 0, 1, 0]
	if (direction ==  'right'):
		dir = [0, 0, 0, 1]
	if (len(game.snake.snake) > 1):
		posOneBeforeLast = game.snake.snake[-1]
		posLast = game.snake.snake[-2]
		if (posLast[0] - posOneBeforeLast[0] > 0):
			dirTail = [0, 0, 0, 1]
		if (posLast[0] - posOneBeforeLast[0] < 0):
			dirTail = [0, 0, 1, 0]
		if (posLast[1] - posOneBeforeLast[1] > 0):
			dirTail = [0, 1, 0, 0]
		if (posLast[1] - posOneBeforeLast[1] < 0):
			dirTail = [1, 0, 0, 0]
	else:
		dirTail = dir

	return (dir + dirTail)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def predict(inputs, theta_1, theta_2, theta_3):

	z2 = np.dot(inputs, theta_1.T)
	a2 = np.maximum(z2, 0, z2)

	z3 = np.dot(a2, theta_2.T)
	a3 = np.maximum(z3, 0, z3)

	z4 = np.dot(a3, theta_3.T)
	a4 = sigmoid(z4)

	maximun = max(a4)
	index = np.where(a4 == maximun)
	realIndex = index[0][0] + 1
	return (realIndex)

def neuralNetwork(data, weights, game):
	theta_1 = weights[0]
	theta_2 = weights[1]
	theta_3 = weights[2]

	index = predict(data, theta_1, theta_2, theta_3)
	if (index == 1 and game.snake.direction != 'down'):
		return('up')
	elif (index == 2 and game.snake.direction != 'up'):
		return ('down')
	elif (index == 3 and game.snake.direction != 'right'):
		return ('left')
	elif (index == 4 and game.snake.direction != 'left'):
		return ('right')
	else:
		return(game.snake.direction)


def main():

	stop = False
	global direction

	ais = []
	for _ in range(1):
		ais.append(Ai([32, 20, 12, 4]))

	weight1 = genfromtxt('data0.csv', delimiter=',')
	weight2 = genfromtxt('data1.csv', delimiter=',')
	weight3 = genfromtxt('data2.csv', delimiter=',')

	while (not stop):
		for ai in ais:
			ai.reset()
			del ai.weights[:]
			ai.weights.append(weight1)
			ai.weights.append(weight2)
			ai.weights.append(weight3)
			game = Game()
			run = True
			fps = 30
			clock = pygame.time.Clock()
			while run and not game.gameOver:

				clock.tick(fps)
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						run = False
						quit()
						break

				draw_window(game)
				inputs = getInputsAI(game)
				directionInputs = checkDir(game)
				data = inputs + directionInputs
				game.snake.direction = neuralNetwork(data, ai.weights, game)

				game.snake.move()

				ai.steps += 1
				ai.stepSinceLastApple += 1

				if (game.snake.snake[0] == game.apple.apple):
					ai.stepSinceLastApple = 0
					ai.applesEaten += 1
				if (ai.stepSinceLastApple > ai.maxSteps):
					game.gameOver = True
					print('steps')

				game.collision()
			print(ai.applesEaten)


		# 	ai.fitness = ai.steps + (2 ** ai.applesEaten + ai.applesEaten ** 2.1 * 500) - (ai.applesEaten ** 1.2 * (0.25 *  ai.steps) ** 1.3)

		# fitnessSum = getFitnessSum(ais)
		# children = []
		# for _ in range(len(ais) - 1):
		# 	parentA = selectParent(ais, fitnessSum)
		# 	parentB = selectParent(ais, fitnessSum)
		# 	child = deepcopy(parentA)
		# 	child.weights =  mixWeights(parentA.weights, parentB.weights)
		# 	if (random.randint(1, 100) <= 20):
		# 		child.mutate()
		# 	children.append(child)
		# ais.sort(key=lambda x: x.fitness, reverse=True)
		# print('Gen: {}, Fit: {}, Apple: {}, Steps: {}'.format(generation, ais[0].fitness, ais[0].applesEaten, ais[0].steps))
		# if (generation % 50 == 0):
		# 	with open('bestWeights.txt', 'w') as f:
		# 		for item in ais[0].weights:
		# 			f.write("%s\n" % item)
		# ais = [ais[0]] + children
		# generation += 1




if __name__ == '__main__':
	main()
