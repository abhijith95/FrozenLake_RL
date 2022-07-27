import pyglet
from pyglet.window import key
import random
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    UP = 2
    LEFT = 3
    DOWN = 4

class TileColor(Enum):
    # enums can be accessed by their members.
    # for example tileColor['START'] will be equal to the tuple
    START = (105,105,105)
    GOAL = (0,255,0)
    FREEZE = (0,0,255)
    HEAT = (255,0,0)        

class game():
    
    class tile:
        """
        Nested class containing information on the position
        and the status of the tile.
        """
        def __init__(self,id,posX,posY,nature,reward,possibleActions):
            self.id = id
            self.pos = [posX,posY]
            self.nature = nature
            self.tileColor = TileColor[nature].value
            self.reward = reward
            self.actions = possibleActions
    
    def __init__(self,*args,**kwargs):
        # super().__init__(*args,**kwargs)
        self.reset()
    
    def reset(self):
        self.tileRows,self.tileCols = 8,8
        self.tileSize = 20
        self.tilesBatch = pyglet.graphics.Batch() # batch rendering of all tiles
        self.startX,self.startY = 100,300 # starting coordinates for the tiles. This will be the NE corner tile
        self.tiles = []
        self.heatTileProb = 1/6
        self.heatTileReward = -10
        self.freezeTileReward = -1
        self.goalTileReward = 10
        
        self.tileInitialization()
    
    def tileInitialization(self):
        """
        Method to initialize the game tiles. Randomly initializes all the tiles with freeze or heat tiles.
        Then randomly choose a number to assign it with start and goal tile, making sure they are not in the
        same position. Corresponding rewards and action lists are also defined. Since each tile offers unique
        set of possible actions.
        """
        
        def actionInitialization(i,j):
            # define possible actions when at a given tile
            # action list for corner tile
            if i == 0:
                if j == 0:
                    actions = [Direction.RIGHT, Direction.DOWN]
                elif j == self.tileCols-1:
                    actions = [Direction.LEFT, Direction.DOWN]
                else:
                    actions = [Direction.LEFT, Direction.DOWN,Direction.RIGHT]
                    
            elif i == self.tileRows-1:
                if j == 0:
                    actions = [Direction.RIGHT, Direction.UP]
                elif j == self.tileCols-1:
                    actions = [Direction.LEFT, Direction.UP]
                else:
                    actions = [Direction.LEFT, Direction.UP,Direction.RIGHT]
            
            # boundary tiles that are not corner tiles
            if 0<i<self.tileRows-1:
                if j == 0:
                    actions = [Direction.UP,Direction.RIGHT,Direction.DOWN]
                elif j == self.tileCols-1:
                    actions = [Direction.UP,Direction.LEFT,Direction.DOWN]
            
            # inner tiles which has no restrictions
            if (0<i<self.tileRows-1) and (0<j<self.tileCols-1):
                actions = [Direction.UP,Direction.LEFT,Direction.DOWN,Direction.RIGHT]

            return actions
        
        # initializing all the tiles with just heat or freeze 
        ctr = 0
        choiceList = ["HEAT","FREEZE"]
        for i in range(self.tileRows):
            for j in range(self.tileCols):
                choice = np.random.choice(choiceList,1,True,[self.heatTileProb,1-self.heatTileProb])
                if choice[0] == "HEAT":
                    tileReward = self.heatTileReward
                    actions = []
                else:
                    tileReward = self.freezeTileReward
                    actions = actionInitialization(i,j)
                self.tiles.append(self.tile(ctr,self.startX+int(j*1.5*self.tileSize),
                                            self.startY-int(i*1.5*self.tileSize),
                                            choice[0],tileReward,actions)
                                  )
                ctr+=1
        # randomly choosing a tile and changing it to start and goal tile
        self.startIndx = random.randint(0,self.tileCols*self.tileRows-1)
        self.tiles[self.startIndx].nature = "START"
        self.tiles[self.startIndx].tileColor = TileColor["START"].value
        self.tiles[self.startIndx].reward = self.freezeTileReward
        
        self.goalIndx = np.copy(self.startIndx)
        while self.goalIndx==self.startIndx:
            # making sure that start and goal tile are not the same tile
            self.goalIndx = random.randint(0,self.tileCols*self.tileRows-1)
            self.tiles[self.goalIndx].nature = "GOAL"
            self.tiles[self.goalIndx].tileColor = TileColor["GOAL"].value
            self.tiles[self.goalIndx].reward = self.goalTileReward   
            self.tiles[self.goalIndx].actions = []     
    
    def actionMapping(self,s,a):
        """
        Returns s' given s and a. Here s will be the tile ID and a takes a value
        among the Direction Enum.
        """
        if a == Direction.RIGHT:
            sprimeID = s+1
        elif a == Direction.LEFT:
            sprimeID = s-1
        elif a == Direction.UP:
            sprimeID = s-self.tileRows
        elif a == Direction.DOWN:
            sprimeID = s+self.tileRows
        return sprimeID
            
class agent(game):
    
    def __init__(self):
        game.__init__(self)
        self.gamma = 0
        self.V = [0]*self.tileCols*self.tileRows
        self.PI = [Direction.DOWN]*self.tileCols*self.tileRows
        self.valueIteration()
        # initializing the agent's starting position
        self.userPos = [self.tiles[self.startIndx].pos[0],
                        self.tiles[self.startIndx].pos[1]
                        ]
        self.currentStateId = np.copy(self.startIndx)
        self.score = 0
        self.status = "None"
    
    def valueIteration(self):
        """
        Here iterative approach is used to calculate the value of each state. In reality
        if enough iteration is done this value will converge to the true value function.
        Since infinite loop is not possible a threshold is given, which is the absolute
        difference in the value function in successive iteration. An error value is 
        measured which is equal to the maximum value difference of all the states.
        This error should be less than the decided threshold
        """
        def distanceCalc(pos1,pos2):
            # calculate the cartesian distance between two points
            xd = (pos1[0]-pos2[0])**2
            yd = (pos1[1]-pos2[1])**2
            distance = (xd+yd)**(0.5)
            return distance
        
        threshold = 0.001
        error = 1
        while error > threshold:
            # looping through all the tiles
            errorList = []
            for i in range(self.tileCols*self.tileRows):
                v = self.V[i]
                possibleActions = []
                possibleRewards = []
                sprimeID = []
                distanceToGoal = []
                if self.tiles[i].actions:
                    # for goal and heat tiles there are no actions, since they are the terminal tiles
                    maxReward = -1000
                    minDistance = 100000
                    for j in range(len(self.tiles[i].actions)):
                        # looping through all possible actions and storing the rewards in an array
                        possibleActions.append(self.tiles[i].actions[j])
                        nextStateId = int(self.actionMapping(self.tiles[i].id,self.tiles[i].actions[j]))
                        sprimeID.append(nextStateId)
                        possibleRewards.append(self.tiles[nextStateId].reward+ (self.gamma * self.V[nextStateId]))
                        distanceToGoal.append(distanceCalc(self.tiles[self.goalIndx].pos,self.tiles[nextStateId].pos))
                        
                        if possibleRewards[j] > maxReward:
                            maxReward = possibleRewards[j]
                            maxIndx = j
                            minDistance = distanceToGoal[j]
                        elif possibleRewards[j] == maxReward:
                               # if two actions have same reward then choose that which is closest to the goal as next state
                            if distanceToGoal[j] < minDistance:
                                minDistance = distanceToGoal[j]
                                maxIndx = j
                    # if more than one actions have the same reward value then the first one is chosen
                    # maxIndx = possibleRewards.index(max(possibleRewards))
                    self.PI[i] = possibleActions[maxIndx]
                    self.V[i] = possibleRewards[maxIndx] 
                    errorList.append(abs(v-self.V[i]))
                else:
                    self.V[i] = self.tiles[i].reward

            error = max(errorList)
    
    def playingGame(self):
        # iterative loop to play the game. Game is played until goal is reached or agent falls in heat tile
        if self.status !="OVER":
            sprimeId = int(self.actionMapping(self.currentStateId,self.PI[self.currentStateId]))
            if self.tiles[sprimeId].nature == "GOAL":
                print("The agent wins!")
                self.score = 100
                print("The score is ",self.score)
                self.status = "OVER"
            if self.tiles[sprimeId].nature == "HEAT":
                print("The agent lost!")
                self.score = -100
                print("The score is ",self.score)
                self.status = "OVER"
            self.currentStateId = sprimeId
            self.userPos = [self.tiles[sprimeId].pos[0],
                            self.tiles[sprimeId].pos[1]
                            ]

class gameWindow(pyglet.window.Window,agent):
    
    def __init__(self,*args,**kwargs):
        agent.__init__(self)
        pyglet.window.Window.__init__(self,*args,**kwargs)
    
    def on_draw(self):
        self.clear() # since a pyglet window has been inhertied into the class
        # tile = pyglet.shapes.Rectangle(100,200,20,20,(255,0,0))
        # tile.draw()
        circle = pyglet.shapes.Circle(x = self.userPos[0]+(self.tileSize/2), 
                                      y = self.userPos[1]+(self.tileSize/2),
                                      radius = self.tileSize/4)
        
        for i in range(len(self.tiles)):
            tile = pyglet.shapes.Rectangle(x = self.tiles[i].pos[0],y = self.tiles[i].pos[1],
                                           width = self.tileSize, height= self.tileSize,
                                           color = self.tiles[i].tileColor)
            tile.draw()
            valueLabel = pyglet.text.Label(str(round(self.V[i])),font_size=6,
                                           x = self.tiles[i].pos[0]+(self.tileSize/2),
                                           y = self.tiles[i].pos[1]+(self.tileSize/2),
                                           anchor_x='center',anchor_y='center')
            valueLabel.draw()
        circle.draw() # drawing circle later so it comes out on top of the tile
    
    def update(self,dt):
        # pass
        self.playingGame()

game = gameWindow(400,400,caption = "Frozen lake!",resizable = True)
pyglet.clock.schedule_interval(game.update, 1)
pyglet.app.run() # command to execute running the window