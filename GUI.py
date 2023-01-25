import sys, pygame, math
from pygame.locals import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import os
import csv

class draw_item:
    def __init__(self):
        self.surface = None
        self.left = 0
        self.top = 0

    def add(self,surface,left,top):
        self.surface = surface
        self.left = left
        self.top = top


class pintura:

    def __init__(self):
        pygame.init()
        
        self.BLACK = 0,0,0
        self.WHITE = 255,255,255
        self.GREY1 = 100,100,100
        self.GREEN = 0,255,0
        self.BLUE = 255,0,0
        self.RED = 0,0,255
        os.chdir("/Users/jaimevarelap/Documents/ITAM/10/Aprendizaje Profundo/Kandinsky/")
        with open('./paleta_rgb.csv') as f:
            self.PALETA=[list(int(num) for num in line) for line in csv.reader(f)]
        self.COLOR_ACTIVO = self.PALETA[0]
        self.QUIT = False
        self.mousebutton = None
        self.mousedown = False
        self.toolset = ["Line","Circle","Curve","Fill","Pick"]
        self.mouse_buttons = ["Left Button","Middle Button","Right Button","Wheel Up","Wheel Down"]
        self.draw_list = []
        self.mouseX = self.mouseY = 0
        self.draw_tool = "Line"
        self.drawstartX = -1
        self.drawendX = -1
        self.drawstartX = -1
        self.drawendY = -1
        self.draw_toggle = False

        self.PointList = []
        self.lastDraw = 0
        self.prediction = np.zeros((512,512,3))
        os.chdir("/Users/jaimevarelap/Documents/ITAM/10/Aprendizaje Profundo/Kandinsky/modelos")
        self.model = tf.keras.models.load_model('modelo24',compile = False)
        #initialize system
        self.initialize()

    def palette(self,clusters):
        width=1024
        palette = np.zeros((width, 128, 3), np.uint8)
        steps = width/len(clusters)
        for idx, centers in enumerate(clusters): 
            palette[int(idx*steps):(int((idx+1)*steps)), : , :] = centers
        return palette

        
    def initialize(self):
        
        #Setup the pygame screen
        self.palette_height = 128
        self.screen_width = 1024
        self.screen_height = 512+self.palette_height
        self.screen_size = (self.screen_width, self.screen_height)    
        self.screen = pygame.display.set_mode(self.screen_size)

        #setup a generic drawing surface/canvas with
        self.canvas = pygame.Surface((self.screen_width/2, self.screen_height-self.palette_height))
        self.img_canvas = pygame.Surface((self.screen_width/2, self.screen_height-self.palette_height))
        self.palette_canvas = pygame.Surface((self.screen_width, self.palette_height))

        #setup a work canvas with black as the transparent colour
        self.work_canvas = pygame.Surface((self.screen_width/2, self.screen_height-self.palette_height))
        self.work_canvas.set_colorkey(self.BLACK)

        #setup a paint canvas
        self.paint_canvas = pygame.Surface((self.screen_width/2, self.screen_height-self.palette_height))

    def radius(self,rectangle):
        x1,y1,x2,y2 = rectangle
        x = (x2-x1)
        y = (y2-y1)
        rad = math.sqrt(x**2+y**2)
        if rad < 3:
            rad = 3
        return rad/2

    def center(self,rectangle):
        x1,y1,x2,y2 = rectangle
        x = abs(x1-x2)
        y = abs(y1-y2)
        #x1 += x/2
        #y1 += y/2
        return (x1,y1)

    def mouse_handler(self,events):

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mousedown = True
                self.mousebutton = event.button  
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mousedown = False
                self.mousebutton = event.button
            self.mouseX, self.mouseY = pygame.mouse.get_pos()

        #manage tool events
        if self.draw_tool == "Line":
            self.draw_line_template()
        if self.draw_tool == "Circle":
            self.draw_circle_template()
        if self.draw_tool == "Curve":
            self.draw_curve_template()
        if self.draw_tool == "Fill":
            self.fill_template()
        if self.draw_tool == "Pick":
            self.pick_template()
        #show mouse state
        self.show_mousestate()

    def pick_template(self):
        if self.mousedown and self.mousebutton == 1:
            self.drawstartX = self.mouseX
            self.drawstartY = self.mouseY
            self.COLOR_ACTIVO = self.screen.get_at((self.drawstartX,self.drawstartY))


    def draw_curve_template(self):
        #on left mouse down we set the basic values
        if self.draw_toggle == False and self.mousedown and self.mousebutton == 1:
            self.drawstartX = self.mouseX
            self.drawendX = self.mouseX
            self.drawstartY = self.mouseY
            self.drawendY = self.mouseY
            self.draw_toggle = True

        #while mouse is down we draw the circle template to the work canvas
        elif self.draw_toggle == True and self.mousedown and self.mousebutton == 1:
            self.drawendX = self.mouseX
            self.drawendY = self.mouseY
            #We clear the work canvas with black
            self.work_canvas.fill(self.BLACK)
            #We draw a circle into the work_canvas in GREY1
            try:
                self.PointList.append((self.drawendX,self.drawendY))
                for point in self.PointList:
                    pygame.draw.circle(self.work_canvas,
                                (self.COLOR_ACTIVO),
                                point,
                                2,
                                0)
            except:  #we cant have a thickness larger than the radius, so catch the exception
                pass
            #blit the work_canvas onto the canvas
            self.draw_tool_template()

        #when we release the left mouse, we draw a white circle to the paint canvas
        elif self.draw_toggle == True and not self.mousedown and self.mousebutton == 1:
            self.draw_toggle = False
            #We draw a circle into the paint_canvas in WHITE
            try:
               for point in self.PointList:
                    pygame.draw.circle(self.paint_canvas,
                                (self.COLOR_ACTIVO),
                                point,
                                2,
                                0)
            except:  #we cant have a thickness larger than the radius, so catch the exception
                pass
            self.PointList = []

    def draw_circle_template(self):
        #on left mouse down we set the basic values
        if self.draw_toggle == False and self.mousedown and self.mousebutton == 1:
            self.drawstartX = self.mouseX
            self.drawendX = self.mouseX
            self.drawstartY = self.mouseY
            self.drawendY = self.mouseY
            self.draw_toggle = True

        #while mouse is down we draw the circle template to the work canvas
        elif self.draw_toggle == True and self.mousedown and self.mousebutton == 1:
            self.drawendX = self.mouseX
            self.drawendY = self.mouseY
            #We clear the work canvas with black
            self.work_canvas.fill(self.BLACK)

            #We draw a circle into the work_canvas in GREY1
            try:
               pygame.draw.circle(self.work_canvas,
                             (self.COLOR_ACTIVO),
                             self.center((self.drawstartX,self.drawstartY,self.drawendX,self.drawendY)),
                             self.radius((self.drawstartX,self.drawstartY,self.drawendX,self.drawendY)),
                             5)
            except:  #we cant have a thickness larger than the radius, so catch the exception
                pass
            #blit the work_canvas onto the canvas
            self.draw_tool_template()

        #when we release the left mouse, we draw a white circle to the paint canvas
        elif self.draw_toggle == True and not self.mousedown and self.mousebutton == 1:
            self.draw_toggle = False
            #We draw a circle into the paint_canvas in WHITE
            try:
               pygame.draw.circle(self.paint_canvas,
                             (self.COLOR_ACTIVO),
                             self.center((self.drawstartX,self.drawstartY,self.drawendX,self.drawendY)),
                             self.radius((self.drawstartX,self.drawstartY,self.drawendX,self.drawendY)),
                             5)
            except:  #we cant have a thickness larger than the radius, so catch the exception
                pass

    def fill_template(self):
        #on left mouse down we set the basic values
        if self.draw_toggle == False and self.mousedown and self.mousebutton == 1:
            self.drawstartX = self.mouseX
            self.drawendX = self.mouseX
            self.drawstartY = self.mouseY
            self.drawendY = self.mouseY
            self.draw_toggle = True

        #while mouse is down we draw the circle template to the work canvas
        elif self.draw_toggle == True and self.mousedown and self.mousebutton == 1:
            self.drawendX = self.mouseX
            self.drawendY = self.mouseY
            #We clear the work canvas with black
            self.work_canvas.fill(self.BLACK)

            #We draw a circle into the work_canvas in GREY1
            try:
                work_arr = pygame.surfarray.array3d(self.canvas)

                cv.floodFill(work_arr,None,(self.drawstartY,self.drawstartX),self.COLOR_ACTIVO)
                pygame.surfarray.blit_array(self.work_canvas, work_arr)
            except:  #we cant have a thickness larger than the radius, so catch the exception
                pass
            #blit the work_canvas onto the canvas
            self.draw_tool_template()

            #when we release the left mouse, we draw a white circle to the paint canvas
        elif self.draw_toggle == True and not self.mousedown and self.mousebutton == 1:
            self.draw_toggle = False
            #We draw a circle into the paint_canvas in WHITE
            try:
                paint_arr = pygame.surfarray.array3d(self.canvas)

                cv.floodFill(paint_arr,None,(self.drawendY,self.drawendX),self.COLOR_ACTIVO)
                pygame.surfarray.blit_array(self.paint_canvas, paint_arr)
            except:  #we cant have a thickness larger than the radius, so catch the exception
                pass
        
    def draw_line_template(self):

        #on left mouse down we set the basic values
        if self.draw_toggle == False and self.mousedown and self.mousebutton == 1:
            self.drawstartX = self.mouseX
            self.drawendX = self.mouseX
            self.drawstartY = self.mouseY
            self.drawendY = self.mouseY
            self.draw_toggle = True

        #while mouse is down we draw the line template to the work canvas
        elif self.draw_toggle == True and self.mousedown and self.mousebutton == 1:
            self.drawendX = self.mouseX
            self.drawendY = self.mouseY
            #We clear the work canvas with black
            self.work_canvas.fill(self.BLACK)
            #We draw a line into the work_canvas in GREY1
            pygame.draw.line(self.work_canvas,
                             (self.COLOR_ACTIVO),
                             (self.drawstartX,self.drawstartY),
                             (self.drawendX,self.drawendY),
                             4)
            #blit the work_canvas onto the canvas
            self.draw_tool_template()

        #when we release the left mouse, we draw a white line to the paint canvas
        elif self.draw_toggle == True and not self.mousedown and self.mousebutton == 1:
            self.draw_toggle = False
            #We draw a line into the paint_canvas in WHITE
            pygame.draw.line(self.paint_canvas,
                             (self.COLOR_ACTIVO),
                             (self.drawstartX,self.drawstartY),
                             (self.drawendX,self.drawendY),
                             4)

    def show_mousestate(self):
        pass
        
    def draw_tool_template(self):
        #add tool_template to the draw items list
        item = draw_item()
        item.add(self.work_canvas,0,0)
        self.draw_list.append(item)

    def canvas_draw(self):
        #NB: for now we clear the canvas with black
        self.canvas.fill(self.BLACK)

        #we first blit our paint canvas
        self.canvas.blit(self.paint_canvas,(0,0))

        #We get all the draw items from the list, and blit them to the canvas
        for i in self.draw_list:
            self.canvas.blit(i.surface,(i.left,i.top))
        
    def draw(self):
        self.canvas_draw()
        self.screen.blit(self.canvas, (0, 0))

        if pygame.time.get_ticks()>self.lastDraw+5000:
            self.lastDraw = pygame.time.get_ticks()
            canvas_tensor = pygame.surfarray.pixels3d(self.canvas)
            canvas_tensor = cv.resize(canvas_tensor,(256,256),interpolation = cv.INTER_AREA)
            canvas_tensor = cv.flip(canvas_tensor, 1)
            canvas_tensor = cv.rotate(canvas_tensor,cv.ROTATE_90_COUNTERCLOCKWISE)
            canvas_tensor = cv.cvtColor(canvas_tensor,cv.COLOR_RGB2BGR)
            canvas_tensor = tf.expand_dims(canvas_tensor,0)/255

            self.prediction = self.model.predict(canvas_tensor)[0]
            self.prediction = cv.resize(self.prediction,(512,512),interpolation=cv.INTER_LANCZOS4)*255
            #self.prediction = cv.cvtColor(self.prediction,cv.COLOR_BGR2GRAY)
            self.prediction = tf.math.round(self.prediction)
            self.prediction = np.int32(self.prediction)
            self.prediction = cv.resize(self.prediction,(512,512))
            self.prediction = cv.flip(self.prediction, 1)
            self.prediction = cv.rotate(self.prediction,cv.ROTATE_90_COUNTERCLOCKWISE)
            self.prediction = self.prediction[:, :, [2, 1, 0]]
            self.img_canvas = pygame.surfarray.make_surface(self.prediction)

        self.screen.blit(self.img_canvas,(self.screen_width/2,0))

        #plt.imshow(self.palette(self.PALETA))
        #print("Paleta: ",self.palette(self.PALETA).shape())
        #print("Surface: ",pygame.surfarray.make_surface(self.palette(self.PALETA)).shape())
        #print("Canvas: ",self.palette_canvas.shape())

        #self.palette_canvas.blit(,(0,0))
        self.screen.blit(pygame.surfarray.make_surface(self.palette(self.PALETA)),(0,self.screen_height-self.palette_height))

    def clear(self):
        """We clear the draw list in preperation for drawing on the canvas"""
        self.draw_list = []

    def run(self):
        """This method provides the main application loop.
           It continues to run until either the ESC key is pressed
           or the window is closed
        """
        while 1:

            #we clear and prepare the draw_list for the canvas
            self.clear()
            events = pygame.event.get()
            for e in events:

                #Set quit state when window is closed
                if e.type == pygame.QUIT :
                    self.QUIT = True
                if e.type == KEYDOWN:
                    #Set quit state on Esc key press
                    if e.key == K_ESCAPE:
                        self.QUIT = True
                    #Toggle Line drawing
                    if e.key == K_l:
                        self.draw_tool = "Line"
                    #Toggle Circle drawing
                    if e.key == K_c:
                        self.draw_tool = "Circle"
                    if e.key == K_k:
                        self.draw_tool = "Curve"
                    if e.key == K_f:
                        self.draw_tool = "Fill"
                    if e.key == K_p:
                        self.draw_tool = "Pick"
                        
                                    
            if self.QUIT:
                #Exit pygame gracefully
                pygame.quit()
                sys.exit(0)

            #call the mouse handler with current events
            self.mouse_handler(events)

            #Process any drawing that needs to be done
            self.draw()

            #flip the display
            pygame.display.flip()

            

if __name__ == "__main__":
    mypaint = pintura()
    mypaint.run()
    
