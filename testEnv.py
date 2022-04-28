# -*- coding: utf-8 -*-
"""
Latest update on Thu Apr 28 13:15:17 2022

@author: mreal
"""


import math as m
from threading import Semaphore
import turtle as t
import numpy as np
sph = Semaphore(1)
class Env():
    def __init__(self):
        #Setting the simulation screen
        self.mar3 = t.Screen()
        self.mar3.clearscreen()
        self.mar3.title('Real Robots Environment')
        self.mar3.bgcolor('#03273c')
        self.mar3.tracer(0)
        self.mar3.setup(width = 1500,height = 800)
        
        
        #AUV:
        self.targ = t.Turtle()
        self.targ.shape('arrow')
        self.targ.speed(0)
        self.targ.color('#68d897','#2ca49a')
       
        self.targ.pensize(2)
        self.targ.penup()
        self.targ.goto(-550,0)
        self.targ.pendown()
        
        self.pest = t.Turtle()
        self.pest.shape('arrow')
        self.pest.speed(0)
        self.pest.color('#8c71b7','#603474')
        
        self.pest.pensize(2)
        self.pest.penup()
        self.pest.goto(-550,0)
        self.pest.pendown()
        
        #ASVs: 
        self.bot1 = t.Turtle()
        self.bot1.shape('arrow')
        self.bot1.speed(0)
        self.bot1.color('#F10aac','#ab0e65')
        self.bot1.penup()
        self.bot1.goto(-700,100)
        self.bot1.pendown()
        self.bot1.pensize(2)
        
        
        self.bot2 = t.Turtle()
        self.bot2.shape('arrow')
        self.bot2.speed(0)
        self.bot2.color('#Eb6a1d','#f5a24d')
        self.bot2.penup()
        self.bot2.goto(-700,-100)
        self.bot2.pendown()
        self.bot2.pensize(2)
        self.mar3.update()
        self.peste = False
        
        #angle error
        self.eang1 = 0
        self.eang1f = 0
        self.eang2 = 0
        self.eang2f = 0
        
        
        #bots coordinates
        self.bot1cx = -700
        self.bot1cy = 100
        self.bot2cx = -700
        self.bot2cy = -100  

        self.bot1cxf = -700
       	self.bot1cyf = 100
        self.bot2cxf = -700
       	self.bot2cyf = -100     
           
        #initial velocities
        self.surge1 = 0
        self.yaw1 = 0
        self.rtheta = 0
        
        self.surge2 = 0
        self.yaw2 = 0
        self.mtheta = 0

        #position of target
       	self.targcx = -550
       	self.targcy = 0
        
        self.targcxi = self.targcx
        self.targcyi = self.targcy
                
        self.targ.setx(self.targcx)
        self.targ.sety(self.targcy)
        
        #position estimation
        self.pestcx = -550
       	self.pestcy = 0
        self.pestcxi = self.pestcx
        self.pestcyi = self.pestcy
        self.pest.setx(self.pestcx)
        self.pest.sety(self.pestcy)
        
        self.norm1f = 0.0
        self.norm2f = 0.0
        
        #distances from target
        self.dist1fx = self.targcx-self.bot1cxf
        self.dist2fx = self.targcx-self.bot2cxf
        self.dist1fy = self.targcy-self.bot1cyf
        self.dist2fy = self.targcy-self.bot2cyf
        
        #distances between bots
        self.ntrx = self.bot2cxf-self.bot1cxf
        self.ntry = self.bot2cyf-self.bot1cyf
        self.ntmx = self.bot1cxf-self.bot2cxf
        self.ntmy = self.bot1cyf-self.bot2cyf
        self.normt = m.sqrt(self.ntrx**2+self.ntry**2)
        
        self.angl = 0
        self.maxen = 400000 #max energy

        #ANALISYS VARIABLES
        self.allsurge1 = []
        self.allsurge2 = []
        self.allyaw1 = []
        self.allyaw2 = []
        self.encomr = []
        self.encomm = []
        
        self.allest = []
        self.allangl1 = []
        self.allangl2 = []
        
        self.alldist1 = []
        self.alldist2 = []
        self.allhead1 = []
        self.allhead2 = []
        self.allvel1 = []
        self.allvel2 = []
        
    #Angle computations
    def angle_normalize(self,x):
        return (((x+m.pi) % (2*m.pi)) - m.pi)    
    
    def angle_comp(self,ay,ax,by,bx,cy,cx):
        ang1 = abs(m.degrees(m.atan2(ay-by,ax-bx)-m.atan2(cy-by,cx-bx)))
        ang2 = abs(m.degrees(m.atan2(by-ay,bx-ax)-m.atan2(by-cy,bx-cx)))
        print(ang1,ang2)
        if ang1>ang2: ang = ang2
        else: ang = ang1
        return ang - 180 if ang > 180 else ang
    
    #Circle intersection computations
    def intersec(self,di,x0,y0,x1,y1,r0,r1):
        if di>r0+r1 or di<abs(r0-r1) or di==0:
            x = float('NaN')
            y = float('NaN')

            return x,y,x,y
        
        a = (r0**2-r1**2+di**2)/(2*di)
        h = m.sqrt(r0**2-a**2)

        
        x2 = x0+a*(x1-x0)/di
        y2 = y0+a*(y1-y0)/di
        
        if di == r0+r1:
            return x2,y2
        
        else:
            x3a = x2+h*(y1-y0)/di
            y3a = y2-h*(x1-x0)/di
            x3b = x2-h*(y1-y0)/di
            y3b = y2+h*(x1-x0)/di

            return x3a,y3a,x3b,y3b
  
            
    def intersec4(self,x0,y0,x1,y1,xl,yl):
        s1x = x0+1
        s1y = y0
        s2x = x0-1
        s2y = y0
        s3x = x1
        s3y = y1+1
        s4x = x1
        s4y = y1-1
        
        
        s13x = abs(s1x-s3x)
        s13y = abs(s1y-s3y)
        dist13 = m.sqrt(s13x**2+s13y**2)
        s1lx = abs(s1x-xl)
        s1ly = abs(s1y-yl)
        dist1l = m.sqrt(s1lx**2+s1ly**2)
        s3lx = abs(xl-s3x)
        s3ly = abs(yl-s3y)
        dist3l = m.sqrt(s3lx**2+s3ly**2)
        
        p1x,p1y,p2x,p2y = self.intersec(dist13,s1x,s1y,s3x,s3y,dist1l,dist3l)
        
        s24x = abs(s2x-s4x)
        s24y = abs(s2y-s4y)
        dist24 = m.sqrt(s24x**2+s24y**2)
        s2lx = abs(s2x-xl)
        s2ly = abs(s2y-yl)
        dist2l = m.sqrt(s2lx**2+s2ly**2)
        s4lx = abs(xl-s4x)
        s4ly = abs(yl-s4y)
        dist4l = m.sqrt(s4lx**2+s4ly**2)
        
        p3x,p3y,p4x,p4y = self.intersec(dist24,s2x,s2y,s4x,s4y,dist2l,dist4l)
        
        if abs(p1x-p3x)<abs(p2x-p4x) and abs(p1y-p3y)<abs(p2y-p4y):
            ex = np.mean([p1x,p3x])
            ey = np.mean([p1y,p3y])
            return ex,ey
        else:
            ex = np.mean([p2x,p4x])
            ey = np.mean([p2y,p4y])
            return ex,ey
        
    def reset(self,fil):
        # state at the start of the simulation
        self.mar3.update()
        self.round = 0
        
        self.targvel =  1.25 #velocity of target
        self.angle = 45     #angle for the formation
        self.currx = 0      #current in x
        self.curry = 0      #current in y
        self.count = 0      #counter
        self.anglrand =  0  #heading of target
        r1 = 0              #random distance in x from initial position of bot 1
        r2 = 0              #random distance in y from initial position of bot 1
        r3 = 0              #random distance in x from initial position of bot 2
        r4 = 0              #random distance in y from initial position of bot 2

        self.bot1cx = -700+r1 
        self.bot1cy = 100+r2
        self.bot2cx = -700+r3
        self.bot2cy = -100+r4

        #Placing the bots in the new position in the simulation
        self.bot1.setx(self.bot1cx)
        self.bot1.sety(self.bot1cy)
        self.bot2.setx(self.bot2cx)
        self.bot2.sety(self.bot2cy)
        
        self.bot1cxf = -700+r1
        self.bot1cyf = 100+r2
        self.bot2cxf = -700+r3
        self.bot2cyf = -100+r4
        
        #Distances from target
        self.dist1fx = self.targcx-self.bot1cxf
        self.dist2fx = self.targcx-self.bot2cxf
        self.dist1fy = self.targcy-self.bot1cyf
        self.dist2fy = self.targcy-self.bot2cyf
        
        #previous action
        self.surge1i = 0
        self.yaw1i = 0
        self.surge2i = 0
        self.yaw2i = 0
        
        #energy consumed
        self.en1 = 0
        self.en2 = 0
        self.en1per = self.en1/self.maxen*100
        self.en2per = self.en2/self.maxen*100
        
        #difference of velocity
        self.difvel1 = self.surge1-self.targvel
        self.difvel2 = self.surge2-self.targvel
        
        #normalized angle error
        self.eangnorm1 = self.eang1/360*100
        self.eangnorm2 = self.eang2/360*100
        
        #distance between bots
        self.norm1f = m.sqrt(self.dist1fx**2+self.dist1fy**2)
        self.norm2f = m.sqrt(self.dist2fx**2+self.dist2fy**2)
        
        #heading of bots
        self.head1 = 0
        self.head2 = 0
        self.headnorm1 = self.head1/360*100
        self.headnorm2 = self.head2/360*100
        
        
        self.allhead1.append(self.head1)
        self.allhead2.append(self.head2)
        self.allvel1.append(self.difvel1)
        self.allvel2.append(self.difvel2)
        self.alldist1.append(self.norm1f)
        self.alldist2.append(self.norm2f)
        

        if fil == 0:
            state = [.01*abs(self.dist1fx),.01*abs(self.dist1fy),.01*self.normt ,.01*self.eangnorm1,  .01*self.headnorm1,.01*self.difvel1,.01*self.en1per]
            
        else:
            state = [.01*abs(self.dist2fx),.01*abs(self.dist2fy), .01*self.normt ,.01*self.eangnorm2,  .01*self.headnorm2,.01*self.difvel2,.01*self.en2per]
            
        self.done = False
    
        return state
    
    def updatestate(self):
        
        xcor1 = self.bot1cx
        ycor1 = self.bot1cy
        xcor2 = self.bot2cx
        ycor2 = self.bot2cy
        
        xcor1f = self.bot1cxf
        ycor1f = self.bot1cyf
        xcor2f = self.bot2cxf
        ycor2f = self.bot2cyf
        
        
        xdif = abs(xcor1-xcor2)
        ydif = abs(ycor1-ycor2)
        self.ntrx = xcor2-xcor1
        self.ntry = ycor2-ycor1
        self.ntmx = xcor1-xcor2
        self.ntmy = ycor1-ycor2
        
        #euclidean distance between bots
        self.normt = m.sqrt(xdif**2+ydif**2)
      
        self.pestcxi = self.pestcx
        self.pestcyi = self.pestcy
        
        #position estimation with intersection of circles
        self.pestcx,self.pestcy = self.intersec4(xcor1,ycor1,xcor2,ycor2,self.targcx,self.targcy)
        
        xdifd = abs(self.pestcx-self.targcx)
        ydifd = abs(self.pestcy-self.targcy)
        estimacio = m.sqrt(xdifd**2+ydifd**2)
        self.allest.append(estimacio)

            
        self.count +=1
        
        #addition of currents if needed
        self.bot1cx = xcor1f+self.currx
       	self.bot1cy = ycor1f+self.curry
       	self.bot2cx = xcor2f+self.currx
       	self.bot2cy = ycor2f+self.curry
           
        
        self.targcxi = self.targcx
        self.targcyi = self.targcy
        
        #new position of target
        self.targcx = self.targcx+self.targvel
        self.targcy = self.targcy
        
        self.targ.setx(self.targcx)
        self.targ.sety(self.targcy)
        
        
        if m.isnan(self.pestcx) == False and m.isnan(self.pestcy) == False: 
            self.pest.setx(self.pestcxi)
            self.pest.sety(self.pestcyi)

        #end of mission
        if self.targcx >300 or self.count>1000:
        
            self.done = True
       
        self.mar3.update()
        return self.done
    
    def upbot1(self):
        
        xcorui = self.pestcxi
        ycorui = self.pestcyi
        xcoru = self.pestcx
        ycoru = self.pestcy
        
        lxdif = xcoru-xcorui
        lydif = ycoru-ycorui
        dirl = m.degrees(m.atan2(lydif,lxdif))
        
        xcor1 = self.bot1cx
        ycor1 = self.bot1cy
        
        
        xdifi = abs(xcor1-xcoru)
        ydifi = abs(ycor1-ycoru)
        self.norm1i = m.sqrt(xdifi**2+ydifi**2)
        
        #new position of bot1
        self.rtheta = self.angle_normalize(self.rtheta)+self.yaw1
        xcor1f = xcor1+self.surge1*m.cos(self.rtheta)
        ycor1f = ycor1+self.surge1*m.sin(self.rtheta)
        
        self.dist1fx = abs(xcoru-xcor1f)
        self.dist1fy = abs(ycoru-ycor1f)
        self.norm1f = m.sqrt(self.dist1fx**2+self.dist1fy**2)
        
        #computation of angle error
        self.eang1 = self.angle_comp(ycor1f, xcor1f, ycoru, xcoru, ycorui, xcorui)-self.angle
        self.allangl1.append(abs(self.eang1))
        
        #difference in heading from target
        self.head1 = m.degrees(self.rtheta)-dirl
        
        self.eang1f = self.eang1

        #computation of power and energy
        pows1 = 60*abs(self.surge1)**3

        powy1 = 80*abs(self.yaw1)**3
        
        self.en1 += pows1+ powy1+40
        self.en1per = self.en1/self.maxen*100
        self.encomr.append(self.en1per)
        
        print('ENERGY bot1: ',self.en1per)
        
       	self.bot1cyf = ycor1f
        self.bot1cxf = xcor1f
        self.eangnorm1 = self.eang1/360*100
        self.headnorm1 = self.head1/360*100
        
        self.bot1.setx(self.bot1cxf)
        self.bot1.sety(self.bot1cyf)
       
           
    def upbot2(self):
        xcorui = self.pestcxi
        ycorui = self.pestcyi
        xcoru = self.pestcx
        ycoru = self.pestcy
        
        lxdif = xcoru-xcorui
        lydif = ycoru-ycorui
        dirl = m.degrees(m.atan2(lydif,lxdif))
        
        xcor2 = self.bot2cx
        ycor2 = self.bot2cy

        xdifi = abs(xcor2-xcoru)
        ydifi = abs(ycor2-ycoru)
        self.norm2i = m.sqrt(xdifi**2+ydifi**2)
        
        #new position of bot2
        self.mtheta = self.angle_normalize(self.mtheta)+self.yaw2
        xcor2f = xcor2+self.surge2*m.cos(self.mtheta)
        ycor2f = ycor2+self.surge2*m.sin(self.mtheta)
        
        self.dist2fx = abs(xcoru-xcor2f)
        self.dist2fy = abs(ycoru-ycor2f)
        self.norm2f = m.sqrt(self.dist2fx**2+self.dist2fy**2)
        
        #angle error computation
        self.eang2 = self.angle_comp(ycor2f, xcor2f, ycoru, xcoru, ycorui,xcorui)-self.angle
        self.allangl2.append(abs(self.eang2))
        self.head2 = m.degrees(self.mtheta)-dirl
       
        self.eang2f = self.eang2
        
        #computation of power and energy
        pows2 = 60*abs(self.surge2)**3
        powy2 = 80*abs(self.yaw2)**3
        self.en2 +=  pows2+powy2 +40
        self.en2per = self.en2/self.maxen*100
        self.encomm.append(self.en2per)
        print('ENERGY bot2: ',self.en2per)
        
       	self.bot2cxf = xcor2f
       	self.bot2cyf = ycor2f       
        self.eangnorm2 = self.eang2/360*100
        self.headnorm2 = self.head2/360*100
        self.bot2.setx(self.bot2cxf)
        self.bot2.sety(self.bot2cyf)


    def Bot1(self,action):

        self.done = False
        actions = action[0]
        self.surge1 =(actions[0]*1.25+1.25)
        print('SURGE bot1: ',self.surge1)
        self.yaw1 = (actions[1]*m.pi/8)
        print('YAW bot1: ',m.degrees(self.yaw1))
        self.allsurge1.append(self.surge1)
        self.allyaw1.append(m.degrees(self.yaw1))
        
        self.upbot1()
        self.difvel1 = self.surge1-self.targvel
        self.allhead1.append(self.head1)
        self.allvel1.append(self.difvel1)
        self.alldist1.append(self.norm1f)
        
        
        state = [.01*abs(self.dist1fx),.01*abs(self.dist1fy),.01*self.normt ,.01*self.eangnorm1,  .01*self.headnorm1,.01*self.difvel1,.01*self.en1per]
        self.yaw1i = self.yaw1
        self.surge1i = self.surge1
        return state
    
    def Bot2(self,action):
        
        self.done = False
        actions = action[0]
        self.surge2 =(actions[0]*1.25+1.25)
        print('SURGE bot2: ',self.surge2)
        self.yaw2 = (actions[1]*m.pi/8)
        print('YAW bot2: ',m.degrees(self.yaw2))
        self.allsurge2.append(self.surge2)
        self.allyaw2.append(m.degrees(self.yaw2))
        
        self.upbot2()
        self.difvel2 = self.surge2-self.targvel
        self.allhead2.append(self.head2)
        self.allvel2.append(self.difvel2)
        self.alldist2.append(self.norm2f)
        
        state = [.01*abs(self.dist2fx),.01*abs(self.dist2fy), .01*self.normt ,.01*self.eangnorm2,  .01*self.headnorm2,.01*self.difvel2,.01*self.en2per]
        self.yaw2i = self.yaw2
        self.surge2i = self.surge2
        return  state
    
    #To gather info for later analysis:
    def Statistics(self):
        meansurge1 = np.mean(self.allsurge1)
        meansurge2 = np.mean(self.allsurge2)
        varsurge1 = np.var(self.allsurge1)
        varsurge2 = np.var(self.allsurge2)
        
        meanyaw1 = np.mean(self.allyaw1)
        meanyaw2 = np.mean(self.allyaw2)
        varyaw1 = np.var(self.allyaw1)
        varyaw2 = np.var(self.allyaw2)
        
        fullest = sum(self.allest)
        fullangl1 = sum(self.allangl1)
        fullangl2 = sum(self.allangl2)
        
        meanangl1 = np.mean(self.allangl1)
        meanangl2 = np.mean(self.allangl2)
        
        meandist1 = np.mean(self.alldist1)
        meandist2 = np.mean(self.alldist2)
        
        statsr = [self.en1per,meansurge1,meanyaw1,varsurge1,varyaw1,fullest,meanangl1,meandist1]
        statsm = [self.en2per,meansurge2,meanyaw2,varsurge2,varyaw2,fullest,meanangl2,meandist2]
        difstats = [self.alldist1,self.alldist2,self.allangl1,self.allangl2,self.allvel1,self.allvel2,self.allhead1,self.allhead2]
        # plt.plot(self.allsurge1)
        # plt.plot(self.allyaw1)
        # plt.plot(self.encomr)
        # plt.xlabel("Step")
        # plt.ylabel("Energy analysis")
        # plt.title('ASV 1')
        # plt.show()  
        
        # plt.plot(self.allsurge2)
        # plt.plot(self.allyaw2)
        # plt.plot(self.encomm)
        # plt.xlabel("Step")
        # plt.ylabel("Energy analysis")
        # plt.title('ASV 2')
        # plt.show()  
        
        # plt.plot(self.alldist1)
        # plt.plot(self.alldist2)
        # # plt.plot(self.encomr)
        # plt.xlabel("Step")
        # plt.ylabel("Energy analysis")
        # plt.title('ASV 1')
        # plt.show()  
        
        return difstats

        
        
        
        
        