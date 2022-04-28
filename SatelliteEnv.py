# -*- coding: utf-8 -*-
"""
Latest update on Wed Apr 27 16:45:45 2022

@author: mreal
"""

import math as m
from threading import Semaphore
import numpy as np
from random import random
sph = Semaphore(1)


class Env():
    def __init__(self):
        self.dt = 1
        self.nepi = 0   #nº of episodes
        self.done = False
        self.reward = 0
        
        #to gather info:
        self.tjx1 = []
        self.tjy1 = []
        self.tjx2 = []
        self.tjy2 = []
        self.tjlx = []
        self.tjly = []

        
        self.g1x1 = []
        self.g1y1 = []
        self.g2x1 = []
        self.g2y1 = []
 
        #set initial coordinates of bots
        self.bot1cx = -700
        self.bot1cy = 100
        self.bot2cx = -700
        self.bot2cy = -100  

        self.bot1cxf = -700
       	self.bot1cyf = 100
        self.bot2cxf = -700
       	self.bot2cyf = -100     
           
        #velocities of each bot
        self.surge1 = 0
        self.yaw1 = 0
        self.surge1i = 0
        self.yaw1i = 0
        self.theta1 = 0
        
        self.surge2 = 0
        self.yaw2 = 0
        self.surge2i = 0
        self.yaw2i = 0
        self.theta2 = 0
        
        #initial position of target
       	self.targcx = -550
       	self.targcy = 0
        self.targcxi = self.targcx
        self.targcyi = self.targcy
        
        #initial position estimation
        self.pestcx = -550
        self.pestcy = 0
        
        #angle errors
        self.eang1 = 0
        self.eang1f = 0
        self.eang2 = 0
        self.eang2f = 0
        
        #distances from target
        self.dist1fx = self.targcx-self.bot1cxf
        self.dist2fx = self.targcx-self.bot2cxf
        self.dist1fy = self.targcy-self.bot1cyf
        self.dist2fy = self.targcy-self.bot2cyf
        self.norm1f = m.sqrt(self.dist1fx**2+self.dist1fy**2)
        self.norm2f = m.sqrt(self.dist2fx**2+self.dist2fy**2)
        
        #distances between bots
        self.nt1x = self.bot2cxf-self.bot1cxf
        self.nt1y = self.bot2cyf-self.bot1cyf
        self.nt2x = self.bot1cxf-self.bot2cxf
        self.nt2y = self.bot1cyf-self.bot2cyf
        self.normt = m.sqrt(self.nt1x**2+self.nt1y**2)
        
        #to process data after training:
        self.lastscore1 = -50000000
        self.lastscore2 = -50000000
        
        self.prevtj1 = 0
        self.prevtj2 = 0
        
        self.freward1 = []
        self.freward2 = []
        
        self.fp31 = []
        self.fp11 = []
        self.fp21 = []
        self.fp61 = []
        self.fp41 = []
        self.fp51 = []
        
        self.fp32 = []
        self.fp12 = []
        self.fp22 = []
        self.fp62 = []
        self.fp42 = []
        self.fp52 = []
        
        self.action1s = []
        self.action2s = []
        self.action1y = []
        self.action2y = []
        self.totalest = []
        
        #initialization of states arrays
        self.states1 = []
        self.states2 = []
        
        
        self.angl = 0

        self.maxen =400000 #max energy
        
        self.targvel = 1.25*self.dt #target velocity
        
        #WEIGHTS:
        self.w1 = .4    
        self.w2 = .1
        self.w3 = .3
        self.w4 = .05
        self.w5 = .05
        self.w6 = .1
        
    #angle computations
    def angle_normalize(self,x):
        return (((x+m.pi) % (2*m.pi)) - m.pi)
    def angle_comp(self,ay,ax,by,bx,cy,cx):
        ang1 = abs(m.degrees(m.atan2(ay-by,ax-bx)-m.atan2(cy-by,cx-bx)))
        ang2 = abs(m.degrees(m.atan2(by-ay,bx-ax)-m.atan2(by-cy,bx-cx)))

        if ang1>ang2: ang = ang2
        else: ang = ang1
        return ang - 180 if ang > 180 else ang
        
    #circle intersection computations
    def intersec(self,di,x0,y0,x1,y1,r0,r1):
        if di>r0+r1 or di<abs(r0-r1) or di==0:
            x = float('NaN')
            y = float('NaN')
            return x,y
        
        a = (r0**2-r1**2+di**2)/(2*di)
        h = m.sqrt(r0**2-a**2)

        x2 = x0+a*(x1-x0)/di
        y2 = y0+a*(y1-y0)/di
        
        if di == r0+r1:
            return x2,y2,x2,y2
        
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
        
    def reset(self,phase):
        # state at the start of the game
        self.reward = 0
        self.nsteps = 0
        self.angle = 45     #angle of the formation
        self.p61 = 0    #energy peanlty bot 1
        self.p62 = 0    #energy penalty bot 2
        self.currx1 = 0     #current in x
        self.curry1 = 0     #current in y
        self.currx2 = 0     #current in x
        self.curry2 = 0     #current in y
        if phase == 0:
            self.anglrand = 0   #heading of target
            self.targvel = 1.25*self.dt
            self.r1 = 0          #random distance in x from initial position of bot 1
            self.r2 = 0         #random distance in y from initial position of bot 1
            self.r3 = 0         #random distance in x from initial position of bot 2
            self.r4 = 0         #random distance in y from initial position of bot 2
        
        #set difficulty of training:
        elif phase == 1:
            self.anglrand = 0
            self.targvel = 1.25*self.dt
            self.r1 = -50+(random()*100)
            self.r2 = -50+(random()*100)
            self.r3 = -50+(random()*100)
            self.r4 = -50+(random()*100)
        elif phase == 2:
            self.r1 = -50+(random()*100)
            self.r2 = -50+(random()*100)
            self.r3 = -50+(random()*100)
            self.r4 = -50+(random()*100)
            self.targvel = (1.35+(-.5+random()))*self.dt
            self.anglrand = 0
        elif phase == 3:
            self.anglrand = -.5+(random())
            self.r1 = -50+(random()*100)
            self.r2 = -50+(random()*100)
            self.r3 = -50+(random()*100)
            self.r4 = -50+(random()*100)
            self.targvel = (1.3+(-.5+random()))*self.dt
        elif phase == 4:
            self.anglrand = 0
            self.r1 = -50+(random()*100)
            self.r2 = -50+(random()*100)
            self.r3 = -50+(random()*100)
            self.r4 = -50+(random()*100)
            self.targvel = (1.3+(-.5+random()))*self.dt
            self.currx1 = -.5+(random())
            self.curry1 = -.5+(random())
            self.currx2 = -.5+(random())
            self.curry2 = -.5+(random())
        elif phase == 5:
            self.anglrand = 0
            self.targvel = 1.25*self.dt
            self.r1 = -50+(random()*100)
            self.r2 = -50+(random()*100)
            self.r3 = -50+(random()*100)
            self.r4 = -50+(random()*100)
            self.cercle = 1
            
            
        self.bot1cx = -700+self.r1
        self.bot1cy = 100+self.r2
        self.bot2cx = -700+self.r3
        self.bot2cy = -100+self.r4
        
        self.bot1cxf = -700+self.r1
        self.bot1cyf = 100+self.r2
        self.bot2cxf = -700+self.r3
        self.bot2cyf = -100+self.r4
        
        self.surge1 = 0
        self.yaw1 = 0
        self.surge1i = 0
        self.yaw1i = 0
        self.theta1 = 0
        
        self.surge2 = 0
        self.yaw2 = 0
        self.surge2i = 0
        self.yaw2i = 0
        self.theta2 = 0
        
        #difference in velocity from target
        self.difvel1 = self.surge1-self.targvel
        self.difvel2 = self.surge2-self.targvel
        
        #diffentece in heading from target
        self.head1 = 0
        self.head2 = 0
        

        self.g1x = []
        self.g1y = []
        self.g2x = []
        self.g2y = []
        
        
        self.en1 = 0 #energy consumed by bot 1
        self.en2 = 0 #energy consumed by bot 2
        self.en1per = self.en1/self.maxen*100
        self.en2per = self.en2/self.maxen*100
        self.eangnorm1 = self.eang1/360*100
        self.eangnorm2 = self.eang2/360*100
        self.headnorm1 = self.head1/360*100
        self.headnorm2 = self.head2/360*100
        self.estim = []
        full_state = []
        
        self.dist1fx = self.targcx-self.bot1cxf
        self.dist2fx = self.targcx-self.bot2cxf
        self.dist1fy = self.targcy-self.bot1cyf
        self.dist2fy = self.targcy-self.bot2cyf

        self.nt1x = self.bot2cxf-self.bot1cxf
        self.nt1y = self.bot2cyf-self.bot1cyf
        self.nt2x = self.bot1cxf-self.bot2cxf
        self.nt2y = self.bot1cyf-self.bot2cyf
        
        #state of bot
        state1 = [.01*abs(self.dist1fx),.01*abs(self.dist1fy),.01*self.normt ,.01*self.eangnorm1, .01*self.headnorm1,.01*self.difvel1,.01*self.en1per]
        self.states1.append(state1)
       
        state2 = [.01*abs(self.dist2fx),.01*abs(self.dist2fy),.01*self.normt , .01*self.eangnorm2, .01*self.headnorm2,.01*self.difvel2,.01*self.en2per]
        self.states2.append(state2)
        self.done = False
        
        full_state.append(state1)
        full_state.append(state2)

        return full_state
    
    def updatestate(self):
        
        xcor1 = self.bot1cx
        ycor1 = self.bot1cy
        xcor2 = self.bot2cx
        ycor2 = self.bot2cy
        
        xcor1f = self.bot1cxf
        ycor1f = self.bot1cyf
        xcor2f = self.bot2cxf
        ycor2f = self.bot2cyf
        
        print('posició bot1:')
        print(xcor1f)
        print(ycor1f)
        print('error: ', self.eang1,'\n')
        print('posició bot2:')
        print(xcor2f)
        print(ycor2f)
        print('error: ', self.eang2,'\n')
        
        #distance between bots
        xdif = abs(xcor1-xcor2)
        ydif = abs(ycor1-ycor2)
        
        self.normt = m.sqrt(xdif**2+ydif**2)
        self.nt1x = xcor2-xcor1
        self.nt1y = ycor2-ycor1
        self.nt2x = xcor1-xcor2
        self.nt2y = ycor1-ycor2
 
        #penalty of distance between bots (p2)
        p2temp = (-(1-m.exp(-((self.normt-100)**2)/500000)))*self.w2

        self.p2 += p2temp
        #position estimation
        self.pestcx,self.pestcy = self.intersec4(xcor1,ycor1,xcor2,ycor2,self.targcx,self.targcy)
        
        xdifd = abs(self.pestcx-self.targcx)
        ydifd = abs(self.pestcy-self.targcy)
        estimacio = m.sqrt(xdifd**2+ydifd**2)
        print('Error Estimació: ',estimacio)
        self.estim.append(estimacio)

        self.reward += self.p2
        
        #adding currents if needed
        self.bot1cx = xcor1f+self.currx1
       	self.bot1cy = ycor1f+self.curry1
       	self.bot2cx = xcor2f+self.currx2
       	self.bot2cy = ycor2f+self.curry2
           
        self.tjlx.append(self.targcx)
        self.tjly.append(self.targcy)

        
        #update target position
        self.targcxi = self.targcx
        self.targcyi = self.targcy
        self.targcx = self.targcx+self.targvel*m.cos(self.angl)
        self.targcy = self.targcy-self.targvel*m.sin(self.angl)

        
        print('Posició targ: ', self.targcx)
        
        #end of episode:
        if self.targcx >200 or self.nsteps > 1200:

            self.angl = 0
            self.finest = np.mean(self.estim)
            self.totalest.append(self.finest)
            self.done = True
            self.nepi +=1
            self.nsteps = 0

        
        
    def upbot1(self):
        #computations of bot 1
        xcorui = self.targcxi
        ycorui = self.targcyi
        xcoru = self.targcx
        ycoru = self.targcy
        
        lxdif = xcoru-xcorui
        lydif = ycoru-ycorui
        dirl = m.degrees(m.atan2(lydif,lxdif))
        
        xcor1 = self.bot1cx
        ycor1 = self.bot1cy
        
        xdifi = abs(xcor1-xcoru)
        ydifi = abs(ycor1-ycoru)
       
        self.norm1i = m.sqrt(xdifi**2+ydifi**2)
        
        #new position of bot1
        self.theta1 = self.angle_normalize(self.theta1)+self.yaw1
        xcor1f = xcor1+self.surge1*m.cos(self.theta1)
        ycor1f = ycor1+self.surge1*m.sin(self.theta1)
        
        self.dist1fx = abs(xcoru-xcor1f)
        self.dist1fy = abs(ycoru-ycor1f)
        self.norm1f = m.sqrt(self.dist1fx**2+self.dist1fy**2)
        
        #angle error computation
        self.eang1 = self.angle_comp(ycor1f, xcor1f, ycoru, xcoru, ycorui, xcorui)-self.angle
        self.head1 = m.degrees(self.theta1)-dirl
        if abs(self.head1) < 1: #penalty on difference of heading
            self.reward1 +=.5*self.w5
        #penalty on angle error (p3)
        p3temp =  (-(1-m.exp(-(self.eang1**2)/100)))*self.w3
        
        self.p31 += p3temp
        
        self.eang1f = self.eang1
       
        #penalty on surge power (p4)
        pows1 = 60*abs(self.surge1/self.dt)**3
        p4temp = (-(1-m.exp(-pows1**2/50000)))*self.w4
        
        self.p41 += p4temp
       
        
        #peanlty on yaw power (p5)
        powy1 = 80*abs(self.yaw1/self.dt)**3
        p5temp = (m.exp(-(self.yaw1**2+self.yaw1i**2)*20)-1)*self.w5
        
        self.p51 += p5temp  
        self.en1 += pows1+ powy1+40
        self.en1per = self.en1/self.maxen*100
        print('ENERGY bot1: ',self.en1per,'\n')
        
        #penalty on energy (p6)
        p6temp = (-(1-m.exp(-(self.en1per**2)/2500)))*self.w6
        
        self.p61 += p6temp+p6temp*self.enp1

        #penalty on distance from target
        p1temp = (-(1-m.exp(-(self.norm1f**2)/50000)))*self.w1
        
        self.p11 += p1temp
        
        self.reward1 += self.p11 + self.p31 + self.p41 + self.p51 + self.p61
        
       	self.bot1cyf = ycor1f
        self.bot1cxf = xcor1f
        
        self.eangnorm1 = self.eang1/360*100
        self.headnorm1 = self.head1/360*100
        
        self.tjx1.append(self.bot1cxf)
       	self.tjy1.append(self.bot1cyf)
        self.g1x.append(self.bot1cxf)
       	self.g1y.append(self.bot1cyf)
        self.action1s.append(self.surge1)
        self.action1y.append(self.yaw1)
           
    def upbot2(self):
        #computations of bot2
        xcorui = self.targcxi
        ycorui = self.targcyi
        xcoru = self.targcx
        ycoru = self.targcy
        
        lxdif = xcoru-xcorui
        lydif = ycoru-ycorui
        dirl = m.degrees(m.atan2(lydif,lxdif))

        
        xcor2 = self.bot2cx
        ycor2 = self.bot2cy

        xdifi = abs(xcor2-xcoru)
        ydifi = abs(ycor2-ycoru)
        self.norm2i = m.sqrt(xdifi**2+ydifi**2)
        self.theta2 = self.angle_normalize(self.theta2)+self.yaw2
        xcor2f = xcor2+self.surge2*m.cos(self.theta2)
        ycor2f = ycor2+self.surge2*m.sin(self.theta2)
        self.dist2fx = abs(xcoru-xcor2f)
        self.dist2fy = abs(ycoru-ycor2f)
        self.norm2f = m.sqrt(self.dist2fx**2+self.dist2fy**2)
        
        #angle error computation
        self.eang2 = self.angle_comp(ycor2f, xcor2f, ycoru, xcoru, ycorui, xcorui)-self.angle

        self.head2 = m.degrees(self.theta2)-dirl
        if abs(self.head2) < 10:    #penalty of heading difference
            self.reward2 +=.5*self.w5
        
        
       #peanlty on angle error (p3)
        p3temp =  (-(1-m.exp(-(self.eang2**2)/100)))*self.w3
        
        
        self.p32 += p3temp
        
        self.eang2f = self.eang2

        #penalty on surge power (p4)
        pows2 = 60*abs(self.surge2/self.dt)**3
        p4temp = (-(1-m.exp(-pows2**2/50000)))*self.w4
        
        self.p42 += p4temp
        #penalty yaw power (p5)
        powy2 = 80*abs(self.yaw2/self.dt)**3
        p5temp = (m.exp(-(self.yaw2**2+self.yaw2i**2)*20)-1)*self.w5

        self.p52 += p5temp 

        
        self.en2 +=  pows2+powy2 +40
        self.en2per = self.en2/self.maxen*100
        print('ENERGY bot2: ',self.en2per,'\n')
        #penalty on energy (p6)
        p6temp = (-(1-m.exp(-(self.en2per**2)/2500)))*self.w6

        self.p62 += p6temp+p6temp*self.enp2

        #penalty on distance from target (p1)
        p1temp = (-(1-m.exp(-(self.norm2f**2)/50000)))*self.w1

        self.p12 += p1temp

        self.reward2 += self.p12 +self.p32 + self.p42+ self.p52+self.p62

       	self.bot2cxf = xcor2f
       	self.bot2cyf = ycor2f       
       	
        self.eangnorm2 = self.eang2/360*100
        self.headnorm2 = self.head2/360*100
           
        self.tjx2.append(self.bot2cxf)
       	self.tjy2.append(self.bot2cyf)
        self.g2x.append(self.bot2cxf)
       	self.g2y.append(self.bot2cyf)
        self.action2s.append(self.surge2)
        self.action2y.append(self.yaw2)

    def Bot1(self,action):
        
        self.indv_rw1 = []
        self.p31 = 0
        self.p41 = 0
        self.p51 = 0
        self.p11 = 0
        
        self.done = False
        self.surge1 = (action[0]*1.25+1.25)*self.dt
        print('SURGE bot1: ',self.surge1)
        self.yaw1 =(action[1]*m.pi/8)*self.dt
        print('YAW bot1: ',m.degrees(self.yaw1))
        
        #penalty on energy spend:
        self.enp1=0
        if self.en1per > 100:
            self.surge1 = 0
            self.yaw1 = 0
            self.enp1 = .5
            
        self.upbot1()
        
        #difference in velocity
        self.difvel1 = self.surge1-self.targvel
        
        state = [.01*abs(self.dist1fx),.01*abs(self.dist1fy), .01*self.normt ,.01*self.eangnorm1, .01*self.headnorm1,.01*self.difvel1,.01*self.en1per]
        self.surge1i = self.surge1
        self.yaw1i = self.yaw1
        self.states1.appd(state)
        return state
    
    def Bot2(self,action):
        
        self.indv_rw2 = []
        self.p12 = 0
        self.p32 = 0
        self.p42 = 0
        self.p52 = 0
        
        self.surge2 =(action[0]*1.25+1.25)*self.dt
        print('SURGE bot2: ',self.surge2)
        self.yaw2 = (action[1]*m.pi/8)*self.dt
        print('YAW bot2: ',m.degrees(self.yaw2))
        
        #penalty on energy spend:
        self.enp2=0
        if self.en2per > 100:
            self.surge2 = 0
            self.yaw2 = 0
            self.enp2 = .5
        self.upbot2()
        self.difvel2 = self.surge2-self.targvel
        
        state = [.01*abs(self.dist2fx),.01*abs(self.dist2fy),.01*self.normt , .01*self.eangnorm2, .01*self.headnorm2,.01*self.difvel2,.01*self.en2per]
        self.yaw2i = self.yaw2
        self.surge2i = self.surge2
        self.states2.append(state)
        return state
    
    def step(self,action1,action2):
        self.reward = 0
        self.reward1 = 0
        self.reward2 = 0
        self.done = False
        self.p2 = 0
        full_state = []
        full_reward = []
        full_indv_rw = []
        state1 = self.Bot1(action1[0])
        state2 = self.Bot2(action2[0])
        
        self.updatestate()
        self.reward1 += self.p2
        self.reward2 += self.p2
        self.indv_rw1 = [self.p31,self.p11,self.p2,self.p61,self.p41,self.p51]
        self.indv_rw2 = [self.p32,self.p12,self.p2,self.p62,self.p42,self.p52]
        print('reward1: ', self.reward1)
        print('reward2: ', self.reward2)
        full_state.append(state1)
        full_state.append(state2)
        full_reward.append(self.reward1)
        full_reward.append(self.reward2)
        full_indv_rw.append(self.indv_rw1)
        full_indv_rw.append(self.indv_rw2)
        return full_reward, full_state, self.done,full_indv_rw
        
    #data analysis:
    def trajectory(self):
        return self.tjx1, self.tjy1, self.tjx2, self.tjy2, self.tjlx, self.tjly, self.actionrs,self.actionry,self.actionms,self.actionmy,self.states1,self.states2
            
    def highscore(self,score):
        if score>self.lastscore1 and len(self.g1x)>2:
            self.lastscore1 = score
            self.g1x1, self.g1y1 = self.g1x, self.g1y
            self.g2x1, self.g2y1 = self.g2x, self.g2y
        for k in range(len(self.tjx1)-self.prevtj1):
            self.freward1.append(score)
            self.freward2.append(score)
    
    
    def highscore1(self,score,a,b,c,d,e,f):
        
        if score>self.lastscore1 and len(self.grx)>2:
            self.lastscore1 = score
            self.g1x1, self.g1y1 = self.g1x, self.g1y

            
        for k in range(len(self.tjx1)-self.prevtj1):
            self.freward1.append(score)
            self.fp31.append(a)
            self.fp11.append(b)
            self.fp21.append(c)
            self.fp61.append(d)
            self.fp41.append(e)
            self.fp51.append(f)
        self.prevtj1 = len(self.tjx1)
        
    def highscore2(self,score,a,b,c,d,e,f):
        
        if score>self.lastscore2 and len(self.g2x)>2:
            self.lastscore2 = score
            self.g2x1, self.g2y1 = self.g2x, self.g2y
            
        for h in range(len(self.tjx2)-self.prevtj2):
            self.freward2.append(score)
            self.fp32.append(a)
            self.fp12.append(b)
            self.fp22.append(c)
            self.fp62.append(d)
            self.fp42.append(e)
            self.fp52.append(f)
        self.prevtj2 = len(self.tjx2)
      
