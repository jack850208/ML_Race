

"""
The template of the script for the machine learning process in game pingpong
"""


import pickle
import numpy as np


ball_position_history = []

filename1=r"F:\python\MLGame\games\pingpong\ml\mlp_model_2P.sav"
model = pickle.load(open(filename1, 'rb'))
print(model)

class MLPlay:
    def __init__(self, side):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.ball_served = False
        self.side = side

        
    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            return "SERVE_TO_LEFT"
        else:
      
            ball_position_history.append(scene_info['ball'])
            platform_center_x = scene_info['platform_1P'][0] + 20
            
            if (len(ball_position_history)) == 1 :
                ball_going_down = 0
                vx=7
                vy=7
            elif ball_position_history [-1][1] - ball_position_history [-2][1] > 0:
                ball_going_down = 1
                vy = ball_position_history[-1][1] - ball_position_history[-2][1]
                vx = ball_position_history[-1][0] - ball_position_history[-2][0]
                
            else:
                ball_going_down = 0
                vy = ball_position_history[-1][1] - ball_position_history[-2][1]
                vx = ball_position_history[-1][0] - ball_position_history[-2][0]
            
            
    
            # 3.3 Put the code here to handle the scene information
    
            inp_temp=np.array([scene_info['ball'][0], scene_info['ball'][1],scene_info['platform_2P'][0] + 20,scene_info['platform_2P'][1] + 30,
                               vx, vy])
            input=inp_temp[np.newaxis, :]
            #print(input)
            if(len(ball_position_history) > 1):
                move=model.predict(input)
            else:
                move = 0
            #print(move)
    #"""            
            # 3.4. Send the instruction for this frame to the game process
            #ball_destination = ball_destination - ball_destination%5
    #        print(ball_destination)
    
    
            if (move > 0.3) :
                 command = "MOVE_RIGHT"
            elif (move < -0.3)  :
                 command = "MOVE_LEFT"
            else:
                 command = "NONE"
                 
            return command


    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False




