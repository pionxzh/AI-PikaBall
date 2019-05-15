import serpent.cv
import serpent.utilities

import os
import gc
import re
import sys
import math
import time
import cv2
import itertools
import collections
import numpy as np

import skimage.io
import skimage.color
import skimage.draw
import skimage.feature
import skimage.filters

from colorama import init
from colorama import Fore, Style
from datetime import datetime
from serpent.game_agent import GameAgent
from serpent.game_frame import GameFrame
from serpent.frame_grabber import FrameGrabber
from serpent.sprite_locator import SpriteLocator
from serpent.input_controller import KeyboardKey
from serpent.frame_transformer import FrameTransformer
from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace

from .helpers.memory import readInfo
from .helpers.loading import playAnimation

init()
#sprite_locator = SpriteLocator()

class SerpentPikaBallGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.previous_game_frame = None

        self.lowerY = np.array([255, 255, 0], np.uint8)
        self.upperY = np.array([255, 255, 10], np.uint8)
        self.lowerR = np.array([255, 0, 0], np.uint8)
        self.upperR = np.array([255, 0, 10], np.uint8)

        self.game_state = None
        self._reset_game_state()

    def setup_key(self):
        self.input_mapping = {
            "JUMP": [KeyboardKey.KEY_UP],
            "RIGHT": [KeyboardKey.KEY_RIGHT],
            "LEFT": [KeyboardKey.KEY_LEFT],
            "UP_HIT": [KeyboardKey.KEY_UP, KeyboardKey.KEY_RETURN],
            "L_HIT": [KeyboardKey.KEY_LEFT, KeyboardKey.KEY_RETURN],
            "DOWN_HIT": [KeyboardKey.KEY_DOWN, KeyboardKey.KEY_RETURN],
            "NONE": []
        }

        self.key_mapping = {
            KeyboardKey.KEY_UP: "UP",
            KeyboardKey.KEY_RIGHT: "RIGHT",
            KeyboardKey.KEY_DOWN: "DOWN",
            KeyboardKey.KEY_LEFT: "LEFT",
            KeyboardKey.KEY_RETURN: "HIT"
        }
        self.action_space = KeyboardMouseActionSpace(
            action=['JUMP', 'RIGHT', 'LEFT', 'UP_HIT', 'L_HIT', 'DOWN_HIT', 'NONE']
        )

        move_inputs = {
            "JUMP": [KeyboardKey.KEY_UP],
            "RIGHT": [KeyboardKey.KEY_RIGHT],
            "LEFT": [KeyboardKey.KEY_LEFT],
            "NO_MOVE": []
        }
        attack_inputs  = {
            "Power Hit": [KeyboardKey.KEY_RETURN],
            "NO_HIT": []
        }
        self.game_inputs = dict()
        for move_label, attack_label in itertools.product(move_inputs, attack_inputs):
            label = f"{move_label.ljust(10)}{attack_label}"
            self.game_inputs[label] = move_inputs[move_label] + attack_inputs[attack_label]
        print(self.game_inputs)



    def setup_play(self):
        #self.cid = 0
        self.trainID = 0
        self.setup_key()
        self.frame_process = False
        self.rewards = list()
        self.started_at = datetime.now()
        self.started_at_str = self.started_at.isoformat()

        latest_epsilon = 1
        model_file_path = 'fighting_movement_dqn_0_1_.h5'
        model_list = os.listdir('model')
        for item in model_list:
            for epsilon in re.findall("\d+\.\d+", item):
                if latest_epsilon > float(epsilon):
                    latest_epsilon = float(epsilon)
                    model_file_path = item
        
        model_file_path = f'model/{model_file_path}'.replace('/', os.sep)
        print(">> LOAD MODEL: ", model_file_path)
        time.sleep(1)

        self.dqn_action = DDQN(
            model_file_path=model_file_path if os.path.isfile(model_file_path) else None,
            input_shape=(114, 162, 4),
            input_mapping=self.input_mapping,
            action_space=self.action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=100 if os.path.isfile(model_file_path) else 1000 ,
            batch_size=32,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=True
        )
        print('Starting Game')
        self.input_controller.tap_key(KeyboardKey.KEY_RETURN)

    def getDifference(self, game_frame, previous_game_frame):
        return game_frame.grayscale_frame - previous_game_frame.grayscale_frame

    # old img process way
    def handle_frame_process(self, game_frame):
        '''
        if not self.frame_process:
            return
        if self.game_frame_buffer.previous_game_frame is not None:
            try:
                threshold = skimage.filters.threshold_otsu(game_frame.grayscale_frame)
            except ValueError:
                threshold = -1

            gray_frame = game_frame.grayscale_frame > threshold
            gray_frame = skimage.filters.gaussian(gray_frame)
            gray_cv_frame = cv2.cvtColor(np.array(gray_frame * 255, dtype='uint8'), cv2.COLOR_GRAY2RGB)

            cv_frame = cv2.cvtColor(np.asarray(game_frame.frame, dtype='uint8'), cv2.COLOR_BGR2RGB)
            cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            frame_threshed = cv2.inRange(cv_frame, self.lowerY, self.upperY)

            # find connected components (pikachu)
            _, cnts, hierarchy, = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(cnts) > 0:
                cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
                for i in range(0, 2):
                    # Draw a rectangular frame around the detected object
                    x, y, w, h = cv2.boundingRect(cnts[i])
                    cv2.rectangle(gray_cv_frame, (x,y), (x+w,y+h), (0,255,0), 2)

            frame_threshed = cv2.inRange(cv_frame, self.lowerR, self.upperR)

            # find connected components (ball)
            _, cnts, hierarchy, = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(cnts) > 0:
                cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
                if len(cnt) > 70:
                    (x,y), radius = cv2.minEnclosingCircle(cnt)
                    cv2.circle(gray_cv_frame, (int(x),int(y)), int(radius), (255,0,0),2)

            self.visual_debugger.store_image_data(
                gray_cv_frame,
                gray_cv_frame.shape,
                "grayscale"
            )
            
            # send difference to debugger (optional)
            frame_difference = self.getDifference(game_frame, self.game_frame_buffer.previous_game_frame)
            bw_frame_difference = frame_difference > 100
            bw_frame_difference = skimage.filters.sobel(bw_frame_difference)
            self.visual_debugger.store_image_data(
                np.array(bw_frame_difference * 255, dtype='uint8'),
                bw_frame_difference.shape,
                "frame_diff"
            )
            '''

    def handle_play(self, game_frame):
        # locate sprite position and existence
        '''
        logo_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_LOGO'], game_frame=game_frame)
        menu_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_MENU'], game_frame=game_frame)
        game_set_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_GAME_SET'], game_frame=game_frame)
        '''

        # append memory data into game state
        (self.game_state["com_x"], self.game_state["com_y"], self.ai_x, self.ai_y, self.ball_x, self.ball_y, self.com_sc, self.ai_sc, self.col_size, self.game_state["col_x"], self.game_state["col_y"]) = readInfo()
        self.game_state["ai_x"].appendleft(self.ai_x)
        self.game_state["ai_y"].appendleft(self.ai_y)
        self.game_state["ball_x"].appendleft(self.ball_x)
        self.game_state["ball_y"].appendleft(self.ball_y)
        self.game_state["ai_score"].appendleft(self.ai_sc)
        self.game_state["com_score"].appendleft(self.com_sc)
        self.game_state["col_size"].appendleft(self.col_size)

        self.handle_frame_process(game_frame)
        '''
        if(logo_locator):
            print('Entering Logo...')
            self.game_state["playing"] = False
            self.handle_menu()
        elif (menu_locator):
            print('Entering Menu...')
            self.game_state["playing"] = False
            self.handle_menu()
        elif (game_set_locator):
            print('Game Set!')
            self.handle_fight_end(game_frame)
        '''

        # judge is-in-game by read pixel value (tricky)
        self.game_frame_img = FrameGrabber.get_frames([0], frame_type="PIPELINE").frames[0].frame
        if self.game_frame_img[100, 81] != 0.7137254901960784:
            self.handle_notInGame()
        else:
            self.game_state["playing"] = True
            self.handle_fight(game_frame)

    def handle_notInGame(self):
        serpent.utilities.clear_terminal()
        print('Currently not in game...please wait..')

        playAnimation(self.game_state["animeIndex"])
        self.game_state["animeIndex"] = self.game_state["animeIndex"] + 1 if self.game_state["animeIndex"] < 3 else 0

        #print(self.game_frame_img[95:105,80:83])

        self.input_controller.tap_key(KeyboardKey.KEY_RETURN)
        time.sleep(0.3)

    def handle_menu(self):
        self.input_controller.tap_key(KeyboardKey.KEY_RETURN)
        time.sleep(4)

    def handle_fight(self, game_frame):
        gc.disable()
        if self.dqn_action.first_run:
            self.dqn_action.first_run = False
            return

        if self.dqn_action.frame_stack is None:
            game_frame_buffer = FrameGrabber.get_frames([0], frame_type="PIPELINE").frames[0]
            self.dqn_action.build_frame_stack(game_frame_buffer.frame)
        else:
            # saving frame pic to analyze
            #self.cid = self.cid + 1
            #game_frame_img = FrameGrabber.get_frames([0], frame_type="PIPELINE").frames[0]
            #skimage.io.imsave(f"frame{self.cid}.png", game_frame_img.frame)
            game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")

            if self.dqn_action.mode == "TRAIN":
                reward = self._calculate_reward()

                self.game_state["reward"] += reward

                self.dqn_action.append_to_replay_memory(
                    game_frame_buffer,
                    reward,
                    terminal=self.game_state["ai_score"][0] == 15
                )

                # Every 2000 steps, save latest weights to disk
                if self.dqn_action.current_step % 1000 == 0:
                    self.dqn_action.save_model_weights(
                        file_path_prefix=f"model/fighting_movement"
                    )

                # Every 20000 steps, save weights checkpoint to disk
                if self.dqn_action.current_step % 10000 == 0:
                    self.dqn_action.save_model_weights(
                        file_path_prefix=f"model/fighting_movement",
                        is_checkpoint=True
                    )
            elif self.dqn_action.mode == "RUN":
                self.dqn_action.update_frame_stack(game_frame_buffer)

            run_time = datetime.now() - self.started_at
            serpent.utilities.clear_terminal()
            print('')
            print(Fore.YELLOW)
            print(Style.BRIGHT)
            print(f"STARTED AT:{self.started_at_str}")
            print(f"RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} s")

            print(Style.RESET_ALL)
            #print("")
            print(Fore.GREEN)
            print(Style.BRIGHT)
            print("MOVEMENT NEURAL NETWORK:\n")
            self.dqn_action.output_step_data()
            print(Style.RESET_ALL)
            print(Style.BRIGHT)
            print(f"CURRENT RUN: {self.game_state['current_run'] }")
            print("")
            print(f"CURRENT RUN   REWARD: {round(self.game_state['reward'], 4)}")
            print(f"CURRENT AI    SCORE: {self.game_state['ai_score'][0]}")
            print(f"CURRENT ENEMY SCORE: {self.game_state['com_score'][0]}")
            print("")
            print(f"PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
            print(Style.RESET_ALL)

            self.dqn_action.pick_action()
            self.dqn_action.generate_action()

            movement_keys = self.dqn_action.get_input_values()

            print("")
            print(Fore.GREEN)
            print(Style.BRIGHT)
            #print(movement_keys)
            print("" + " + ".join(list(map(lambda k: self.key_mapping.get(k), movement_keys))))
            print(Style.RESET_ALL)
            print("")
            print(f"AI:        ({self.game_state['ai_x'][0]}, {self.game_state['ai_y'][0]})")
            print(f"COM:       ({self.game_state['com_x']}, {self.game_state['com_y']})")
            print(f"BALL:      ({self.game_state['ball_x'][0]}, {self.game_state['ball_y'][0]})")
            print(f"Collision: ({self.game_state['col_x']}, {self.game_state['col_y']}, {self.game_state['col_size'][0]})")
            print(f"Distance:   {self.game_state['distance'][0]}")

            self.input_controller.handle_keys(movement_keys)


            if self.dqn_action.current_action_type == "PREDICTED":
                self.game_state["run_predicted_actions"] += 1

            self.dqn_action.erode_epsilon(factor=2)
            self.dqn_action.next_step()

            self.game_state["current_run"] += 1

            if self.game_state['ai_score'][0] == 15 or self.game_state['com_score'][0] == 15:
                # Game over
                self.game_state["ai_score"].appendleft(0)
                self.game_state["com_score"].appendleft(0)
                self.handle_fight_end(game_frame)

    def handle_fight_end(self, game_frame):
        self.game_state["playing"] = False
        self.input_controller.handle_keys([])
        self.game_state["current_run"] += 1
        self.handle_fight_training(game_frame)

    def handle_fight_training(self, game_frame):
        #self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        serpent.utilities.clear_terminal()
        gc.enable()
        gc.collect()
        gc.disable()
        print("TRAIN MODE")
        self.input_controller.handle_keys([])

        if self.dqn_action.mode == "TRAIN":
            for i in range(16):
                serpent.utilities.clear_terminal()
                print("")
                print(Fore.GREEN)
                print(Style.BRIGHT)
                print(f"TRAINING ON MINI-BATCHES: {i + 1}/16")
                print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 25 == 0 else ''}")
                print(Style.RESET_ALL)

                self.dqn_action.train_on_mini_batch()

        self.game_state["run_predicted_actions"] = 0

        if self.dqn_action.mode in ["TRAIN", "RUN"]:
            if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                self.dqn_action.update_target_model()

            if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                self.dqn_action.enter_run_mode()
            else:
                self.dqn_action.enter_train_mode()
                
        self.input_controller.tap_key(KeyboardKey.KEY_RETURN)
        time.sleep(2)

    def _reset_game_state(self):
        self.game_state = {
            "reward": 0,
            "animeIndex": 0,
            "current_run": 1,
            "playing": False,
            "run_predicted_actions": 0,
            "ai_x": collections.deque(np.full((4,), 0), maxlen=4),
            "ai_y": collections.deque(np.full((4,), 0), maxlen=4),
            "ai_score": collections.deque(np.full((4,), 0), maxlen=4),
            "ball_x": collections.deque(np.full((4,), 0), maxlen=4),
            "ball_y": collections.deque(np.full((4,), 0), maxlen=4),
            "com_score": collections.deque(np.full((4,), 0), maxlen=4),
            "col_size": collections.deque(np.full((4,), 6), maxlen=4),
            "com_x": 36,
            "com_y": 244,
            "col_x": 0,
            "col_y": 0,
            "distance": collections.deque(np.full((20,), 100), maxlen=20),
        }

    def _calculate_reward(self):
        reward = 0
        distance = math.sqrt(abs(self.game_state["ai_x"][0] - self.game_state["ball_x"][0])**2 + abs(self.game_state["ai_y"][0] - self.game_state["ball_y"][0])**2)
        self.game_state["distance"].appendleft(int(distance))

        # collision with ball
        collision = self.game_state["distance"][0] < 80 and self.game_state["distance"][1] < 80 and self.game_state["distance"][2] < 80 and self.game_state["distance"][0] > self.game_state["distance"][1] and self.game_state["distance"][1] < self.game_state["distance"][2]
        if collision:
            reward += 0.25
        

        # power hit
        if self.game_state["col_size"][0] > 0 and self.game_state["distance"][0] < 90 and self.game_state["col_y"] != 272:
            reward += 0.5

        # AI gain score
        if self.game_state["ai_score"][0] > self.game_state["ai_score"][1]:
            reward += 1

        # Com gain score
        if self.game_state["com_score"][0] > self.game_state["com_score"][1]:
            reward += -1

        if reward > 1:
            reward = 1

        self.game_state["reward"] = reward
        return reward
