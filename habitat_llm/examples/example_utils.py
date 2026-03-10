#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from habitat_llm.agent.env import EnvironmentInterface


def pil_get_font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


class DebugVideoUtil:
    """
    This class provides an interface wrapper for creating, saving, and viewing third person videos of individual skill runs using the EnvironmentInterface API.

    For example, see `execute_skill` function below.
    NOTE: This code was largely adapted from the evaluation_runner.py
    """

    def __init__(
        self, env_interface_arg: EnvironmentInterface, output_dir: str
    ) -> None:
        """
        Construct the DebugVideoUtil instance from an EnvironmentInterface.

        :param env_interface_arg: The EnvironmentInterface instance.
        :param output_dir: The desired directory for saving output frames and videos.
        """

        self.env_interface = env_interface_arg

        # Declare container to store frames used for generating video
        self.frames: List[Any] = []

        self.output_dir = output_dir

        self.num_agents = 0
        for _agent_conf in self.env_interface.conf.evaluation.agents.values():
            self.num_agents += 1

        self.previous_action: DefaultDict[int, List[Tuple[str, Any]]] = defaultdict(
            list
        )
        self.dialogue: DefaultDict[int, List[Any]] = defaultdict(list)
        # self.current_instruction = (
        #     self.env_interface.env.env.env._env.current_episode.instruction
        # )

    def __get_combined_frames(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        For each agent, extract the observation from the "third_rgb" sensor and merge them into a single split-screen image.

        :param batch: A dict mapping observation names to values.
        :return: The composite image as a numpy array.
        """
        # Extract first agent frame
        images = []
        for obs_name, obs_value in batch.items():
            if "third_rgb" in obs_name:
                if self.num_agents == 1:
                    if "0" in obs_name or "main_agent" in obs_name:
                        images.append(obs_value)
                else:
                    images.append(obs_value)

        # Extract dimensions of the first image
        height, width = images[0].shape[1:3]

        # Create an empty canvas to hold the concatenated images
        concat_image = np.zeros((height, width * len(images), 3), dtype=np.uint8)

        # Iterate through the images and concatenate them horizontally
        for i, image in enumerate(images):
            concat_image[:, i * width : (i + 1) * width] = image.cpu()

        return concat_image

    def _store_for_video(
        self, observations: Dict[str, Any], hl_actions: Dict[int, Any]
    ) -> None:
        """
        Store a video with observations and text from an observation dict and an agent to action metadata dict.
        NOTE: Could probably go into utils?

        :param observations: A dict mapping observation names to values.
        :param hl_actions: A dict mapping agent action indices to actions.
        """
        frames_concat = self.__get_combined_frames(observations)
        frames_concat = np.ascontiguousarray(frames_concat).copy()
        H, W = frames_concat.shape[:2]

        pil_img = Image.fromarray(frames_concat).convert("RGBA")
        draw = ImageDraw.Draw(pil_img)
        font = pil_get_font(22)
        white = (255, 255, 255, 255)

        def normalize_for_draw(s: str) -> str:
            s = (
                s.replace("\\'", "'")
                .replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\r", "\r")
            )
            replacements = {
                "\u2018": "'",
                "\u2019": "'",
                "\u201B": "'",
                "\u2032": "'",
                "\u201C": '"',
                "\u201D": '"',
                "\u2033": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
                "\u00A0": " ",
                "\u2026": "...",
            }
            for k, v in replacements.items():
                s = s.replace(k, v)
            return s

        def text_size(text: str) -> tuple[int, int]:
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return r - l, b - t

        def format_action_text(action: tuple, prev_action=None) -> str:
            """
            Returns a readable text like:
            'Explore the room kitchen_1 (prev: Navigate to hallway_1)'
            """
            if not action or not isinstance(action, (tuple, list)):
                return str(action)

            name = action[0]
            args: List[Any] = []
            for a in action[1:]:
                if isinstance(a, (tuple, list)):
                    args.extend(x for x in a if x not in (None, "", "None"))
                elif a not in (None, "", "None"):
                    args.append(a)

            def fmt(name, args):
                templates = {
                    "Clean": lambda a: f"Clean {a[0]}" if a else "Clean",
                    "Close": lambda a: f"Close {a[0]}" if a else "Close something",
                    "Explore": lambda a: f"Explore the room {a[0]}"
                    if a
                    else "Explore a room",
                    "Fill": lambda a: f"Fill {a[0]}" if a else "Fill something",
                    "Navigate": lambda a: f"Navigate to {a[0]}" if a else "Navigate",
                    "Open": lambda a: f"Open {a[0]}" if a else "Open something",
                    "Pick": lambda a: f"Pick up {a[0]}" if a else "Pick up something",
                    "Place": lambda a: (
                        f"Place {a[0]} {a[1]} {a[2]}"
                        if len(a) >= 3
                        else f"Place {a[0]}"
                    ),
                    "Pour": lambda a: f"Pour into {a[0]}" if a else "Pour",
                    "PowerOff": lambda a: f"Turn off {a[0]}" if a else "Turn off",
                    "PowerOn": lambda a: f"Turn on {a[0]}" if a else "Turn on",
                    "Rearrange": lambda a: (
                        f"Rearrange {a[0]} {a[1]} {a[2]}"
                        if len(a) >= 3
                        else f"Rearrange {a[0]}"
                    ),
                    "SendMessageTool": lambda a: "Send message",
                    "Wait": lambda a: "Wait",
                    "Done": lambda a: "Done",
                }
                if name in templates:
                    try:
                        return templates[name](args)
                    except Exception:
                        return f"{name} " + " ".join(args)
                return f"{name} " + " ".join(args)

            current_text = fmt(name, args)

            if prev_action and isinstance(prev_action, (tuple, list)):
                prev_name = prev_action[0]
                prev_args = [a for a in prev_action[1:] if a not in (None, "", "None")]
                prev_text = fmt(prev_name, prev_args)
                current_text += f" (prev: {prev_text})"

            return current_text

        for idx, action in hl_actions.items():
            if not action:
                continue
            name = action[0] if isinstance(action, (tuple, list)) and action else None
            if name in (None, "", "None", "SyntaxError"):
                continue
            if name == "SendMessageTool":
                if len(action) > 1 and action[1] not in (None, "", "None"):
                    self.dialogue[idx].append(action[1])
                if (
                    not self.previous_action[idx]
                    or self.previous_action[idx][-1] != action
                ):
                    self.previous_action[idx].append(action)
                continue
            if not self.previous_action[idx] or self.previous_action[idx][-1] != action:
                self.previous_action[idx].append(action)

        for idx in range(self.num_agents):
            if not self.previous_action[idx]:
                continue
            agent_name = "Human" if str(idx) == "1" else "Robot"
            last_action = self.previous_action[idx][-1]
            prev_action = (
                self.previous_action[idx][-2]
                if len(self.previous_action[idx]) > 1
                else None
            )
            readable = format_action_text(last_action, prev_action)
            text = f"{agent_name}: {readable}"
            draw.text(
                (20, (int(idx) + 1) * 50),
                normalize_for_draw(text),
                fill=white,
                font=font,
            )

        col_margin = 20
        col_w = W // 2 - col_margin * 2
        line_gap = 8
        box_h = int(H * 0.28)
        base_y = H - box_h + 30

        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle(
            (0, H - box_h, W, H), fill=(0, 0, 0, int(255 * 0.35))
        )
        pil_img = Image.alpha_composite(pil_img, overlay)
        draw = ImageDraw.Draw(pil_img)

        def place_message(
            header: str, msg: str, x_left: int, base_y: int, max_col_w: int
        ):
            header = normalize_for_draw(header)
            msg = normalize_for_draw(msg)

            draw.text((x_left, base_y), header, fill=white, font=font)
            hd_w, hd_h = text_size(header)

            first_line_x = x_left + hd_w + 8
            indent_px = 28
            max_text_width = max_col_w - (first_line_x - x_left)

            words = msg.split(" ")
            cur = ""
            lines = []
            for w in words:
                test = w if not cur else cur + " " + w
                tw, th = text_size(test)
                if tw <= max_text_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)

            last_y = base_y
            for i, line in enumerate(lines):
                x = first_line_x if i == 0 else x_left + indent_px
                draw.text((x, last_y), line, fill=white, font=font)
                _, th = text_size(line)
                last_y += th + line_gap
            return last_y

        robot_msgs = self.dialogue.get(0, []) or self.dialogue.get(2, [])
        if robot_msgs:
            latest_msg = robot_msgs[-1]
            count = len(robot_msgs)
            place_message(f"Robot[{count}]:", latest_msg, col_margin, base_y, col_w)

        human_msgs = self.dialogue.get(1, [])
        if human_msgs:
            latest_msg = human_msgs[-1]
            count = len(human_msgs)
            place_message(
                f"Human[{count}]:", latest_msg, W // 2 + col_margin, base_y, col_w
            )

        pil_img = pil_img.convert("RGB")
        frames_concat = np.asarray(pil_img)
        frames_concat = np.ascontiguousarray(frames_concat).copy()
        self.frames.append(frames_concat)
        return

    def _make_video(self, play: bool = True, postfix: str = "") -> None:
        """
        Makes a video from a pre-processed set of frames using imageio and saves it to the output directory.

        :param play: Whether or not to play the video immediately.
        :param postfix: An optional postfix for the video file name.
        """
        out_file = f"{self.output_dir}/videos/video-{postfix}.mp4"
        print(f"Saving video to {out_file}")
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
        writer = imageio.get_writer(
            out_file,
            fps=30,
            quality=4,
        )
        for frame in self.frames:
            writer.append_data(frame)

        writer.close()
        if play:
            print("     ...playing video, press 'q' to continue...")
            self.play_video(out_file)

    def clear(self) -> None:
        """
        Clear the frames and previous action data.
        """
        self.frames = []
        self.previous_action = defaultdict(list)
        self.dialogue = defaultdict(list)

    def play_video(self, filename: str) -> None:
        """
        Play and loop video from a filepath with cv2.

        :param filename: The filepath of the video.
        """
        cap = cv2.VideoCapture(filename)
        last_time = time.time()
        while cap.isOpened():
            if time.time() - last_time > 1.0 / 30:
                last_time = time.time()
                ret, frame = cap.read()
                # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

                if ret:
                    cv2.imshow("Image", frame)
                else:
                    # looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def execute_skill(
    high_level_skill_actions: Dict[Any, Any],
    llm_env,
    make_video: bool = True,
    vid_postfix: str = "",
    play_video: bool = True,
) -> Tuple[Dict[Any, Any], Dict[Any, Any], List[Any]]:
    """
    Execute a high-level skill from a string (e.g. as produced by the planner).
    Can create and display a video of the running skill.

    :param high_level_skill_actions: The map of agent indices to actions. TODO: typing
    :param llm_env: The planner instance. TODO: typing
    :param make_video: whether or not to create, save, and display a video of the skill.
    :param vid_postfix: An optional postfix for the video file. For example, the action name.
    :param play_video: Whether or not to immediately play the generated video.
    :return: A tuple with two dict(the first contains responses per-agent skill, the second contains the number of skill steps taken) and a list of frames.
    """
    dvu = DebugVideoUtil(
        llm_env.env_interface, llm_env.env_interface.conf.paths.results_dir
    )

    # Get the env observations
    observations = llm_env.env_interface.get_observations()
    agent_idx = list(high_level_skill_actions.keys())[0]
    skill_name = high_level_skill_actions[agent_idx][0]

    # Set up the variables
    skill_steps = 0
    max_skill_steps = 1500
    skill_done = None

    # While loop for executing skills
    while not skill_done:
        # Check if the maximum number of steps is reached
        assert (
            skill_steps < max_skill_steps
        ), f"Maximum number of steps reached: {skill_name} skill fails."

        # Get low level actions and responses
        low_level_actions, responses = llm_env.process_high_level_actions(
            high_level_skill_actions, observations
        )

        assert (
            len(low_level_actions) > 0
        ), f"No low level actions returned. Response: {responses.values()}"

        # Check if the agent finishes
        if any(responses.values()):
            skill_done = True

        # Get the observations
        obs, reward, done, info = llm_env.env_interface.step(low_level_actions)
        observations = llm_env.env_interface.parse_observations(obs)

        if make_video:
            dvu._store_for_video(observations, high_level_skill_actions)

        # Increase steps
        skill_steps += 1

    if make_video and skill_steps > 1:
        dvu._make_video(postfix=vid_postfix, play=play_video)

    return responses, {"skill_steps": skill_steps}, dvu.frames
