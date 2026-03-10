#!/usr/bin/env python3
"""Implements SendMessageTool which is used by agents to communicate with each other during an episode"""
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

if TYPE_CHECKING:
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface
    from habitat_llm.llm.base_llm import BaseLLM

from habitat_llm.tools import PerceptionTool, get_prompt


class SendMessageTool(PerceptionTool):
    """
    An ActionTool designed to be used by agents to send messages to each other.
    This tool uses a language model to generate the message based on the context or user input.
    It now also checks if recipient agents are in a Done state (by examining agent_action_history)
    and skips those agents, reporting their Done status in the proxy information.
    """

    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.llm = None
        self.env_interface = None
        self.skill_config = skill_config
        self.prompt_maker = None

        system_feedback_mode = {
            "Opaque": [True, False, False, False],
            "Binary": [False, True, False, False],
            "Causal": [False, True, True, False],
            "Traceable": [False, True, True, True],
        }
        mode = self.skill_config.get("system_feedback_mode", "Opaque")
        if mode not in system_feedback_mode:
            raise ValueError(
                f"Invalid system_feedback_mode: {mode}. "
                f"Valid options are: {list(system_feedback_mode.keys())}"
            )
        opaque, binary, causal, trace = system_feedback_mode[mode]
        self.response_unsure_delivery = opaque
        self.response_success_state = binary
        self.append_failure_reason = causal
        self.append_agent_status = trace

        self.speech_recognition_range = self.skill_config.get(
            "speech_recognition_range", 7.0
        )

        if type(self.speech_recognition_range) not in [int, float]:
            raise ValueError("speech_recognition_range should be a number")
        if self.speech_recognition_range < 0:
            raise ValueError("speech_recognition_range should be positive")
        if self.speech_recognition_range == 0:
            self.speech_recognition_range = float("inf")

        self.name_mapping = self.skill_config.get("name_mapping", {})
        if self.name_mapping == {}:
            self.use_names = False
        else:
            self.use_names = True
            self.name_mapping = {int(k): v for k, v in self.name_mapping.items()}
            # self.name_mapping = {0: "Spot", 1: "Alice"}

    def to(self, device: str):
        return self

    def set_environment(self, env: "EnvironmentInterface") -> None:
        """
        Sets the tool's environment_interface variable.
        """
        self.env_interface = env

    def set_llm(self, llm: "BaseLLM") -> None:
        """
        Sets the tool's LLM interface object.
        """
        self.llm = llm
        self.prompt_maker = get_prompt(self.skill_config.prompt, self.llm.llm_conf)

    def _get_agent_distance(self, agent_uid1: str, agent_uid2: str) -> float:
        """
        Compute the Euclidean distance between two agents based on their base positions.
        """
        agent1 = self.env_interface.sim.agents_mgr[agent_uid1].articulated_agent
        agent2 = self.env_interface.sim.agents_mgr[agent_uid2].articulated_agent
        pos1 = np.array(list(agent1.base_pos))
        pos2 = np.array(list(agent2.base_pos))
        return round(np.linalg.norm(pos1 - pos2), 2)

    def _get_room_for_entity(self, entity_uid: str) -> str:
        """
        Retrieve the room name for the given agent using the dynamic world graph.

        The entity identifier is converted (if necessary) to the expected format (e.g. "agent_0")
        before the agent node is looked up in the world graph.
        Then, the world graph's room-assignment logic is used to obtain the room.
        If no room is found, "unknown" is returned.
        """
        if isinstance(entity_uid, int) or (
            isinstance(entity_uid, str) and not entity_uid.startswith("agent_")
        ):
            agent_name = f"agent_{entity_uid}"
        else:
            agent_name = entity_uid

        try:
            dg = self.env_interface.world_graph[self.env_interface.robot_agent_uid]
        except KeyError:
            return "unknown"

        try:
            agent_node = dg.get_node_from_name(agent_name)
        except ValueError:
            return "unknown"

        room_node = dg.find_room_of_entity(agent_node, verbose=False)
        if room_node is not None:
            return room_node.name
        return "unknown"

    def _is_agent_in_proximity(
        self, sender_uid: str, receiver_uid: str
    ) -> Tuple[bool, tuple]:
        """
        Determine if two agents are in proximity by checking if they are in the same room
        or within 10 meters. Returns a tuple (in_proximity, (distance, room_sender, room_receiver)).
        """
        room_sender = self._get_room_for_entity(sender_uid)
        room_receiver = self._get_room_for_entity(receiver_uid)
        distance = self._get_agent_distance(sender_uid, receiver_uid)
        if (
            room_sender != "unknown"
            and room_receiver != "unknown"
            and room_sender == room_receiver
        ):
            return True, (distance, room_sender, room_receiver)
        if distance <= self.speech_recognition_range:
            return True, (distance, room_sender, room_receiver)
        return False, (distance, room_sender, room_receiver)

    def _build_success_message(self, recipient_info: Dict[int, Dict]) -> str:
        base_message = "Successfully sent message"
        if self.response_success_state:
            agents = [
                f"Agent {uid}"
                if not self.use_names
                else self.name_mapping.get(uid, f"Agent {uid}")
                for uid, info in recipient_info.items()
                if info["message_delivered"]
            ]
            if agents:
                base_message += f": {', '.join(agents)}"
        return base_message

    def _build_failure_message(self, recipient_info: Dict[int, Dict]) -> str:
        failure_message = "Message send failed"
        if self.append_failure_reason:
            failure_message += f" (not in the same room or within {self.speech_recognition_range} meters)"
            if self.append_agent_status:
                details = []
                for uid, info in recipient_info.items():
                    if info["is_done"] and not info["can_wake_up"]:
                        details.append(f"[Agent {uid}: Done]")
                    else:
                        details.append(
                            f"[Agent {uid} in {info['room_receiver']} Distance: {info['distance']}m]"
                        )
                failure_message += " " + " ".join(details)
        return failure_message

    def process_action(self, input_query: str, observations: dict) -> Tuple[None, str]:
        sender_id = observations.get("agent_id", None)
        if sender_id is None:
            raise ValueError("Sender agent id not provided in observations")
        if not self.env_interface:
            raise ValueError("Environment Interface not set")

        if hasattr(self.env_interface, "agent_action_history"):
            all_agent_ids = self.env_interface.agent_action_history.keys()
        else:
            raise ValueError("Unable to retrieve agent IDs from the environment.")

        if not (isinstance(input_query, str) and len(input_query.strip()) > 0):
            if self.append_failure_reason:
                return (
                    None,
                    "Message send failed: empty message field (ex. SendMessageTool[Message_content])",
                )
            return None, "Message send failed: empty message field"
        message = input_query.strip()

        all_agent_ids = self.env_interface.agent_action_history.keys()
        recipient_info = dict()
        any_message_delivered = False

        for uid in all_agent_ids:
            if uid == sender_id:
                continue

            is_in_proximity, prox_details = self._is_agent_in_proximity(sender_id, uid)
            last_action = self.env_interface.agent_action_history[uid][-1]
            is_done = "Done" in last_action.action
            can_wake_up = (
                self.env_interface.can_wake_up_agent(uid) if is_done else False
            )
            wake_up_triggered = False
            message_delivered = False

            if is_in_proximity:
                if not is_done:
                    self.env_interface.send_message_to_agent(sender_id, uid, message)
                    message_delivered = True
                    any_message_delivered = True
                elif can_wake_up:
                    self.env_interface.send_message_to_agent(sender_id, uid, message)
                    self.env_interface.wake_up_agent(uid)
                    message_delivered = True
                    any_message_delivered = True
                    wake_up_triggered = True

            recipient_info[uid] = {
                "proximity": is_in_proximity,
                "is_done": is_done,
                "can_wake_up": can_wake_up,
                "wake_up_triggered": wake_up_triggered,
                "message_delivered": message_delivered,
                "distance": prox_details[0],
                "room_sender": prox_details[1],
                "room_receiver": prox_details[2],
            }

        if self.response_unsure_delivery:
            return (
                None,
                "Message sent done, but it's unclear if the other agent received it.",
            )
        else:
            if any_message_delivered:
                return None, self._build_success_message(recipient_info)
            else:
                return None, self._build_failure_message(recipient_info)

    @property
    def description(self) -> str:
        """
        Returns the description of this tool as provided in configuration.
        """
        return self.skill_config.description

    def process_high_level_action(
        self, input_query: str, observations: dict
    ) -> Tuple[None, str]:
        """
        Placeholder method to satisfy abstract method requirement.
        """
        return self.process_action(input_query, observations)

    @property
    def argument_types(self) -> list:
        """
        Returns the types of arguments required for this tool.
        """
        return ["message_topic_or_content"]
