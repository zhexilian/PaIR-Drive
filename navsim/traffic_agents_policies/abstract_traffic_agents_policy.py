from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.metric_caching.metric_cache import MetricCache


class AbstractTrafficAgentsPolicy(ABC):
    """Interface for background traffic agents in NAVSIM."""

    @abstractmethod
    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        pass

    @abstractmethod
    def get_list_of_simulated_object_types(self) -> List[TrackedObjectType]:
        """
        Returns the list of object types that the policy simulates.
        For all remaining objects, the ground truth future tracks are used.
        The policy should only return the tracks for the object types it simulates.
        The remaining objects are automatically merged to the DetectionsTracks.
        """

    @abstractmethod
    def simulate_environment(
        self, simulated_ego_states: npt.NDArray[np.float64], metric_cache: MetricCache
    ) -> List[DetectionsTracks]:
        """Return environment tracks for every simulated ego state."""

    @abstractmethod
    def simulate_traffic_agents(
        self, simulated_ego_states: npt.NDArray[np.float64], metric_cache: MetricCache
    ) -> List[DetectionsTracks]:
        """
        Simulates the (reactive) behavior of traffic agents,
        given that the ego agent follows the trajectory provided.
            :param simulated_ego_states: trajectory the ego-vehicle will follow
            :param metric_cache: general metric cache with describing the state of all agents and their environment
            :return: DetectionsTracks object containing the simulated traffic agents
        """
