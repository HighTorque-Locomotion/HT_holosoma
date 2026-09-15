"""PiPlus-S WBT observations."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg
from holosoma.config_values.wbt.g1.observation import actor_obs_shared, critic_obs_shared_terms

piplus_s_wbt_observation = ObservationManagerCfg(groups={
    "actor_obs": actor_obs_shared,
    "critic_obs": ObsGroupCfg(concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_shared_terms),
})
