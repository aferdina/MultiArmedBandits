import pytest
from multiarmedbandits.environments import BaseBanditEnv


@pytest.mark.parametrize(
    "env",
    [pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("testbed_env")],
)
def test_mab_envs(env: BaseBanditEnv) -> None:
    for play_action in range(2):
        _, get_reward, _, _ = env.step(play_action)
        
