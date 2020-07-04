import reilly as rl


def test_sarsa_appr():
    env = rl.Taxi()
    session = rl.PyBindSession(env)
    agent = rl.SemiGradientSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        features=1,
        tilings=2,
        tilings_offset=[1],
        tile_size=[1]
    )
    session.add_agent(agent)
    session.run(100, 10, 10)


def test_expected_sarsa_appr():
    env = rl.Taxi()
    session = rl.PyBindSession(env)
    agent = rl.SemiGradientExpectedSarsa(
        actions=env.actions,
        alpha=0.1,
        epsilon=0.03,
        gamma=0.99,
        features=1,
        tilings=2,
        tilings_offset=[1],
        tile_size=[1]
    )
    session.add_agent(agent)
    session.run(100, 10, 10)
