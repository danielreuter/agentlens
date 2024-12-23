from agentlens.client import Observation, task, use


@task
async def top_level_task():
    obs = use(Observation)
    return obs


@task
async def child_task():
    obs = use(Observation)
    return obs


@task
async def parent_calls_child():
    obs_parent = use(Observation)
    c_obs = await child_task()
    return obs_parent, c_obs


async def test_top_level_observation():
    obs = await top_level_task()
    assert obs.parent is None
    assert obs.children == []


async def test_nested_observations():
    p_obs, c_obs = await parent_calls_child()
    assert c_obs.parent is p_obs
    assert c_obs in p_obs.children


async def test_chained_observations():
    @task
    async def grandchild():
        return use(Observation)

    @task
    async def child():
        c_obs = use(Observation)
        gc_obs = await grandchild()
        return c_obs, gc_obs

    @task
    async def parent():
        p_obs = use(Observation)
        c_obs, gc_obs = await child()
        return p_obs, c_obs, gc_obs

    p, c, gc = await parent()
    assert c.parent is p
    assert gc.parent is c


async def test_multiple_siblings():
    @task
    async def sibling_task():
        return use(Observation)

    @task
    async def parent():
        p_obs = use(Observation)
        s1 = await sibling_task()
        s2 = await sibling_task()
        return p_obs, [s1, s2]

    p_obs, siblings = await parent()
    assert len(p_obs.children) == 2
    assert all(s.parent is p_obs for s in siblings)


async def test_access_observation_self():
    @task
    async def some_task():
        obs = use(Observation)
        return obs.id

    obs_id = await some_task()
    assert obs_id is not None
