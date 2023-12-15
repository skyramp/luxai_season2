import ujson
import os
from typing import List, Tuple
from luxai_s2.map.board import BoardStateDict
import numpy as np
from dataclasses import dataclass, asdict
import copy
from typing import Dict, List, Any

@dataclass
class Replay:
    meta: dict
    steps: List[dict]
    player_names: list

def load_replay(fname, fname_meta, max_steps, aq_limit):
    with open(fname) as in_f:
        replay = ujson.load(in_f)
    with open(fname_meta) as in_f:
        meta = ujson.load(in_f)
    
    
    board = None
    last_step = None

    replay_steps = []

    for step_i, step_view in enumerate(replay["steps"][:max_steps]):
        step_obs = ujson.loads(step_view[0]['observation']['obs'])

        if step_i > 0:
            assert last_step
            last_step['actions'] = {}

            for pid in range(2):
                # rem_obs = {k:v for k, v in step_view[pid]['observation'].items() if k!='obs'}
                # print(rem_obs)
                player_action = step_view[pid]["action"] or {}
                for k in player_action:
                    if k.startswith("unit_"):
                        player_action[k] = player_action[k][:aq_limit]
                player_info = step_view[pid]["info"]
                last_step['actions'][f"player_{pid}"] = player_action

            replay_steps += [last_step]
            last_step = None

        teams = step_obs["teams"]
        units = step_obs["units"]
        for pn, udict in units.items():
            for uname, u in udict.items():
                u['action_queue'] = u['action_queue'][:aq_limit]
        factories = step_obs["factories"]

        if step_i == 0:
            # we have initial board
            bs = step_obs['board']
            assert board is None
            board = dict( rubble=np.array(bs["rubble"], dtype=np.int8),
                                    ore = np.array(bs["ore"], dtype=np.int8),
                                    ice = np.array(bs["ice"], dtype=np.int8),
                                    lichen = np.array(bs["lichen"], dtype=np.int8),
                                    lichen_strains = np.array(bs["lichen_strains"], dtype=np.int8),
                                    valid_spawns_mask = np.array(bs["valid_spawns_mask"], dtype=np.int8),
                                    factories_per_team = bs["factories_per_team"])

        else:
            bs = step_obs['board']
            for k, v in bs.items():
                if isinstance(v, list):
                    # print(f"raw {k}")
                    board[k] = np.array(v, dtype=np.int8)
                elif isinstance(v, dict):
                    # print(f"delta {k}")
                    # print(v)
                    for p_str, x in v.items():
                        a = p_str.split(",")
                        board[k][int(a[0]),int(a[1])] = x
                else:
                    board[k] = v
        
        real_env_steps = step_i - (board['factories_per_team'] * 2 + 1)
        
        if step_i == 0:
            phase = 0
        elif real_env_steps < 0:
            phase = 1
        else:
            phase = 2
        
        last_step = dict( 
            step = step_i,
            real_env_steps = real_env_steps,
            ep_id = meta['id'],
            phase = phase,
            board = copy.deepcopy(board),
            teams = teams,
            units = units,
            factories = factories,
            actions = None,
            player_names = replay["info"]['TeamNames'],
            subm_ids = [z["submissionId"] for z in meta['agents']],
            factories_per_team = board["factories_per_team"]
        )
    
    return Replay(meta=meta, steps=replay_steps, player_names = replay["info"]['TeamNames'])