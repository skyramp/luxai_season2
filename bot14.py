import copy
import json
import math
import random
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from luxai_s2 import LuxAI_S2

from lux.factory import Factory
from lux.unit import Unit, move_deltas

np_dtype = np.float32
torch_dtype = torch.float32
amp_dtype = torch.float32  # or torch.bfloat16
EPS=1e-5


### WGS

import numpy as np

from lux.config import EnvConfig
from lux.kit import obs_to_game_state
from lux.unit import Unit, move_deltas
import copy
import skimage.segmentation
from scipy.signal import convolve2d, fftconvolve

from lux.utils import direction_to


def to_tuple(pos):
    if isinstance(pos, tuple):
        return pos
    if isinstance(pos, list):
        return tuple(pos)
    return tuple(pos.tolist())


STAGE_BID = 0
STAGE_PLACEMENT = 1
STAGE_ACT = 2


def distance(p1, p2):
    pd = int(np.sum(np.abs(p1 - p2)))
    return pd

def total_cargo(u:Unit):
    return u.cargo.ice + u.cargo.ore + u.cargo.metal + u.cargo.water

def get_nearest(p, p_candidates):
    min_dist = 1000
    min_p = None
    for pn in p_candidates:
        pd = distance(p,pn)
        if pd < min_dist:
            min_dist = pd
            min_p = pn
    return min_p, min_dist

def get_nearest_fromlist(pl, p_candidates):
    min_dist = 1000
    min_p = None
    for p in pl:
        for pn in p_candidates:
            pd = distance(p,pn)
            if pd < min_dist:
                min_dist = pd
                min_p = pn
    return min_p, min_dist

def get_nearest_fromlist_avoid(pl, p_candidates, avoid_list):
    min_dist = 1000
    candidates = []
    for p in pl:
        for pn in p_candidates:
            pd = distance(p,pn)
            if pd < min_dist:
                min_dist = pd
                candidates = [pn]
            elif pd == min_dist:
                candidates += [pn]

    if not candidates:
        return None, min_dist

    candidates.sort(key=lambda p: get_nearest_fromlist(p, avoid_list)[1], reverse=True)
    return candidates[0], min_dist


def flood_fill(bounds, pos_list):
    visited = np.zeros_like(bounds)
    to_visit = copy.copy(pos_list)
    out = np.zeros_like(bounds)
    while to_visit:
        pn = to_visit.pop(0)
        if visited[pn[0], pn[1]]:
            continue

        visited[pn[0], pn[1]] = 1
        if bounds[pn[0], pn[1]]:
            continue

        out[pn[0], pn[1]] = 1
        for d in range(1, 5):
            pv = pn + move_deltas[d]
            if pv[0] < 0 or pv[1] < 0 or pv[0] >= visited.shape[0] or pv[1] >= visited.shape[1]:
                continue
            if visited[pv[0], pv[1]]:
                continue
            to_visit += [pv]
    return out


def fast_flood_fill(bounds, cpos):
    bout = (bounds > 0).copy().astype(int)
    for p in cpos:
        skimage.segmentation.flood_fill(bout, to_tuple(p), 2, connectivity=1, in_place=True)
    return bout == 2

def make_force_map(pos, relax=0.1):
    ax, ay = np.ogrid[0:48, 0:48]
    dst_map = np.abs(ax-pos[0])+np.abs(ay-pos[1])
    return 1.0/(dst_map**2+relax)

def make_dist_map(pos, relax=0.1):
    ax, ay = np.ogrid[0:48, 0:48]
    dst_map = np.abs(ax-pos[0])+np.abs(ay-pos[1])
    return dst_map

class WrappedGameState2:
    def get_adjacent_full(self, pos):
        res = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                pa = to_tuple(pos + np.array([i, j]))
                if self.is_valid_pos(pa):
                    res += [pa]
        return res

    def is_valid_pos(self, target_pos):
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[0] >= self.map_shape[0] or target_pos[1] >= self.map_shape[1]:
            return False
        return True


    def unit_direction_to(self, u, pos2, avoid_pos_set):
        if to_tuple(u.pos) == to_tuple(pos2):
            return 0
        ndir = None
        ndist = (1000,1000)
        for d in range(1,5):
            npos = u.pos + move_deltas[d]
            mc = u.move_cost(self.gs, d)
            if mc is None:
                continue
            if to_tuple(npos) in avoid_pos_set:
                continue
            dist = (distance(npos, pos2), mc)
            if dist < ndist:
                ndir = d
                ndist = dist
        if ndir is not None:
            return ndir

        return direction_to(u.pos, pos2)

    def can_move_direction(self, u:Unit, d):
        mc = u.move_cost(self.gs, d)
        if mc is not None and u.power >= mc + u.unit_cfg.ACTION_QUEUE_POWER_COST:
            return True
        return False

    def make_goodness_mask(self, player_name, min_rubble):
        player_id = int(player_name[-1])
        opponent_id = 1 - player_id
        opponent_name = f"player_{opponent_id}"
        bounds_orig = (self.gs.board.rubble > min_rubble) | (self.ice_without_factories > 0) | (self.ore_without_factories > 0) | (self.factory_tiles[opponent_name])

        goodness = np.zeros(self.gs.board.rubble.shape, dtype=np.float32)
        for pos in np.argwhere(self.gs.board.valid_spawns_mask):
            bo = bounds_orig.copy()
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    bo[pos[0] + i, pos[1] + j] = 0
            ff = fast_flood_fill(bo, [pos])
            goodness[pos[0], pos[1]] = ff.sum()
        return goodness

    def __init__(self, env_cfg: EnvConfig, step: int, obs):
        gs = obs_to_game_state(step, env_cfg, obs)
        self.gs = gs
        self.step = step
        self.obs = obs

        x, y = np.ogrid[0:48*2+1, 0:48*2+1]
        self.kernel = 1/(np.abs(x-48)+np.abs(y-48)+1)
        self.kernel_sq = self.kernel**2
        # self.kernel/=np.sum(self.kernel)

        # noinspection PyTypeChecker
        self.env_cfg: EnvConfig = gs.env_cfg
        self.map_shape = (self.env_cfg.map_size, self.env_cfg.map_size)

        self.pos_to_factory_map = {}
        self.pos_to_unit_map = {}

        self.ice_without_factories = self.gs.board.ice.copy()
        self.ore_without_factories = self.gs.board.ore.copy()

        self.total_lichen = {f"player_{t}": 0 for t in range(2)}
        self.total_water = {f"player_{t}": 0 for t in range(2)}
        self.total_heavy = {f"player_{t}": 0 for t in range(2)}
        self.total_light = {f"player_{t}": 0 for t in range(2)}
        self.factory_tiles = {}
        self.lichen = {}
        self.lichen_boundary = {}

        self.factory_board = {f"player_{t}": np.zeros(self.gs.board.rubble.shape) for t in range(2)}
        self.heavy_board = {f"player_{t}": np.zeros(self.gs.board.rubble.shape) for t in range(2)}
        self.light_board = {f"player_{t}": np.zeros(self.gs.board.rubble.shape) for t in range(2)}

        for t in range(2):
            pn = f'player_{t}'

            ft = np.zeros_like(self.gs.board.ice)
            player_lichen = np.zeros_like(self.gs.board.lichen)

            for f in gs.factories[pn].values():
                self.factory_board[pn][f.pos[0], f.pos[1]] = 1
                lichen_mask = gs.board.lichen_strains == f.strain_id
                player_lichen[lichen_mask] = gs.board.lichen[lichen_mask]
                self.total_lichen[pn] += np.sum(gs.board.lichen[lichen_mask])
                self.total_water[pn] += f.cargo.water
                for pa in self.get_adjacent_full(f.pos):
                    self.pos_to_factory_map[to_tuple(pa)] = f
                    self.ice_without_factories[pa[0], pa[1]] = 0
                    self.ore_without_factories[pa[0], pa[1]] = 0
                    ft[pa[0], pa[1]] = 1

            self.factory_tiles[pn] = ft
            self.lichen[pn] = player_lichen

            for u in gs.units[pn].values():

                if u.unit_type == "HEAVY":
                    self.heavy_board[pn][u.pos[0], u.pos[1]] = 1
                    self.total_heavy[pn] += 1
                else:
                    self.light_board[pn][u.pos[0], u.pos[1]] = 1
                    self.total_light[pn] += 1

                self.pos_to_unit_map[to_tuple(u.pos)] = u


        for t in range(2):
            pn = f'player_{t}'
            po = f'player_{1-t}'
            lb = (self.gs.board.rubble > 0) | (self.ice_without_factories > 0) | (self.ore_without_factories > 0) | (self.factory_tiles[po])
            cp = []
            for f in gs.factories[pn].values():
                for pa in self.get_adjacent_full(f.pos):
                    lb[pa[0], pa[1]] = 0
                cp += [f.pos]
            self.lichen_boundary[pn] = fast_flood_fill(lb, cp)




#######


class TurnState:
    def __init__(self, wgs: WrappedGameState2, player_name):
        self.power_pickup_requests = defaultdict(int)
        self.factory_remaining_power = {fname: f.power for fname, f in wgs.gs.factories[player_name].items()}

######### actions

from typing import Any

import numpy as np

from lux.factory import Factory
from lux.unit import Unit, move_deltas

ACTION_MOVE = 0
ACTION_TRANSFER = 1
ACTION_PICKUP = 2
ACTION_DIG = 3
ACTION_SD = 4
ACTION_RECHARGE = 5

RESOURCE_ICE = 0
RESOURCE_ORE = 1
RESOURCE_WATER = 2
RESOURCE_METAL = 3
RESOURCE_POWER = 4

UNIT_HEAVY = "HEAVY"
UNIT_LIGHT = "LIGHT"

ALL_RESOURCES = list(range(5))
ALL_DIRECTIONS = list(range(5))
ALL_DIRECTIONS_WITHOUT_CENTER = list(range(1,5))

#### bid

class BidActionBase:
    def can_execute(self, wgs:WrappedGameState2):
        return True

    def execute(self, wgs:WrappedGameState2):
        raise NotImplementedError


class BidActionFixedAmount(BidActionBase):
    def __init__(self, amount):
        self._amount = amount

    def __repr__(self):
        return f"bid({self._amount})"

    def execute(self, wgs:WrappedGameState2):
        return dict(faction="AlphaStrike", bid=self._amount)



#### unit

class UnitActionBase:
    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        raise NotImplementedError

    def can_execute(self, u:Unit, wgs:WrappedGameState2):
        raise NotImplementedError

    def pre_execute(self, u:Unit, wgs:WrappedGameState2, turn_state: TurnState):
        pass

    def execute(self, u:Unit, wgs:WrappedGameState2, turn_state: TurnState):
        raise NotImplementedError

    def get_next_pos(self, u: Unit, wgs: WrappedGameState2):
        return to_tuple(u.pos)

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        raise NotImplementedError


class UnitNoopAction(UnitActionBase):
    def __repr__(self):
        return f"unit_noop"

    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        return 0

    def can_execute(self, u:Unit, wgs:WrappedGameState2):
        return True

    def execute(self, u:Unit, wgs:WrappedGameState2, turn_state: TurnState):
        raise NotImplementedError

    def get_next_pos(self, u: Unit, wgs: WrappedGameState2):
        return to_tuple(u.pos)

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        return False


class UnitTypeMoveAction(UnitActionBase):
    def __init__(self, utype, direction):
        self._utype = utype
        self._direction = direction

    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        if self._direction == 0:
            return 0
        return u.move_cost(wgs.gs, self._direction)  # + 1

    def can_execute(self, u: Unit, wgs: WrappedGameState2):
        if u.unit_type != self._utype:
            return False

        # if self._direction == 0 and u.power > u.unit_cfg.BATTERY_CAPACITY * 0.1:
        #     return False

        if self._direction == 0:
            return True

        pnext = to_tuple(u.pos + move_deltas[self._direction])
        if not wgs.is_valid_pos(pnext):
            return False

        fnext = wgs.pos_to_factory_map.get(pnext)
        if fnext and fnext.team_id != u.team_id:
            return False

        return True

    def __repr__(self):
        return f"unittype_move({self._utype}, {self._direction})"

    def execute(self, u: Unit, wgs: WrappedGameState2, turn_state: TurnState):
        return [u.move(self._direction)]

    def get_next_pos(self, u: Unit, wgs:WrappedGameState2):
        if u.power < self.power_cost(u, wgs) + u.unit_cfg.ACTION_QUEUE_POWER_COST:
            return to_tuple(u.pos)
        return to_tuple(u.pos + move_deltas[self._direction])

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        return u.unit_type == self._utype and action[0] == ACTION_MOVE and action[1] == self._direction


class UnitTypeDigAction(UnitActionBase):
    def __init__(self, utype):
        self._utype = utype

    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        return u.dig_cost(wgs.gs)

    def can_execute(self, u: Unit, wgs: WrappedGameState2):
        if u.unit_type != self._utype:
            return False

        fnext = wgs.pos_to_factory_map.get(to_tuple(u.pos))
        if fnext:
            return False

        if wgs.gs.board.lichen_strains[u.pos[0], u.pos[1]] in wgs.gs.teams[u.agent_id].factory_strains:
            return False

        if not wgs.ice_without_factories[u.pos[0], u.pos[1]] and not wgs.ore_without_factories[u.pos[0], u.pos[1]] and not wgs.gs.board.rubble[u.pos[0], u.pos[1]] and not wgs.gs.board.lichen[u.pos[0], u.pos[1]]:
            return False

        return True

    def __repr__(self):
        return f"unittype_dig({self._utype})"

    def execute(self, u: Unit, wgs: WrappedGameState2, turn_state: TurnState):
        return [u.dig()]

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        return u.unit_type == self._utype and action[0] == ACTION_DIG


class UnitTypeRechargeAction(UnitActionBase):
    def __init__(self, utype):
        self._utype = utype

    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        return 0

    def can_execute(self, u: Unit, wgs: WrappedGameState2):
        if u.unit_type != self._utype:
            return False

        fnext = wgs.pos_to_factory_map.get(to_tuple(u.pos))
        if fnext:
            return False

        return True

    def __repr__(self):
        return f"unittype_recharge({self._utype})"

    def execute(self, u: Unit, wgs: WrappedGameState2, turn_state: TurnState):
        return [u.recharge(u.unit_cfg.BATTERY_CAPACITY)]

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        return u.unit_type == self._utype and action[0] == ACTION_RECHARGE


def distance_to_nearest_factory(u: Unit, wgs: WrappedGameState2):
    min_dist = 1000
    for fn, f in wgs.gs.factories[f"player_{u.team_id}"].items():
        dist = np.abs(u.pos - f.pos).sum()
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist

def distance_to_nearest_factory_tile(u: Unit, wgs: WrappedGameState2):
    dist_pair = get_nearest_fromlist(u.pos, np.argwhere(wgs.factory_tiles[u.agent_id]))
    return dist_pair[1]

def distance_to_nearest_heavy(u: Unit, wgs: WrappedGameState2):
    dist_pair = get_nearest_fromlist(u.pos, [x.pos for x in wgs.gs.units[u.agent_id].values() if x.unit_type == UNIT_HEAVY and x.unit_id != u.unit_id])
    return dist_pair[1]

class UnitTypeTransferCargoAction(UnitActionBase):
    def __init__(self, utype):
        self._utype = utype
        self._min_amount = 1

    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        return 0

    def can_execute(self, u: Unit, wgs: WrappedGameState2):
        if u.unit_type != self._utype:
            return False

        if total_cargo(u) < self._min_amount:
            return False

        for d in ALL_DIRECTIONS:
            pnext = to_tuple(u.pos + move_deltas[d])
            fnext = wgs.pos_to_factory_map.get(pnext)
            if fnext and fnext.team_id == u.team_id:
                return True

        for d in ALL_DIRECTIONS_WITHOUT_CENTER:
            pnext = to_tuple(u.pos + move_deltas[d])
            unext = wgs.pos_to_unit_map.get(pnext)
            if not unext or unext.team_id != u.team_id:
                continue
            if total_cargo(unext) + self._min_amount >= unext.unit_cfg.CARGO_SPACE:
                continue
            return True

        return True

    def __repr__(self):
        return f"unittype_transfer_cargo({self._utype})"

    def execute(self, u: Unit, wgs: WrappedGameState2, turn_state: TurnState):
        for d in ALL_DIRECTIONS:
            pnext = to_tuple(u.pos + move_deltas[d])
            fnext = wgs.pos_to_factory_map.get(pnext)
            if fnext and fnext.team_id == u.team_id:
                if u.cargo.ice:
                    return [u.transfer(d, RESOURCE_ICE, u.cargo.ice)]
                elif u.cargo.ore:
                    return [u.transfer(d, RESOURCE_ORE, u.cargo.ore)]

        candidates = []
        for d in ALL_DIRECTIONS_WITHOUT_CENTER:
            pnext = to_tuple(u.pos + move_deltas[d])
            unext = wgs.pos_to_unit_map.get(pnext)
            if not unext or unext.team_id != u.team_id:
                continue

            space_left = unext.unit_cfg.CARGO_SPACE - total_cargo(unext)
            if space_left < self._min_amount:
                continue

            dist_pair = get_nearest_fromlist(unext.pos, np.argwhere(wgs.factory_tiles[u.agent_id]))
            if dist_pair[0] is None:
                cand_score = (1000,0)
                candidates += [(cand_score, d, space_left)]
                continue

            cand_score = (dist_pair[1], -space_left)
            candidates += [(cand_score, d, space_left)]
            continue

        candidates.sort(key=lambda x:x[0])

        if candidates:
            d = candidates[0][1]
            amount = candidates[0][2]
            if u.cargo.ice:
                return [u.transfer(d, RESOURCE_ICE, amount)]
            elif u.cargo.ore:
                return [u.transfer(d, RESOURCE_ORE, amount)]

        return [u.move(0)]

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        return u.unit_type == self._utype and action[0] == ACTION_TRANSFER and action[2] != RESOURCE_POWER


class UnitTypeTransferPowerAction(UnitActionBase):
    def __init__(self, utype):
        self._utype = utype
        self._min_transfer_amount = 10
        self._min_keep = 10

    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        return 0

    def can_execute(self, u: Unit, wgs: WrappedGameState2):
        if u.unit_type != self._utype:
            return False

        if u.power < u.unit_cfg.ACTION_QUEUE_POWER_COST + self._min_keep + self._min_transfer_amount:
            return False

        for d in ALL_DIRECTIONS_WITHOUT_CENTER:
            pnext = to_tuple(u.pos + move_deltas[d])
            fnext = wgs.pos_to_factory_map.get(pnext)
            if fnext:
                # do not transfer power to factory
                continue

            unext = wgs.pos_to_unit_map.get(pnext)
            if not unext or unext.team_id != u.team_id:
                continue
            max_space = unext.unit_cfg.BATTERY_CAPACITY - unext.power
            max_amount = min(max_space, u.power - u.unit_cfg.ACTION_QUEUE_POWER_COST - self._min_keep)
            if max_amount < self._min_transfer_amount:
                continue
            return True

        return False

    def __repr__(self):
        return f"unittype_transfer_power({self._utype})"

    def execute(self, u: Unit, wgs: WrappedGameState2, turn_state: TurnState):
        candidates = []
        for d in ALL_DIRECTIONS_WITHOUT_CENTER:
            pnext = to_tuple(u.pos + move_deltas[d])
            fnext = wgs.pos_to_factory_map.get(pnext)
            if fnext:
                continue

            unext = wgs.pos_to_unit_map.get(pnext)
            if not unext or unext.team_id != u.team_id:
                continue

            max_space = unext.unit_cfg.BATTERY_CAPACITY - unext.power
            max_amount = min(max_space, u.power - u.unit_cfg.ACTION_QUEUE_POWER_COST - self._min_keep)
            if max_amount < self._min_transfer_amount:
                continue

            nearest_resource_to_target_pair = get_nearest_fromlist(unext.pos, np.argwhere(wgs.ore_without_factories | wgs.ice_without_factories))
            cand_score = (-1 if unext.unit_type == UNIT_HEAVY else 0, nearest_resource_to_target_pair[1])
            candidates += [(cand_score, d, max_amount)]

        candidates.sort(key=lambda x:x[0])
        if candidates:
            d = candidates[0][1]
            max_amount = candidates[0][2]
            return [u.transfer(d, RESOURCE_POWER, max_amount )]

        return [u.move(0)]

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        return u.unit_type == self._utype and action[0] == ACTION_TRANSFER and action[2] == RESOURCE_POWER


class UnitTypePickupCargoAction(UnitActionBase):
    def __init__(self, utype):
        self._utype = utype

    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        return 0

    def can_execute(self, u: Unit, wgs: WrappedGameState2):
        if u.unit_type != self._utype:
            return False
        return False

    def __repr__(self):
        return f"unittype_pickup_cargo({self._utype})"

    def execute(self, u: Unit, wgs: WrappedGameState2, turn_state: TurnState):
        raise NotImplementedError()

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        return u.unit_type == self._utype and action[0] == ACTION_PICKUP and action[2] != RESOURCE_POWER


class UnitTypePickupPowerAction(UnitActionBase):
    def __init__(self, utype):
        self._utype = utype
        self._min_amount = 10

    def power_cost(self, u:Unit, wgs:WrappedGameState2):
        return 0

    def pre_execute(self, u:Unit, wgs:WrappedGameState2, turn_state: TurnState):

        fnext = wgs.pos_to_factory_map.get(to_tuple(u.pos))
        if not fnext or fnext.team_id != u.team_id:
            return

        turn_state.power_pickup_requests[fnext.unit_id] += u.unit_cfg.BATTERY_CAPACITY - u.power

    def can_execute(self, u: Unit, wgs: WrappedGameState2):
        if u.unit_type != self._utype:
            return False

        # if u.power > 0.8 * u.unit_cfg.BATTERY_CAPACITY:
        #     return False

        fnext = wgs.pos_to_factory_map.get(to_tuple(u.pos))
        if not fnext or fnext.team_id != u.team_id:
            return False

        if u.unit_cfg.BATTERY_CAPACITY - u.power < self._min_amount:
            return False

        if fnext.power < self._min_amount:
            return False
        #
        # if u.unit_type == UNIT_LIGHT and fnext.power < 500:
        #     return False

        return True

    def __repr__(self):
        return f"unittype_pickup_power({self._utype})"

    def execute(self, u: Unit, wgs: WrappedGameState2, turn_state: TurnState):
        fnext = wgs.pos_to_factory_map.get(to_tuple(u.pos))
        if not fnext or fnext.team_id != u.team_id:
            return

        total_factory_power_requests = turn_state.power_pickup_requests.get(fnext.unit_id)
        if total_factory_power_requests is None:
            return

        scale = min(1, fnext.power / (total_factory_power_requests+1) )
        scaled_amount = int( (u.unit_cfg.BATTERY_CAPACITY - u.power) * scale )
        if scaled_amount < self._min_amount:
            return

        # assert turn_state.factory_remaining_power[fnext.unit_id] >= scaled_amount
        turn_state.factory_remaining_power[fnext.unit_id] -= scaled_amount
        return [u.pickup(RESOURCE_POWER, scaled_amount)]

    def is_me(self, u: Unit, wgs: WrappedGameState2, action):
        return u.unit_type == self._utype and action[0] == ACTION_PICKUP and action[2] == RESOURCE_POWER




# factory

class FactoryActionBase:
    def can_execute(self, f:Factory, wgs:WrappedGameState2):
        raise NotImplementedError

    def execute(self, f:Factory, wgs:WrappedGameState2):
        raise NotImplementedError

    def get_next_pos(self, f: Factory, wgs:WrappedGameState2):
        return None


    def is_me(self, action):
        raise NotImplementedError

class FactoryNoopAction(FactoryActionBase):
    def can_execute(self, f:Factory, wgs:WrappedGameState2):
        return True

    def execute(self, f:Factory, wgs:WrappedGameState2):
        return None

    def get_next_pos(self, f: Factory, wgs:WrappedGameState2):
        return None

    def __repr__(self):
        return f"factory_noop"

    def is_me(self, action):
        return False

class FactoryBuildHeavyAction(FactoryActionBase):
    def can_execute(self, f: Factory, wgs: WrappedGameState2):
        return f.can_build_heavy(wgs.gs)

    def execute(self, f, wgs):
        return f.build_heavy()

    def get_next_pos(self, f: Factory, wgs:WrappedGameState2):
        return to_tuple(f.pos)

    def __repr__(self):
        return f"factory_build_heavy"

    def is_me(self, action):
        return action == 1

class FactoryBuildLightAction(FactoryActionBase):
    def can_execute(self, f: Factory, wgs: WrappedGameState2):
        return f.can_build_light(wgs.gs)

    def execute(self, f, wgs):
        return f.build_light()

    def __repr__(self):
        return f"factory_build_light"

    def get_next_pos(self, f: Factory, wgs:WrappedGameState2):
        return to_tuple(f.pos)

    def is_me(self, action):
        return action == 0

class FactoryWaterLichenAction(FactoryActionBase):
    def can_execute(self, f: Factory, wgs: WrappedGameState2):
        return f.can_water(wgs.gs)

    def execute(self, f, wgs):
        return f.water()

    def __repr__(self):
        return f"factory_water"

    def is_me(self, action):
        return action == 2




#### action maker

import torch

from common.buffers import BufferSpec
from lux.utils import my_turn_to_place_factory

class ActionsMaker:
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.map_size = env_cfg.map_size
        self.bid_actions = []
        self.factory_actions = []
        self.unit_actions = []

    def inverse_actions(self, wgs: WrappedGameState2, player_name, player_actions):
        act_bid = np.zeros((len(self.bid_actions),), dtype=np_dtype)
        act_placement = np.zeros(shape=(self.map_size, self.map_size), dtype=np_dtype)
        act_unit = np.zeros(shape=(self.map_size, self.map_size, len(self.unit_actions)), dtype=np_dtype)
        act_factory = np.zeros(shape=(self.map_size, self.map_size, len(self.factory_actions)), dtype=np_dtype)

        if 'bid' in player_actions:
            amnt = player_actions['bid']
            scores = [ (np.abs(x._amount - amnt), ei) for ei, x in enumerate(self.bid_actions) ]
            scores.sort(key=lambda x: x[0])
            act_bid[scores[0][1]] = 1

        if 'spawn' in player_actions:
            pos = player_actions['spawn']
            act_placement[pos[0], pos[1]] = 1

        for uname, u in wgs.gs.units[player_name].items():
            if uname in player_actions:
                next_action = player_actions[uname][0]
                # print("inverse", uname, u.pos, 'raw action', next_action)

            elif u.action_queue:
                next_action = u.action_queue[0]
                # print("inverse", uname, u.pos, 'from queue action', next_action)

            else:
                # print("inverse", uname, u.pos, 'noop action')
                # act_unit[u.pos[0], u.pos[1], 0] = 1
                continue


            for ei, x in enumerate(self.unit_actions):
                if x.can_execute(u, wgs) and x.is_me(u, wgs, next_action) and u.power >= x.power_cost(u, wgs):
                    act_unit[u.pos[0], u.pos[1], ei] = 1
                    break

            # else:
                # print("U not found", uname, u.pos, u.cargo, next_action)
                # act_unit[u.pos[0], u.pos[1], 0] = 1


        for fname, f in wgs.gs.factories[player_name].items():
            if fname in player_actions:
                for ei, x in enumerate(self.factory_actions):
                    if x.is_me(player_actions[fname]):
                        if not x.can_execute(f, wgs):
                            # print("F found but cannot execute", fname, f.pos, f.cargo, ei, repr(x), player_actions[fname])
                            act_factory[f.pos[0], f.pos[1], 0] = 1
                            break

                        act_factory[f.pos[0], f.pos[1], ei] = 1
                        break
                else:
                    act_factory[f.pos[0], f.pos[1], 0] = 1
            else:
                act_factory[f.pos[0], f.pos[1], 0] = 1

        return dict(
            bid=torch.as_tensor(act_bid),
            unit=torch.as_tensor(act_unit),
            factory=torch.as_tensor(act_factory),
            placement=torch.as_tensor(act_placement).flatten()
        )

    def get_actions_spec(self):
        return dict(
            bid=(len(self.bid_actions, )),
            unit=(self.map_size, self.map_size, len(self.unit_actions)),
            factory=(self.map_size, self.map_size, len(self.factory_actions)),
            placement=(self.map_size * self.map_size, ),
        )

    def get_actions_mask_spec(self):
        return dict(
            bid=BufferSpec((len(self.bid_actions),), torch.bool),
            unit=BufferSpec((self.map_size, self.map_size, len(self.unit_actions)), torch.bool),
            factory=BufferSpec((self.map_size, self.map_size, len(self.factory_actions)), torch.bool),
            placement=BufferSpec((self.map_size * self.map_size, ), torch.bool)
        )

    def make_actions_mask(self, wgs, player_name):
        # action masks
        gs = wgs.gs

        act_bid = np.zeros((len(self.bid_actions),), dtype=np_dtype)
        if wgs.step == 0:
            act_bid[:] = 1

        act_placement = np.zeros(shape=(self.map_size, self.map_size, 1), dtype=np_dtype)

        if wgs.step > 0:
            factories_to_place = gs.teams[player_name].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(gs.teams[player_name].place_first, wgs.step)
            if factories_to_place > 0 and my_turn_to_place:
                act_placement[:,:,0] = wgs.gs.board.valid_spawns_mask == 1

        act_unit = np.zeros(shape=(self.map_size, self.map_size, len(self.unit_actions)), dtype=np_dtype)
        for u in gs.units[player_name].values():
            for ei, ua in enumerate(self.unit_actions):
                if ua.can_execute(u, wgs): # and u.power >= ua.power_cost(u, wgs) + u.unit_cfg.ACTION_QUEUE_POWER_COST:
                    act_unit[u.pos[0], u.pos[1], ei] = 1

        act_factory = np.zeros(shape=(self.map_size, self.map_size, len(self.factory_actions)), dtype=np_dtype)
        for f in gs.factories[player_name].values():
            for ei, fa in enumerate(self.factory_actions):
                if fa.can_execute(f, wgs):
                    act_factory[f.pos[0], f.pos[1], ei] = 1

        return dict(
            bid=torch.as_tensor(act_bid, dtype=torch_dtype),
            unit=torch.as_tensor(act_unit, dtype=torch_dtype),
            factory=torch.as_tensor(act_factory, dtype=torch_dtype),
            placement=torch.as_tensor(act_placement, dtype=torch_dtype).flatten()
        )

#####



def make_action_maker(env_cfg):
    amaker = ActionsMaker(env_cfg)
    amaker.bid_actions = [
        BidActionFixedAmount(0),
        BidActionFixedAmount(10),
        BidActionFixedAmount(20),
    ]

    amaker.factory_actions = [
        FactoryNoopAction(),
        FactoryBuildHeavyAction(),
        FactoryBuildLightAction(),
        FactoryWaterLichenAction()
    ]

    amaker.unit_actions = [
#         UnitNoopAction()
    ]

    amaker.unit_actions += [
        *[UnitTypeMoveAction(UNIT_HEAVY, d) for d in ALL_DIRECTIONS],
        UnitTypeDigAction(UNIT_HEAVY),
        UnitTypeTransferCargoAction(UNIT_HEAVY),
        UnitTypeTransferPowerAction(UNIT_HEAVY),
        UnitTypePickupPowerAction(UNIT_HEAVY),
        UnitTypeRechargeAction(UNIT_HEAVY),
    ]

    amaker.unit_actions += [
        *[UnitTypeMoveAction(UNIT_LIGHT, d) for d in ALL_DIRECTIONS],
        UnitTypeDigAction(UNIT_LIGHT),
        UnitTypeTransferCargoAction(UNIT_LIGHT),
        UnitTypeTransferPowerAction(UNIT_LIGHT),
        UnitTypePickupPowerAction(UNIT_LIGHT),
        UnitTypeRechargeAction(UNIT_LIGHT),
    ]

    return amaker


############# features
from dataclasses import dataclass
from typing import Iterable, List, Dict

import torch

from common.buffers import BufferSpec
from lux.config import EnvConfig

@dataclass
class FeaturesShape:
    game: int
    board: int

@dataclass
class FeaturesIndices:
    game: List[int]
    board: List[int]


class FeatureBase:
    def __init__(self, name):
        self._name = name
        self._idx_offset = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def offset_begin(self):
        return self._idx_offset

    @property
    def offset_end(self):
        return self._idx_offset + self.shape[-1]

    def enable(self, idx):
        self._idx_offset = idx

    def disable(self):
        self._idx_offset = None

    def call(self, wgs: WrappedGameState2, *args):
        raise NotImplementedError

    def get_type(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def enabled(self):
        return self._idx_offset is not None


class GameLambdaFeature(FeatureBase):
    def __init__(self, name, lmbd):
        super().__init__(name)
        self._lmbd = lmbd

    @property
    def shape(self):
        return (1,)

    def get_type(self):
        return 0

    def call(self, wgs: WrappedGameState2,  *args):
        return torch.as_tensor(self._lmbd(wgs, *args), dtype=torch_dtype)


class BoardLambdaFeature(FeatureBase):
    def __init__(self, name, lmbd):
        super().__init__(name)
        self._lmbd = lmbd

    @property
    def shape(self):
        return (None, None, 1)

    def get_type(self):
        return 1

    def call(self, wgs: WrappedGameState2, *args):
        res = torch.as_tensor(self._lmbd(wgs,  *args), dtype=torch_dtype)
        while len(res.shape) < len(self.shape):
            res = res.unsqueeze(-1)
        return res

class CellLambdaFeature(FeatureBase):
    def __init__(self, name, lmbd):
        super().__init__(name)
        self._lmbd = lmbd

    @property
    def shape(self):
        return (None, None, 1)

    def get_type(self):
        return 1

    def call(self, wgs: WrappedGameState2, *args):
        return float(self._lmbd(wgs,  *args))

    # def call(self, wgs: WrappedGameState2, *args):
    #     return torch.as_tensor(self._lmbd(wgs,  *args), dtype=torch.float)


class NonReplacingDict(dict):
    def __setitem__(self, key, value):
        assert key not in self
        return super().__setitem__(key, value)



class FeaturesMaker:
    def __init__(self, env_cfg: EnvConfig):
        self._map_size = env_cfg.map_size

        self.game_features = []
        self.board_features = []
        self.player_unit_features = []
        self.player_factory_features = []
        self.player_factory_tile_features = []
        self.opponent_unit_features = []
        self.opponent_factory_features = []
        self.opponent_factory_tile_features = []

        self._all_features:Dict[str, FeatureBase] = NonReplacingDict()
        self._features_shape = [0, 0]

    def list_feature_names(self):
        return list(self._all_features.keys())

    def finalize_features(self):
        for f in (
                  self.game_features
                + self.board_features
                + self.player_unit_features
                + self.player_factory_features
                + self.player_factory_tile_features
                + self.opponent_unit_features
                + self.opponent_factory_features
                + self.opponent_factory_tile_features):
            self._all_features[f.name] = f

    def reset_features(self):
        for fn, f in self._all_features.items():
            f.disable()
        self._features_shape = [0, 0]

    def enable_features(self, names: Iterable[str]):
        for name in names:
            feat = self._all_features[name]
            if feat.enabled:
                continue
            feat.enable(self._features_shape[feat.get_type()])
            self._features_shape[feat.get_type()] += feat.shape[-1]

    def shape(self, names: Iterable[str]) -> FeaturesShape:
        s = [0,0]
        for name in names:
            feat = self._all_features[name]
            assert feat.enabled
            s[feat.get_type()] += feat.shape[-1]
        return FeaturesShape(*s)

    def indices(self, names: Iterable[str]) -> FeaturesIndices:
        idx = [[],[]]
        for name in names:
            feat = self._all_features[name]
            assert feat.enabled
            idx[feat.get_type()] += list(range(feat.offset_begin, feat.offset_end))
        return FeaturesIndices(*idx)

    def get_obs_spec(self):
        return dict(
            game=BufferSpec((self._features_shape[0],), dtype=torch_dtype),
            board=BufferSpec((self._map_size, self._map_size, self._features_shape[1]), dtype=torch_dtype)
        )

    def collect_features(self, wgs: WrappedGameState2, player_name):
        player_id = int(player_name[-1])
        opponent_id = 1-player_id
        opponent_name = f"player_{opponent_id}"

        game_features_vec = torch.zeros((self._features_shape[0],), dtype=torch_dtype)
        board_features_vec = torch.zeros((self._map_size, self._map_size, self._features_shape[1]), dtype=torch_dtype)

        for feat in self.game_features:
            if not feat.enabled:
                continue
            game_features_vec[feat.offset_begin:feat.offset_end] = (feat.call(wgs, player_name, opponent_name))

        for feat in self.board_features:
            if not feat.enabled:
                continue
            board_features_vec[:,:,feat.offset_begin:feat.offset_end] = (feat.call(wgs, player_name, opponent_name))

        for uname, u in wgs.gs.units[player_name].items():
            for feat in self.player_unit_features:
                if not feat.enabled:
                    continue
                board_features_vec[u.pos[0], u.pos[1], feat.offset_begin:feat.offset_end] = (feat.call(wgs, uname, u))

        for fname, f in wgs.gs.factories[player_name].items():
            for feat in self.player_factory_features:
                if not feat.enabled:
                    continue
                board_features_vec[f.pos[0], f.pos[1], feat.offset_begin:feat.offset_end] = (feat.call(wgs, fname, f))

            for feat in self.player_factory_tile_features:
                if not feat.enabled:
                    continue
                for pos in wgs.get_adjacent_full(f.pos):
                    board_features_vec[pos[0], pos[1], feat.offset_begin:feat.offset_end] = (feat.call(wgs, fname, f, pos))

        for uname, u in wgs.gs.units[opponent_name].items():
            for feat in self.opponent_unit_features:
                if not feat.enabled:
                    continue
                board_features_vec[u.pos[0], u.pos[1], feat.offset_begin:feat.offset_end] = (feat.call(wgs, uname, u))

        for fname, f in wgs.gs.factories[opponent_name].items():
            for feat in self.opponent_factory_features:
                if not feat.enabled:
                    continue
                board_features_vec[f.pos[0], f.pos[1], feat.offset_begin:feat.offset_end] = (feat.call(wgs, fname, f))

            for feat in self.opponent_factory_tile_features:
                if not feat.enabled:
                    continue
                for pos in wgs.get_adjacent_full(f.pos):
                    board_features_vec[pos[0], pos[1], feat.offset_begin:feat.offset_end] = (feat.call(wgs, fname, f, pos))

        return dict(game=game_features_vec, board=board_features_vec)

####### fmaker

def if_unit_can_move(wgs, u, d):
    mc = u.move_cost(wgs.gs, d)
    if mc is None:
        return False
    if len(u.action_queue) and u.action_queue[0][0] == ACTION_MOVE and u.action_queue[0][1] == d:
        return u.power >= mc
    return u.power >= mc + u.unit_cfg.ACTION_QUEUE_POWER_COST


def make_features_maker(env_cfg):
    fm = FeaturesMaker(env_cfg)

    def null_to_num(x, v):
        if x is None:
            return v
        return x

    fm.game_features += [GameLambdaFeature("real_env_steps", lambda wgs, p, o: max(0, wgs.gs.real_env_steps / env_cfg.max_episode_length))]
    fm.board_features += [BoardLambdaFeature("is_player0", lambda wgs, p, o: p == "player_0")]

    fm.board_features += [BoardLambdaFeature("board_real_env_steps", lambda wgs, p, o: max(0, wgs.gs.real_env_steps / env_cfg.max_episode_length))]
    fm.board_features += [BoardLambdaFeature("board_factories_per_team", lambda wgs, p, o: wgs.gs.board.factories_per_team)]

    fm.board_features += [BoardLambdaFeature("board_is_day", lambda wgs, p, o: (max(0, wgs.gs.real_env_steps) % env_cfg.CYCLE_LENGTH) < env_cfg.DAY_LENGTH)]
    fm.board_features += [BoardLambdaFeature("board_day_part", lambda wgs, p, o: (max(0, wgs.gs.real_env_steps) % env_cfg.CYCLE_LENGTH) / env_cfg.DAY_LENGTH)]

    fm.board_features += [BoardLambdaFeature("ice", lambda wgs, p, o: wgs.ice_without_factories)]
    fm.board_features += [BoardLambdaFeature("ice_force", lambda wgs, p, o: fftconvolve(wgs.ice_without_factories,wgs.kernel_sq, mode='same'))]

    fm.board_features += [BoardLambdaFeature("ore", lambda wgs, p, o: wgs.ore_without_factories)]
    fm.board_features += [BoardLambdaFeature("ore_force", lambda wgs, p, o: fftconvolve(wgs.ore_without_factories,wgs.kernel_sq, mode='same'))]

    fm.board_features += [BoardLambdaFeature("valid_spawns_mask", lambda wgs, p, o: wgs.gs.board.valid_spawns_mask if wgs.gs.real_env_steps < 0 else 0)]

    fm.board_features += [BoardLambdaFeature("player_fact_force", lambda wgs, p, o: fftconvolve(wgs.factory_board[p], wgs.kernel_sq, mode='same') )]
    fm.board_features += [BoardLambdaFeature("player_heavy_force", lambda wgs, p, o: fftconvolve(wgs.heavy_board[p], wgs.kernel_sq, mode='same') )]
    fm.board_features += [BoardLambdaFeature("player_light_force", lambda wgs, p, o: fftconvolve(wgs.light_board[p], wgs.kernel_sq, mode='same') )]

    fm.board_features += [BoardLambdaFeature("opponent_fact_force", lambda wgs, p, o: fftconvolve(wgs.factory_board[o], wgs.kernel_sq, mode='same') )]
    fm.board_features += [BoardLambdaFeature("opponent_heavy_force", lambda wgs, p, o: fftconvolve(wgs.heavy_board[o], wgs.kernel_sq, mode='same') )]
    fm.board_features += [BoardLambdaFeature("opponent_light_force", lambda wgs, p, o: fftconvolve(wgs.light_board[o], wgs.kernel_sq, mode='same') )]

    fm.board_features += [BoardLambdaFeature("player_lichen", lambda wgs, p, o: wgs.lichen[p] > 0)]
    fm.board_features += [BoardLambdaFeature("player_lichen_val", lambda wgs, p, o:wgs.lichen[p] / env_cfg.MAX_LICHEN_PER_TILE)]

    fm.board_features += [BoardLambdaFeature("opponent_lichen", lambda wgs, p, o: wgs.lichen[o] > 0)]
    fm.board_features += [BoardLambdaFeature("opponent_lichen_val", lambda wgs, p, o:wgs.lichen[o] / env_cfg.MAX_LICHEN_PER_TILE)]

    fm.board_features += [BoardLambdaFeature("player_lichen_boundary", lambda wgs, p, o: wgs.lichen_boundary[p])]
    fm.board_features += [BoardLambdaFeature("opponent_lichen_boundary", lambda wgs, p, o: wgs.lichen_boundary[o])]

    fm.board_features += [BoardLambdaFeature("rubble_val", lambda wgs, p, o: wgs.gs.board.rubble / env_cfg.MAX_RUBBLE)]
    fm.board_features += [BoardLambdaFeature("has_rubble", lambda wgs, p, o: wgs.gs.board.rubble > 0)]
    fm.board_features += [BoardLambdaFeature("playable_area", lambda wgs, p, o: 1)]

    fm.board_features += [BoardLambdaFeature("player_total_factories", lambda wgs, p, o: len(wgs.gs.factories[p]) / 5)]
    fm.board_features += [BoardLambdaFeature("opponent_total_factories", lambda wgs, p, o: len(wgs.gs.factories[o]) / 5)]

    fm.board_features += [BoardLambdaFeature("player_total_heavy", lambda wgs, p, o: wgs.total_heavy[p] / 100)]
    fm.board_features += [BoardLambdaFeature("player_total_light", lambda wgs, p, o: wgs.total_light[p] / 100)]

    fm.board_features += [BoardLambdaFeature("opponent_total_heavy", lambda wgs, p, o: wgs.total_heavy[o] / 100)]
    fm.board_features += [BoardLambdaFeature("opponent_total_light", lambda wgs, p, o: wgs.total_light[o] / 100)]

    fm.board_features += [BoardLambdaFeature("player_total_water", lambda wgs, p, o: wgs.total_water[p] / 10000)]
    fm.board_features += [BoardLambdaFeature("opponent_total_water", lambda wgs, p, o: wgs.total_water[o] / 10000)]

    fm.board_features += [BoardLambdaFeature("player_total_lichen", lambda wgs, p, o: wgs.total_lichen[p] / 10000)]
    fm.board_features += [BoardLambdaFeature("opponent_total_lichen", lambda wgs, p, o: wgs.total_lichen[o] / 10000)]

    fm.player_unit_features += [CellLambdaFeature("player_units", lambda wgs, uname, u: 1)]
    fm.player_unit_features += [CellLambdaFeature("player_units_is_heavy", lambda wgs, uname, u: u.unit_type == UNIT_HEAVY)]
    fm.player_unit_features += [CellLambdaFeature("player_units_is_light", lambda wgs, uname, u: u.unit_type == UNIT_LIGHT)]
    fm.player_unit_features += [CellLambdaFeature("player_units_cargo_ice_rel", lambda wgs, uname, u: u.cargo.ice / wgs.env_cfg.ROBOTS[u.unit_type].CARGO_SPACE)]
    fm.player_unit_features += [CellLambdaFeature("player_units_cargo_ice_abs", lambda wgs, uname, u: u.cargo.ice / 1000)]
    fm.player_unit_features += [CellLambdaFeature("player_units_cargo_ore_rel", lambda wgs, uname, u: u.cargo.ore / wgs.env_cfg.ROBOTS[u.unit_type].CARGO_SPACE)]
    fm.player_unit_features += [CellLambdaFeature("player_units_cargo_ore_abs", lambda wgs, uname, u: u.cargo.ore / 1000)]
    fm.player_unit_features += [CellLambdaFeature("player_units_has_ice", lambda wgs, uname, u: u.cargo.ice > 0)]
    fm.player_unit_features += [CellLambdaFeature("player_units_has_ore", lambda wgs, uname, u: u.cargo.ore > 0)]
    fm.player_unit_features += [CellLambdaFeature("player_units_cargo_rel", lambda wgs, uname, u: total_cargo(u) / wgs.env_cfg.ROBOTS[u.unit_type].CARGO_SPACE)]
    fm.player_unit_features += [CellLambdaFeature("player_units_cargo_abs", lambda wgs, uname, u: total_cargo(u) / 1000)]
    fm.player_unit_features += [CellLambdaFeature("player_units_power_rel", lambda wgs, uname, u: u.power / wgs.env_cfg.ROBOTS[u.unit_type].BATTERY_CAPACITY)]
    fm.player_unit_features += [CellLambdaFeature("player_units_power_abs", lambda wgs, uname, u: u.power / 1000)]

    fm.player_factory_features += [CellLambdaFeature("player_factories", lambda wgs, fname, f: 1)]
    fm.player_factory_features += [CellLambdaFeature("player_factories_metal", lambda wgs, fname, f: f.cargo.metal / 1000)]
    fm.player_factory_features += [CellLambdaFeature("player_factories_water", lambda wgs, fname, f: f.cargo.water / 1000)]
    fm.player_factory_features += [CellLambdaFeature("player_factories_ice", lambda wgs, fname, f: f.cargo.ice / 1000)]
    fm.player_factory_features += [CellLambdaFeature("player_factories_ore", lambda wgs, fname, f: f.cargo.ore / 1000)]
    fm.player_factory_features += [CellLambdaFeature("player_factories_power", lambda wgs, fname, f: f.power / 10000)]
    fm.player_factory_features += [CellLambdaFeature("player_factories_water_cost", lambda wgs, fname, f: f.water_cost(wgs.gs) / 10)]

    fm.player_factory_tile_features += [CellLambdaFeature("player_factory_tiles", lambda wgs, fname, f, ft: 1)]

    fm.opponent_unit_features += [CellLambdaFeature("opponent_units", lambda wgs, uname, u: 1)]
    fm.opponent_unit_features += [CellLambdaFeature("opponent_units_is_heavy", lambda wgs, uname, u: u.unit_type == UNIT_HEAVY)]
    fm.opponent_unit_features += [CellLambdaFeature("opponent_units_is_light", lambda wgs, uname, u: u.unit_type == UNIT_LIGHT)]
    fm.opponent_unit_features += [CellLambdaFeature("opponent_units_power_rel", lambda wgs, uname, u: u.power / wgs.env_cfg.ROBOTS[u.unit_type].BATTERY_CAPACITY)]
    fm.opponent_unit_features += [CellLambdaFeature("opponent_units_power_abs", lambda wgs, uname, u: u.power / 1000)]

    fm.opponent_unit_features += [CellLambdaFeature("opponent_units_can_move_1", lambda wgs, uname, u: if_unit_can_move(wgs, u, 1) )  ]
    fm.opponent_unit_features += [CellLambdaFeature("opponent_units_can_move_2", lambda wgs, uname, u: if_unit_can_move(wgs, u, 2) )  ]
    fm.opponent_unit_features += [CellLambdaFeature("opponent_units_can_move_3", lambda wgs, uname, u: if_unit_can_move(wgs, u, 3) )  ]
    fm.opponent_unit_features += [CellLambdaFeature("opponent_units_can_move_4", lambda wgs, uname, u: if_unit_can_move(wgs, u, 4) )  ]

    fm.opponent_factory_features += [CellLambdaFeature("opponent_factories", lambda wgs, fname, f: 1)]

    fm.opponent_factory_tile_features += [CellLambdaFeature("opponent_factory_tiles", lambda wgs, fname, f, ft: 1)]

    fm.finalize_features()

    fm.enable_features(fm.list_feature_names())
    return fm


###### model

from common.symmetric_layers_torch import SymmetricConv2d
from torch.distributions import Categorical, Multinomial
import torch.nn.functional as F
import torch
from torch import nn


def to_device(x, device, non_blocking=False):
    if isinstance(x, dict):
        return {k: to_device(v, device, non_blocking) for k, v in x.items()}
    return x.to(device, non_blocking=non_blocking)

def to_cpu(x):
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    return x.cpu()


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


class Scaling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Conv2d(in_features, in_features, 1, groups=in_features, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


class Concat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class ConvRelu(nn.Module):
    def __init__(self, in_chan, out_chan, size, batch_norm):
        super().__init__()
        self.residual_conv = nn.Conv2d(in_chan, out_chan, (size, size), padding="same")
        self.bn = nn.BatchNorm2d(out_chan) if batch_norm else nn.Identity(out_chan)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.bn(res)
        res = F.relu(res)
        return res


class ResidualDualConvRelu(nn.Module):
    def __init__(self, in_chan, size, batch_norm):
        super().__init__()
        self.residual_conv = nn.Conv2d(in_chan, in_chan, (size, size), padding="same")
        self.bn1 = nn.BatchNorm2d(in_chan) if batch_norm else nn.Identity(in_chan)
        self.residual_conv2 = nn.Conv2d(in_chan, in_chan, (size, size), padding="same")
        self.bn2 = nn.BatchNorm2d(in_chan) if batch_norm else nn.Identity(in_chan)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.bn1(res)
        res = F.relu(res)
        res = self.residual_conv2(res)
        res = self.bn2(res)
        res = res + x
        res = F.relu(res)
        return res


class FeatureExtractor(nn.Module):
    def __init__(self, in_chan, features, layers, batch_norm):
        super().__init__()
        self.blks = nn.Sequential(
            ConvRelu(in_chan, features, 1, batch_norm=batch_norm),
            *(ResidualDualConvRelu(features, 3, batch_norm=batch_norm) for _ in range(layers)),
        )

    def forward(self, state):
        return self.blks(state)


def estimate_loss_prob(pred, target, amask, cw):
    res = {}
    for k in pred:
        loss_grid = -pred[k].log() * target[k] * amask[k] * cw[k]
        # print(loss_grid.shape)
        loss = loss_grid.sum() / target[k].shape[0]
        res[k] = loss
    return res


def estimate_loss_logprob(logpred, target, amask, cw):
    res = {}
    for k in logpred:
        loss_grid = -logpred[k] * target[k] * amask[k] * cw[k]
        # print(loss_grid.shape)
        loss = loss_grid.sum() / target[k].shape[0]
        res[k] = loss
    return res


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("placeholder", torch.tensor(1e-5, requires_grad=False))
        self.register_buffer("neginf", torch.tensor(-1e20, dtype=torch.float, requires_grad=False))
        self.register_buffer("one", torch.tensor(1, dtype=torch.float, requires_grad=False))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.float, requires_grad=False))
        self.register_buffer("smallfloat", torch.tensor(1e-45, dtype=torch.float, requires_grad=False))

    @torch.jit.export
    def forward(self, x_game, x_board):
        raise NotImplementedError

    def softmax_logprob(self, logits, amask):
        res = {}
        for k in logits:
            logits_masked = logits[k]
            logprobs = torch.log_softmax(logits_masked, dim=-1)  # + self.placeholder
            res[k] = logprobs
        return res

    def softmax_raw(self, logits, amask):
        res = {}
        for k in logits:
            logits_masked = logits[k]
            probs = torch.softmax(logits_masked, dim=-1)  # + self.placeholder
            res[k] = probs
        return res

    def softmax_masked(self, logits, amask):
        res = {}
        for k in logits:
            block_mask = amask[k].sum(-1, keepdims=True) > 0
            logits_masked = torch.where(torch.where(block_mask.bool(), amask[k], True).bool(), logits[k], self.neginf)
            probs = torch.softmax(logits_masked, dim=-1) + self.placeholder
            res[k] = probs
        return res

    def softmax_logprob_masked(self, logits, amask):
        res = {}
        for k in logits:
            #             block_mask = amask[k].sum(-1, keepdims=True)>0
            logits_masked = logits[k] + (amask[k] + self.smallfloat).log()
            logprobs = torch.log_softmax(logits_masked, dim=-1)  # + self.placeholder
            res[k] = logprobs
        return res


## policy
class Policy_Bid(ModelBase):
    def __init__(self, in_channels_grid, in_channels_game, out_channels_unit, out_channels_factory, out_channels_bid):
        super().__init__()

        fe_bid_features = 16

        self.fe_bid = nn.Sequential(
            nn.Conv2d(in_channels_grid, 32, (1,1), padding="same"),
            nn.ReLU(),
            FeatureExtractor(32, fe_bid_features, 12, batch_norm=True)
        )

        self.policy_bid = nn.Sequential(
            nn.Conv2d(fe_bid_features, 1, (1, 1), padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48*48, out_channels_bid)
        )

    @torch.jit.export
    def forward(self, x_game, x_board):
        board = x_board.permute((0,3,1,2))
        fe_bid = self.fe_bid(board)
        logits_bid = self.policy_bid(fe_bid)
        return dict(
            bid=logits_bid,
        )


class Policy_Placement(ModelBase):
    def __init__(self, in_channels_grid, in_channels_game, out_channels_unit, out_channels_factory, out_channels_bid):
        super().__init__()

        fe_placement_features = 16

        self.fe_placement = nn.Sequential(
            nn.Conv2d(in_channels_grid, 64, (1,1), padding="same" ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            FeatureExtractor(64, fe_placement_features, 24, batch_norm=True)
        )

        self.policy_placement = nn.Sequential(
            nn.Conv2d(fe_placement_features, 1, (1, 1), padding="same"),
            Transpose((0, 2, 3, 1)),
            nn.Flatten(),
        )

    @torch.jit.export
    def forward(self, x_game, x_board):
        board = x_board.permute((0,3,1,2))
        fe_placement = self.fe_placement(board)
        logits_placement = self.policy_placement(fe_placement)
        return dict(
            placement=logits_placement,
        )


class Policy_Game(ModelBase):
    def __init__(self, in_channels_grid, in_channels_game, out_channels_unit, out_channels_factory, out_channels_bid):
        super().__init__()

        fe_game_features = 64
        fe_f_features = 16

        self.fe_game = nn.Sequential(
            nn.Conv2d(in_channels_grid, 128, (1, 1), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            FeatureExtractor(128, fe_game_features, 16, batch_norm=True)
        )

        self.fe_f = nn.Sequential(
            nn.Conv2d(in_channels_grid, 64, (1, 1), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            FeatureExtractor(64, fe_f_features, 12, batch_norm=True)
        )

        self.policy_unit = nn.Sequential(
            nn.Conv2d(fe_game_features, 64, (1, 1), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels_unit, (1, 1), padding="same"),
            Transpose((0, 2, 3, 1)),
        )

        self.policy_factory = nn.Sequential(
            nn.Conv2d(fe_f_features, 64, (1, 1), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels_factory, (1, 1), padding="same"),
            Transpose((0, 2, 3, 1)),
        )

    @torch.jit.export
    def forward(self, x_game, x_board):
        board = x_board.permute((0, 3, 1, 2))

        fe_game = self.fe_game(board)
        fe_f = self.fe_f(board)

        logits_unit = self.policy_unit(fe_game)
        logits_factory = self.policy_factory(fe_f)

        return dict(
            unit=logits_unit,
            factory=logits_factory,
        )




# agent stuff




class StatelessAgentBase:
    def __init__(self, env_cfg: EnvConfig) -> None:
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, player, step: int, obs, remainingOverageTime: int = 60):
        raise NotImplementedError

    def act(self, player, step: int, obs, remainingOverageTime: int = 60):
        raise NotImplementedError

    def update_stats(self, player, step: int, old_stats, new_stats):
        pass


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig, wrapped: StatelessAgentBase) -> None:
        self.player = player
        self.opp_player = ""
        if self.player == "player_0":
            self.opp_player = "player_1"
        else:
            self.opp_player = "player_0"
        self.env_cfg: EnvConfig = env_cfg
        self.wrapped = wrapped

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        return self.wrapped.early_setup(self.player, step, obs, remainingOverageTime)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        return self.wrapped.act(self.player, step, obs, remainingOverageTime)

    def update_stats(self, step: int, old_stats, new_stats):
        return self.wrapped.update_stats(self.player, step, old_stats, new_stats)




@dataclass
class GameStepActions:
    bid: Any = None
    placement: Any = None
    units: Dict[str, List[Any]] = None
    factories: Dict[str, Any] = None

    def merge(self):
        if self.bid:
            return self.bid
        if self.placement:
            return self.placement

        acts = {}
        if self.units:
            acts.update(self.units)
        if self.factories:
            acts.update(self.factories)
        return acts


def allocate_actions( actions_list ):
    # action = [ unit, action_ei, next_pos, prob ]

    unit_to_uix_map = {}
    pos_to_tix_map = {}

    units = []
    targets = []
    variables = []  # (unit_ix, label, target_ix, prob)

    for uname, action_ei, next_pos, pprob, z in actions_list:
        try:
            tix = pos_to_tix_map[next_pos]
        except KeyError:
            tix = len(targets)
            targets += [next_pos]
            pos_to_tix_map[next_pos] = tix

        try:
            uix = unit_to_uix_map[uname]
        except KeyError:
            uix = len(units)
            units += [uname]
            unit_to_uix_map[uname] = uix

        variables += [(uix, action_ei, tix, pprob, z)]

    cost_matrix = np.full((len(units), len(targets)), -50, dtype=np.float32)
    label_matrix = np.full((len(units), len(targets)), -1)

    for uix, lbl, tix, pprob, z in variables:
        pold = cost_matrix[uix, tix]
        p = max(-49, (np.log(pprob) + z) if pprob > 0 else -49)
        if p > pold:
            cost_matrix[uix, tix] = p
            label_matrix[uix, tix] = lbl

    # print(cost_matrix, file=sys.stderr)
    # print(label_matrix, file=sys.stderr)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    result = []
    for uix, tix in zip(row_ind, col_ind):
        label = label_matrix[uix, tix]
        cost = cost_matrix[uix, tix]
        result += [ (units[uix], targets[tix], label, cost ) ]

    return result

class ModelAgent(StatelessAgentBase):
    def __init__(self, env_cfg, fmaker: FeaturesMaker, amaker: ActionsMaker, policy_bid, policy_placement, policy_game, lookahead):
        super().__init__(env_cfg)
        self._fmaker = fmaker
        self._amaker = amaker
        self._lookahead = lookahead
        self._policy_bid = policy_bid
        self._policy_placement = policy_placement
        self._policy_game = policy_game
        pass

    def early_setup(self, player_name, step: int, obs, remainingOverageTime: int = 60):
        return self.do_act(player_name, step, obs, remainingOverageTime)

    def act(self, player_name, step: int, obs, remainingOverageTime: int = 60):
        return self.do_act(player_name, step, obs, remainingOverageTime)

    def do_act(self, player_name, step: int, obs, remainingOverageTime) -> dict:
        t0 = time.time()

        try:
            gsa = self.predict(player_name, step, obs, self._lookahead)
        except Exception as exc:
            print(f"step {step} {player_name} exception {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return {}

        if gsa.units:
            for uname in list(gsa.units.keys()):
                uactions = gsa.units[uname]
                aq = obs["units"][player_name][uname]["action_queue"]
                if len(aq) and aq[0][0] == uactions[0][0] and aq[0][1] == uactions[0][1] and aq[0][2] == uactions[0][2]:
                    del gsa.units[uname]
                    continue
                if not len(aq) and uactions[0][0] == 0 and uactions[0][1] == 0:
                    del gsa.units[uname]
                    continue
        actions = gsa.merge()

        print(f"step {step} {player_name} duration {time.time() - t0:.3f} rem {remainingOverageTime}", file=sys.stderr)
        # for uname, u in obs["units"][player_name].items():
        #     print("-U->>>", player_name, uname, u["pos"], json.dumps(np.array(u["action_queue"]).tolist()), file=sys.stderr)
        #     if uname in actions:
        #         print("<<<-", "action", json.dumps(np.array(actions[uname]).tolist()), file=sys.stderr)
        return actions

    def make_simulated_env(self):
        return LuxAI_S2(verbose=0)

    def predict(self, player_name, step, obs, lookahead) -> GameStepActions:
        player_id = int(player_name[-1])
        opponent_id = 1-player_id
        opponent_name = f'player_{opponent_id}'

        # t0 = time.time()
        final_player_actions = self.predict_iter(player_name, step, obs, self.env_cfg)
        # t1 = time.time()
        # print(f"self predict took {t1-t0}", file=sys.stderr)

        if not lookahead:
            return final_player_actions

        if not final_player_actions.units:
            return final_player_actions

        try:
            simulated_env = self.make_simulated_env()
            simulated_env.reset(seed=0)
            simulated_env.state = simulated_env.state.from_obs(obs, self.env_cfg)
            simulated_env.env_cfg = simulated_env.state.env_cfg
            simulated_env.env_cfg.verbose = 0
            simulated_env.env_steps = simulated_env.state.env_steps

            sim_obs = None
            player_sim_actions = copy.deepcopy(final_player_actions)

            for i in range(lookahead):

                if sim_obs is None: # 1st iter
                    try:
                        opponent_sim_actions = self.predict_iter(opponent_name, step, obs, self.env_cfg)
                    except Exception as exc:
                        print(f"exception simulating opponent: {exc}", file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)
                        opponent_sim_actions = GameStepActions(None, None, {}, {})
                        pass
                else:
                    opponent_sim_actions = GameStepActions(None, None, {}, {}) # self.predict_iter(opponent_name, simulated_env.env_steps, sim_obs[opponent_name], simulated_env.env_cfg)

                sim_obs, _, _, _ = simulated_env.step({player_name: player_sim_actions.merge(), opponent_name: opponent_sim_actions.merge()})

                simulated_env.env_cfg.ROBOTS["HEAVY"].ACTION_QUEUE_POWER_COST = 0
                simulated_env.env_cfg.ROBOTS["LIGHT"].ACTION_QUEUE_POWER_COST = 0

                if len(simulated_env.agents) == 0:
                    break

                player_sim_actions = self.predict_iter(player_name, simulated_env.env_steps, sim_obs[player_name], simulated_env.env_cfg)
                for k in final_player_actions.units:
                    if k in player_sim_actions.units:
                        final_player_actions.units[k] += player_sim_actions.units[k]
                    else:
                        final_player_actions.units[k] += [np.array([0, 0, 0, 0, 0, 1])]

        except Exception as exc:
            print(f"lookahead failed: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

        return final_player_actions

    @torch.no_grad()
    def predict_iter(self, player_name, step, obs, env_cfg) -> GameStepActions:
        wgs = WrappedGameState2(env_cfg, step, obs)
        full_obs = {k: v.unsqueeze(0) for k, v in self._fmaker.collect_features(wgs, player_name).items()}
        act_mask = {k: v.unsqueeze(0) for k, v in self._amaker.make_actions_mask(wgs, player_name).items()}

        if step == 0:
            logits = self._policy_bid.forward(full_obs["game"],full_obs["board"])
            mout_probs = self._policy_bid.softmax_masked(logits, act_mask)
        elif wgs.gs.real_env_steps < 0:
            logits = self._policy_placement.forward(full_obs["game"],full_obs["board"])
            mout_probs = self._policy_placement.softmax_masked(logits, act_mask)
        else:
            logits = self._policy_game.forward(full_obs["game"],full_obs["board"])
            mout_probs = self._policy_game.softmax_masked(logits, act_mask)

        return self.transform_actions(wgs, player_name, mout_probs, act_mask)

    def transform_actions(self, wgs2, player_name, mout_probs, act_mask) -> GameStepActions:
        player_id = int(player_name[-1])
        opponent_id = 1-player_id
        opponent_name = f'player_{opponent_id}'

        if wgs2.step == 0:
            probs_bid = mout_probs['bid'][0]
            sample_bid = np.argmax(probs_bid)
            act = self._amaker.bid_actions[sample_bid]
            return GameStepActions(bid=act.execute(wgs2))

        if wgs2.gs.real_env_steps < 0:
            act_placement = wgs2.gs.board.valid_spawns_mask > 0
            factories_to_place = wgs2.gs.teams[player_name].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(wgs2.gs.teams[player_name].place_first, wgs2.step)
            if factories_to_place > 0 and my_turn_to_place:
                probs_placement = mout_probs['placement'][0].view((48,48))
                masked_placement = (probs_placement+0.001) * act_placement
                place = np.unravel_index(np.argmax(masked_placement), masked_placement.shape)
                return GameStepActions(placement=dict(spawn=np.array(place), metal=150, water=150))

        if wgs2.gs.real_env_steps < 0:
            return GameStepActions(None, None, {}, {})

        actions = GameStepActions(None, None, {}, {})

        probs_unit = mout_probs['unit'][0]
        probs_factory = mout_probs['factory'][0]

        act_mask_unit = act_mask['unit'][0]
        act_mask_factory = act_mask['factory'][0]

        masked_probs_unit = probs_unit * act_mask_unit

        actions_candidates = []
        for uname, u in wgs2.gs.units[player_name].items():
            for ei in range(len(self._amaker.unit_actions)):
                act = self._amaker.unit_actions[ei]
                if not act.can_execute(u, wgs2): # or u.power < act.power_cost(u, wgs2) + u.unit_cfg.ACTION_QUEUE_POWER_COST:
                    continue

                npos = to_tuple(act.get_next_pos(u, wgs2))

                z = 0

                aq_cost = u.unit_cfg.ACTION_QUEUE_POWER_COST
                if u.power < act.power_cost(u, wgs2) + aq_cost:
                    z -= 5

                fnext = wgs2.pos_to_factory_map.get(npos)
                if fnext and to_tuple(fnext.pos) == npos and to_tuple(u.pos) == npos:
                    z -= 2

                if u.unit_type == UNIT_HEAVY:
                    z += 20

                if isinstance(act, UnitTypeMoveAction) and act._direction == 0 and u.power > 0.5 * u.unit_cfg.BATTERY_CAPACITY:
                    z -= 2

                # if isinstance(act, UnitTypePickupPowerAction) and u.power > 0.8 * u.unit_cfg.BATTERY_CAPACITY:
                #     z -= 2

        # if u.power > 0.8 * u.unit_cfg.BATTERY_CAPACITY:
        #     return False

                actions_candidates += [ (uname, ei, npos, masked_probs_unit[u.pos[0],u.pos[1],ei],z) ]

        # print(actions_candidates, file=sys.stderr)

        # t0 = time.time()
        actions_solved = allocate_actions(actions_candidates)
        # t1 = time.time()
        # if t1-t0 > 0.1:
        #     print(f"allocate_actions took {t1-t0}", file=sys.stderr)

        turn_state = TurnState(wgs2, player_name)

        # t0 = time.time()

        for uname, npos, ei, cost in actions_solved:
            u = wgs2.gs.units[player_name][uname]
            act = self._amaker.unit_actions[ei]
            if u.power >= act.power_cost(u, wgs2) + u.unit_cfg.ACTION_QUEUE_POWER_COST:
                act.pre_execute(u, wgs2, turn_state)

        next_pos_taken = set()
        for uname, npos, ei, cost in actions_solved:
            u = wgs2.gs.units[player_name][uname]
            act = self._amaker.unit_actions[ei]

            if u.power >= act.power_cost(u, wgs2) + u.unit_cfg.ACTION_QUEUE_POWER_COST:
                z = act.execute(u, wgs2, turn_state)
            else:
                z = [u.move(0)]

            next_pos_taken |= {npos}
            if z is None:
                continue

            actions.units[uname] = z

        # t1 = time.time()
        # if t1-t0 > 0.1:
        #     print(f"execute units took {t1-t0}", file=sys.stderr)

        total_heavies = 0
        total_light = 0
        for u in wgs2.gs.units[player_name].values():
            if u.unit_type == UNIT_HEAVY:
                total_heavies += 1
            else:
                total_light += 1

        for fname, f in wgs2.gs.factories[player_name].items():
            for s, ei in sorted([  (probs_factory[f.pos[0], f.pos[1], ei] * act_mask_factory[f.pos[0], f.pos[1], ei], ei) for ei in range(len(self._amaker.factory_actions)) ], reverse=True ):
                act = self._amaker.factory_actions[ei]
                if not act.can_execute(f, wgs2):
                    continue

                # if isinstance(act, FactoryBuildLightAction) and wgs2.gs.real_env_steps > 20 and total_heavies < 4*wgs2.gs.board.factories_per_team:
                #     continue

                if isinstance(act, FactoryBuildLightAction) and total_light > 5*total_heavies:
                    continue

                #
                # if isinstance(act, FactoryBuildLightAction) and wgs2.total_heavy[player_name] < wgs2.total_heavy[opponent_name]:
                #     continue

                npos = act.get_next_pos(f, wgs2)
                if npos is not None and npos in next_pos_taken:
                    continue

                z = act.execute(f, wgs2)
                if z is not None:
                    actions.factories[fname] = z

                break

        for fname, f in wgs2.gs.factories[player_name].items():
            if fname in actions.factories:
                continue

            if wgs2.gs.real_env_steps > 900:
                steps_remain = wgs2.env_cfg.max_episode_length - wgs2.gs.real_env_steps
                if f.can_water(wgs2.gs) and f.cargo.water > 20 + f.water_cost(wgs2.gs) * steps_remain:
                    actions.factories[fname] = f.water()
                    continue

        return actions




def make_model_agent(env_cfg, fname_bid, fname_placement, fname_game, la):
    fmaker = make_features_maker(env_cfg)
    amaker = make_action_maker(env_cfg)
    fmaker.enable_features(fmaker.list_feature_names())

    obs_spec = fmaker.get_obs_spec()
    policy_bid = Policy_Bid(obs_spec["board"].shape[-1],
                          obs_spec["game"].shape[-1],
                          len(amaker.unit_actions),
                          len(amaker.factory_actions),
                          len(amaker.bid_actions))
    policy_placement = Policy_Placement(obs_spec["board"].shape[-1],
                          obs_spec["game"].shape[-1],
                          len(amaker.unit_actions),
                          len(amaker.factory_actions),
                          len(amaker.bid_actions))
    policy_game = Policy_Game(obs_spec["board"].shape[-1],
                          obs_spec["game"].shape[-1],
                          len(amaker.unit_actions),
                          len(amaker.factory_actions),
                          len(amaker.bid_actions))

    with open(fname_bid, 'rb') as in_f:
        checkpoint = torch.load(in_f, map_location="cpu")
        policy_bid.load_state_dict(checkpoint["model"])
    with open(fname_placement, 'rb') as in_f:
        checkpoint = torch.load(in_f, map_location="cpu")
        policy_placement.load_state_dict(checkpoint["model"])
    with open(fname_game, 'rb') as in_f:
        checkpoint = torch.load(in_f, map_location="cpu")
        policy_game.load_state_dict(checkpoint["model"])

    policy_bid.eval()
    policy_placement.eval()
    policy_game.eval()
    return ModelAgent(env_cfg, fmaker, amaker, policy_bid, policy_placement, policy_game, la)




