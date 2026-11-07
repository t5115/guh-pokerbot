"""Unchanged Fullhouse checkpoint adapted to Manchester's public API.

51 features, five actions, no eval7 or external data files at runtime.
Inputs use 100 chips per big blind; actual stack depth is preserved.
"""
import random
import numpy as np
from player_action import PlayerAction, DEALER
from fh_features import encode_state_dict
from fh_network import Network
from fh_shortstack import _push_fold
from fh_weights import load_arrays


class Model:
    def __init__(self, random_seed=20261107, use_shortstack_chart=True):
        arrays = load_arrays()
        self.equity_tables = tuple(arrays.pop(k) for k in ('paired', 'suited', 'offsuit'))
        arrays['_model_type'] = ('average_strategy' if 'strategy_w0' in arrays else 'advantage')
        self.network = Network(arrays)
        # Keep plain tuple RNG state in the submitted pickle.
        self.rng_state = random.Random(random_seed).getstate()
        self.use_shortstack_chart = use_shortstack_chart
        self.inference_calls = 0
        self.chart_calls = 0
        self.last_features = None
        self.last_strategy = None
        self.hand = []
        self.player_id = None

    def seed(self, hand, player_id):
        self.hand = list(hand)
        self.player_id = player_id
        self.initial_stacks = None
        self.board_seen = 0
        self.history_bet = 0
        self.history_last_raise = 0
        self.n_raises = 0
        self.last_raiser = None

    def _observe(self, actions, game_state):
        if self.initial_stacks is None:
            if game_state.street != 'preflop':
                raise ValueError('First decision must be preflop')
            self.initial_stacks = [s + b for s, b in zip(game_state.stacks, game_state.bets)]
            n = game_state.num_players
            sb = game_state.button if n == 2 else (game_state.button + 1) % n
            bb = (sb + 1) % n
            self.history_bet = max(min(game_state.small_blind, self.initial_stacks[sb]),
                                   min(game_state.big_blind, self.initial_stacks[bb]))
            self.history_last_raise = game_state.big_blind
        # The engine includes our previous action exactly once; do not append
        # the action returned by run() separately.
        for action in actions:
            if action.kind == 'reveal_card' and action.player_id == DEALER:
                self.board_seen += 1
                if self.board_seen in (3, 4, 5):
                    self.history_bet = 0
                    self.history_last_raise = game_state.big_blind
                    self.n_raises = 0
                    self.last_raiser = None
            elif action.kind == 'raise':
                increase = action.amount - self.history_bet
                if increase >= self.history_last_raise:
                    self.history_last_raise = increase
                    self.n_raises += 1
                    self.last_raiser = action.player_id
                self.history_bet = max(self.history_bet, action.amount)

    def encode_public_state(self, game_state):
        me = self.player_id
        scale = 100.0 / game_state.big_blind
        # Preserve Fullhouse's uncapped owed feature (call_amount is capped).
        raw_owed = max(0, game_state.current_bet - game_state.bets[me])
        pot = game_state.pot * scale
        owed = raw_owed * scale
        stack = game_state.stacks[me] * scale
        legal = [raw_owed > 0, True, False, False, False]
        if game_state.can_raise:
            after_call = pot + owed
            room = stack - owed
            minimum = game_state.last_raise * scale
            half = round(0.5 * after_call)
            legal[2] = half >= minimum and room > half
            legal[3] = after_call >= minimum and room > after_call
            opponents = [game_state.stacks[p] * scale for p in game_state.players_in_hand if p != me]
            effective = min(stack, max(opponents, default=0))
            legal[4] = effective / max(pot, 1) < 4
        return {
            'your_cards': [c.rank + c.suit.lower() for c in self.hand],
            'community_cards': [c.rank + c.suit.lower() for c in game_state.board],
            'seat_to_act': me, 'dealer': game_state.button,
            'street': game_state.street, 'pot': pot,
            'your_stack': stack, 'amount_owed': owed,
            'current_bet': game_state.current_bet * scale,
            'your_bet_this_street': game_state.bets[me] * scale,
            'can_check': raw_owed == 0,
            'players': [dict(seat=p, stack=game_state.stacks[p] * scale,
                             bet_this_street=game_state.bets[p] * scale,
                             is_folded=game_state.folded[p])
                        for p in range(game_state.num_players)],
            'invested': {p: (self.initial_stacks[p] - game_state.stacks[p]) * scale
                         for p in range(game_state.num_players)},
            'n_raises': self.n_raises, 'last_raiser': self.last_raiser,
            'legal_mask': legal,
        }

    def _raise(self, game_state, amount):
        if not game_state.can_raise:
            return PlayerAction.call(self.player_id)
        low, high = game_state.raise_range
        return PlayerAction.raise_to(self.player_id, int(min(high, max(low, round(amount)))))

    def run(self, actions, game_state):
        if not 2 <= game_state.num_players <= 6:
            raise ValueError('This checkpoint supports two to six seats')
        self._observe(actions, game_state)
        state = self.encode_public_state(game_state)
        if self.use_shortstack_chart and game_state.street == 'preflop':
            chart = _push_fold(state)
            if chart is not None:
                self.chart_calls += 1
                if chart['action'] == 'fold':
                    return PlayerAction.fold(self.player_id)
                if chart['action'] == 'all_in':
                    return self._raise(game_state, game_state.max_raise_to)
                return PlayerAction.call(self.player_id)
        features, legal = encode_state_dict(state, self.equity_tables)
        strategy = self.network.strategy(features, legal)
        if not np.isfinite(strategy).all() or not np.isclose(strategy.sum(), 1):
            raise ValueError('Invalid model probabilities')
        self.inference_calls += 1
        self.last_features = features
        self.last_strategy = strategy
        rng = random.Random()
        rng.setstate(self.rng_state)
        draw = rng.random()
        self.rng_state = rng.getstate()
        action_id = min(int(np.searchsorted(np.cumsum(strategy), draw, side='right')), 4)
        if not legal[action_id]:
            action_id = int(np.flatnonzero(legal)[-1])
        if action_id == 0:
            return PlayerAction.fold(self.player_id)
        if action_id == 1:
            return PlayerAction.call(self.player_id)
        if action_id == 4:
            return self._raise(game_state, game_state.max_raise_to)
        fraction = 0.5 if action_id == 2 else 1.0
        scale = 100.0 / game_state.big_blind
        increment = round(fraction * (state['pot'] + state['amount_owed'])) / scale
        return self._raise(game_state, game_state.current_bet + increment)

    def end_game(self, actions, game_state):
        pass
