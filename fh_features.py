# Adapted from advitrocks9/fullhouse-bot (MIT, Advit Arora 2026).
# See FULLHOUSE_LICENSE.txt. Original weights and feature layout preserved.
import numpy as np
from card_utils import Card, evaluate

FEATURE_DIM = 51
N_ACTIONS = 5
STARTING_STACK = 10_000
LEGAL_MASK_OFFSET = 46

def hand_score(cards):
    """Manchester's exact best-five rank packed in eval7's numeric format."""
    ranked = evaluate(cards)
    score = ranked[0] << 24
    for i, rank in enumerate(ranked[1:]):
        score |= (rank - 2) << (16 - 4 * i)
    return score

def _evaluate_hand_strength(c0, c1, board_idx):
    cards = [Card("23456789TJQKA"[c % 13], "CDHS"[c // 13])
             for c in [c0, c1] + board_idx]
    return hand_score(cards) / 135004160.0

def count_straight_out_ranks(rank_present: list[bool]) -> int:
    """Count distinct ranks that would complete a straight."""
    out_ranks = set()
    for start in range(9):
        present_count = 0
        missing = []
        for r in range(start, start + 5):
            if rank_present[r]:
                present_count += 1
            else:
                missing.append(r)
        if present_count == 4:
            out_ranks.update(missing)
    # Wheel: A-2-3-4-5 (ranks 12, 0, 1, 2, 3)
    wheel = [12, 0, 1, 2, 3]
    present_count = sum(1 for r in wheel if rank_present[r])
    if present_count == 4:
        for r in wheel:
            if not rank_present[r]:
                out_ranks.add(r)
    return len(out_ranks)

def detect_draws(
    hole_indices: list[int],
    board_indices: list[int],
    street: int,
) -> tuple[bool, bool, bool, int, bool]:
    """Detect draws from card indices (0-51).

    Returns (flush_draw, oesd, gutshot, draw_outs, nut_flush_draw).
    All draw indicators are False on the river (street == 3).
    """
    if street >= 3 or not board_indices:
        return False, False, False, 0, False

    all_cards = hole_indices + board_indices

    # Flush draw
    suit_counts = [0, 0, 0, 0]
    for c in all_cards:
        suit_counts[c // 13] += 1
    max_suit_count = max(suit_counts)
    flush_draw = max_suit_count == 4
    flush_suit = suit_counts.index(max_suit_count) if flush_draw else -1

    # Nut flush draw: hero holds highest card of the flush suit
    nut_flush_draw = False
    if flush_draw:
        flush_cards_on_board = [c % 13 for c in board_indices if c // 13 == flush_suit]
        flush_cards_hero = [c % 13 for c in hole_indices if c // 13 == flush_suit]
        if flush_cards_hero:
            all_flush_ranks = flush_cards_on_board + flush_cards_hero
            nut_flush_draw = max(flush_cards_hero) == max(all_flush_ranks)

    # Straight draw
    rank_present = [False] * 13
    for c in all_cards:
        rank_present[c % 13] = True

    straight_out_ranks = count_straight_out_ranks(rank_present)
    oesd = straight_out_ranks >= 2
    gutshot = straight_out_ranks == 1

    # Count total outs
    flush_outs = 9 if flush_draw else 0
    straight_outs = straight_out_ranks * 4
    overlap = min(straight_out_ranks, 2) if flush_draw and straight_out_ranks > 0 else 0
    draw_outs = flush_outs + straight_outs - overlap

    return flush_draw, oesd, gutshot, draw_outs, nut_flush_draw

def board_texture(board_indices: list[int]) -> tuple[bool, bool, float, bool]:
    """Compute board texture features.

    Returns (board_paired, board_monotone, board_wet, straight_possible).
    """
    if not board_indices:
        return False, False, 0.0, False

    ranks = [c % 13 for c in board_indices]
    suits = [c // 13 for c in board_indices]

    # Paired
    rank_set = set(ranks)
    paired = len(rank_set) < len(ranks)

    # Monotone: all board cards share a suit
    monotone = len(set(suits)) == 1

    # Wet: fraction of board card pairs within 2 ranks of each other
    n = len(ranks)
    if n < 2:
        wet = 0.0
    else:
        close_pairs = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if abs(ranks[i] - ranks[j]) <= 2:
                    close_pairs += 1
        wet = close_pairs / total_pairs

    # Straight possible: 3+ board cards fit in one 5-rank window
    rank_present = [False] * 13
    for r in ranks:
        rank_present[r] = True
    straight_possible = False
    for start in range(9):
        count = sum(1 for r in range(start, start + 5) if rank_present[r])
        if count >= 3:
            straight_possible = True
            break
    if not straight_possible:
        # Wheel
        wheel_count = sum(1 for r in (12, 0, 1, 2, 3) if rank_present[r])
        if wheel_count >= 3:
            straight_possible = True

    return paired, monotone, wet, straight_possible

def encode_state_dict(
    state: dict,
    equity_tables: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    eq_paired, eq_suited, eq_offsuit = equity_tables

    _RANK_ORDER = "23456789TJQKA"
    _SUIT_ORDER = "cdhs"

    def card_to_idx(card_str: str) -> int:
        return _SUIT_ORDER.index(card_str[1]) * 13 + _RANK_ORDER.index(card_str[0])

    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    my_cards = state["your_cards"]
    community = state.get("community_cards", [])
    players = state.get("players", [])
    my_seat = state["seat_to_act"]
    n_players = len(players)
    dealer = state["dealer"]
    street_name = state.get("street", "preflop")
    street = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}[street_name]
    pot = state.get("pot", 0)
    my_stack = state.get("your_stack", STARTING_STACK)
    to_call = state.get("amount_owed", 0)
    current_bet = state.get("current_bet", 0)

    c0 = card_to_idx(my_cards[0])
    c1 = card_to_idx(my_cards[1])
    r0, r1 = c0 % 13, c1 % 13
    s0, s1 = c0 // 13, c1 // 13
    hi, lo = max(r0, r1), min(r0, r1)
    board_idx = [card_to_idx(c) for c in community]

    if r0 == r1:
        features[0] = eq_paired[r0]
    elif s0 == s1:
        features[0] = eq_suited[hi, lo]
    else:
        features[0] = eq_offsuit[hi, lo]

    if street > 0 and community:
        features[1] = _evaluate_hand_strength(c0, c1, board_idx)

    fd, oesd, gs, outs, nfd = detect_draws([c0, c1], board_idx, street)
    features[2] = 1.0 if fd else 0.0
    features[3] = 1.0 if oesd else 0.0
    features[4] = 1.0 if gs else 0.0
    features[5] = outs / 20.0
    features[6] = 1.0 if nfd else 0.0

    features[7] = hi / 12.0
    features[8] = lo / 12.0
    features[9] = 1.0 if s0 == s1 else 0.0
    features[10] = 1.0 if r0 == r1 else 0.0
    features[11] = (hi - lo - 1) / 12.0 if hi != lo else 0.0

    if board_idx:
        bp, bm, bw, sp = board_texture(board_idx)
        features[12] = 1.0 if bp else 0.0
        features[13] = 1.0 if bm else 0.0
        board_ranks = [b % 13 for b in board_idx]
        features[14] = sum(1 for br in board_ranks if br > hi) / 5.0
        features[15] = bw
        features[16] = 1.0 if sp else 0.0

    features[17 + street] = 1.0
    pos = (my_seat - dealer) % max(n_players, 2)
    features[21 + min(pos, 5)] = 1.0

    features[27] = pot / STARTING_STACK
    features[28] = my_stack / STARTING_STACK

    active_stacks = [p.get("stack", 0) for p in players if not p.get("is_folded")]
    min_stack = min(active_stacks) if active_stacks else 0
    spr = min_stack / max(pot, 1)
    features[29] = min(max(spr, 0.0), 10.0) / 10.0

    features[30] = (
        to_call / (pot + to_call) if to_call > 0 and (pot + to_call) > 0 else 0.0
    )

    opp_stacks = sorted(
        [
            p.get("stack", 0)
            for p in players
            if p.get("seat") != my_seat and not p.get("is_folded")
        ],
        reverse=True,
    )
    for i in range(min(len(opp_stacks), 5)):
        features[31 + i] = opp_stacks[i] / STARTING_STACK

    n_raises = state["n_raises"]
    last_raiser = state["last_raiser"]
    has_aggressor = last_raiser is not None

    features[36] = min(n_raises, 4) / 4.0
    features[37] = 1.0 if has_aggressor else 0.0
    features[38] = 1.0 if last_raiser == my_seat else 0.0
    features[39] = state.get("your_bet_this_street", 0) / STARTING_STACK
    hand_invested = state["invested"]
    invested = hand_invested[my_seat]
    features[40] = invested / STARTING_STACK
    features[41] = current_bet / max(pot, 1)
    features[42] = to_call / max(my_stack, 1)

    max_vill_bets = 0
    max_vill_invested = 0
    for p in players:
        if p.get("seat") != my_seat and not p.get("is_folded"):
            max_vill_bets = max(max_vill_bets, p.get("bet_this_street", 0))
            seat = p["seat"]
            if hand_invested:
                vi = hand_invested.get(seat, 0)
            else:
                vi = STARTING_STACK - p.get("stack", STARTING_STACK)
            if vi > max_vill_invested:
                max_vill_invested = vi
    features[43] = max_vill_bets / STARTING_STACK
    features[44] = max_vill_invested / STARTING_STACK

    n_active = sum(1 for p in players if not p.get("is_folded"))
    features[45] = n_active / 6.0

    legal_mask = np.asarray(state["legal_mask"], dtype=np.bool_)
    for a in range(N_ACTIONS):
        features[LEGAL_MASK_OFFSET + a] = float(legal_mask[a])

    return features, legal_mask
