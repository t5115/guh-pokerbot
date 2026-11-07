# Adapted from advitrocks9/fullhouse-bot (MIT, Advit Arora 2026).
# See FULLHOUSE_LICENSE.txt. Original weights and feature layout preserved.
BIG_BLIND = 100
POSITIONS_6MAX = ("BTN", "SB", "BB", "LJ", "HJ", "CO")
_RANK_VAL = {r: i for i, r in enumerate("23456789TJQKA")}

def _hand_class(cards: list[str]) -> str:
    r0, s0 = cards[0][0], cards[0][1]
    r1, s1 = cards[1][0], cards[1][1]
    if _RANK_VAL[r0] < _RANK_VAL[r1]:
        r0, r1 = r1, r0
        s0, s1 = s1, s0
    if r0 == r1:
        return r0 + r1
    return r0 + r1 + ("s" if s0 == s1 else "o")


def _position(state: dict) -> str:
    n = len(state.get("players", []))
    seat = state["seat_to_act"]
    dealer = state["dealer"]
    if n == 2:
        return "BTN" if seat == dealer else "BB"
    if n == 3:
        return ("BTN", "SB", "BB")[(seat - dealer) % n]
    if n == 4:
        return ("BTN", "SB", "BB", "CO")[(seat - dealer) % n]
    if n == 5:
        return ("BTN", "SB", "BB", "HJ", "CO")[(seat - dealer) % n]
    if n >= 6:
        offset = (seat - dealer) % n
        return POSITIONS_6MAX[offset]
    offset = (seat - dealer) % n
    if offset == 0:
        return "BTN"
    if offset == n - 1:
        return "BB"
    if offset == n - 2:
        return "SB"
    return "CO"


_PF_TIER1 = {
    "AA",
    "KK",
    "QQ",
    "JJ",
    "TT",
    "99",
    "88",
    "77",
    "AKs",
    "AKo",
    "AQs",
    "AQo",
    "AJs",
    "ATs",
    "AJo",
    "KQs",
    "KJs",
    "KQo",
    "ATo",
    "A9s",
    "A8s",
    "A7s",
    "A6s",
    "A5s",
    "A4s",
    "A3s",
    "A2s",
}
_PF_TIER2 = _PF_TIER1 | {
    "66",
    "55",
    "44",
    "33",
    "22",
    "KTs",
    "K9s",
    "K8s",
    "QJs",
    "QTs",
    "Q9s",
    "JTs",
    "J9s",
    "T9s",
    "T8s",
    "98s",
    "87s",
    "76s",
    "KJo",
    "KTo",
    "QJo",
    "QTo",
    "JTo",
    "A9o",
    "A8o",
}
_PF_TIER3 = _PF_TIER2 | {
    "K7s",
    "K6s",
    "K5s",
    "K4s",
    "K3s",
    "K2s",
    "Q8s",
    "Q7s",
    "Q6s",
    "J8s",
    "J7s",
    "T7s",
    "97s",
    "86s",
    "75s",
    "65s",
    "54s",
    "K9o",
    "K8o",
    "Q9o",
    "J9o",
    "T9o",
    "98o",
    "87o",
    "A7o",
    "A6o",
    "A5o",
    "A4o",
    "A3o",
    "A2o",
}
_PF_CALL_TIGHT = {
    "AA",
    "KK",
    "QQ",
    "JJ",
    "TT",
    "99",
    "88",
    "77",
    "AKs",
    "AKo",
    "AQs",
    "AQo",
    "AJs",
    "ATs",
    "A9s",
    "KQs",
    "KJs",
    "KTs",
    "QJs",
}
_PF_CALL_MEDIUM = _PF_CALL_TIGHT | {
    "66",
    "55",
    "44",
    "AJo",
    "ATo",
    "A8s",
    "A7s",
    "A6s",
    "A5s",
    "KQo",
    "KJo",
    "QTs",
    "JTs",
    "T9s",
}
_PF_CALL_WIDE = _PF_CALL_MEDIUM | {
    "33",
    "22",
    "A4s",
    "A3s",
    "A2s",
    "K9s",
    "QJo",
    "Q9s",
    "J9s",
    "A9o",
    "A8o",
    "A7o",
}


def _push_fold(state: dict) -> dict | None:
    stack = state["your_stack"]
    players = state.get("players", [])
    my_seat = state["seat_to_act"]
    max_opp = max(
        (
            p.get("stack", 0) or 0
            for p in players
            if p.get("seat") != my_seat and not p.get("is_folded")
        ),
        default=stack,
    )
    eff_bb = min(stack, max_opp) / BIG_BLIND
    if eff_bb > 12:
        return None

    hclass = _hand_class(state["your_cards"])
    pos = _position(state)
    owed = state.get("amount_owed", 0)

    if owed > 0:
        pot_odds = owed / (state["pot"] + owed) if (state["pot"] + owed) > 0 else 1.0
        if hclass in _PF_TIER1:
            return {"action": "all_in"}
        if eff_bb <= 6:
            call_range = _PF_CALL_MEDIUM if pot_odds < 0.35 else _PF_CALL_TIGHT
        elif eff_bb <= 10:
            call_range = _PF_CALL_WIDE if pot_odds < 0.35 else _PF_CALL_MEDIUM
        else:
            call_range = _PF_CALL_WIDE
        if hclass in call_range:
            return {"action": "call"}
        return {"action": "fold"}

    push_range = (
        _PF_TIER1
        if pos in ("LJ", "HJ")
        else (
            _PF_TIER2
            if pos == "CO"
            else _PF_TIER3
            if pos in ("BTN", "SB")
            else _PF_CALL_WIDE
        )
    )
    if hclass in push_range:
        return {"action": "all_in"}
    if state.get("can_check", False):
        return {"action": "check"}
    return {"action": "fold"}


