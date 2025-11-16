from __future__ import annotations

from .utils import chess_manager, GameContext
from .policy_engine import load_policy_engine


# Load-once policy engine (hot reload will rerun module scope automatically)
policy_engine = load_policy_engine()


@chess_manager.entrypoint
def choose_move(ctx: GameContext):
    """Called by the ChessHacks backend whenever our bot must move."""
    best_move, move_probs = policy_engine.select_move(ctx.board)
    ctx.logProbabilities(move_probs)
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Reset hook for new games (currently stateless model)."""
    ctx.log("Resetting game context")
