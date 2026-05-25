import numpy as np
from typing import List
from agents.base_agent import BaseAgent

ACTION_FOLD        = 0
ACTION_CHECK_CALL  = 1
ACTION_RAISE_MIN   = 2
ACTION_RAISE_50    = 3
ACTION_RAISE_75    = 4
ACTION_RAISE_POT   = 5
ACTION_RAISE_OVERBET = 6
ACTION_ALL_IN      = 7

# obs layout (from poker_env.py encode_obs)
# [0:52]   hole cards
# [52:104] board
# 104      pot ratio
# 105      my stack ratio
# 106      opp stack ratio
# [107:111] street one-hot
# 111      position
# 112      SPR
# [113:122] last 3 actions
# [122:127] opp stats: [vpip, pfr, af_norm, fold_cbet, wtsd]
_POT_IDX   = 104
_STK_IDX   = 105
_OPP_VPIP  = 122   # opponent VPIP  (0~1)
_OPP_PFR   = 123   # opponent PFR   (0~1)
_OPP_AF    = 124   # opponent AF normalised (0~1, maps 0-5 -> 0-1)
_OPP_FOLD  = 125   # fold-to-cbet   (0~1)
_OPP_WTSD  = 126   # went-to-showdown (0~1)

# 對手 AF 閨値：> 0.6 認定激進型，啟動 trap play
_AGGRO_THRESHOLD = 0.6


class StrategicAgent(BaseAgent):
    """
    策略型 Agent。

    修正：
    - reward shaping 改用 obs[122:127] 的真實對手統計（不再用乾擾的 index 126）
    - self-play 下 obs 的對手統計是實際記錄的，trap play 才有意義
    - 對手 fold 率高時（fold_cbet > 0.6）強化 bluff reward
    """

    def __init__(self, obs_size: int, action_size: int,
                 mc_simulations: int = 200, **kwargs):
        super().__init__(obs_size, action_size, **kwargs)
        self.mc_simulations      = mc_simulations
        self.equity_threshold_raise = 0.65
        self.equity_threshold_call  = 0.40

    # ------------------------------------------------------------------
    # reward shaping
    # ------------------------------------------------------------------
    def compute_reward_shaping(
        self, action: int, obs: np.ndarray, base_reward: float
    ) -> float:
        pot_ratio = float(np.clip(obs[_POT_IDX], 0, 1))
        my_stack  = float(np.clip(obs[_STK_IDX], 0, 1))

        # 安全取出對手統計（obs 小於附加 index 時防御）
        def _get(idx, default=0.5):
            return float(np.clip(obs[idx], 0, 1)) if len(obs) > idx else default

        opp_vpip  = _get(_OPP_VPIP)
        opp_af    = _get(_OPP_AF)     # 0=very passive, 1=very aggressive
        opp_fold  = _get(_OPP_FOLD)   # fold-to-cbet rate

        shaping = base_reward
        opp_is_aggro = opp_af > _AGGRO_THRESHOLD
        opp_folds_a_lot = opp_fold > 0.6

        # ---- Trap play 模式（對手激進） ----
        if opp_is_aggro:
            # check/call 裝弱 → 小 bonus
            if action == ACTION_CHECK_CALL:
                shaping += 0.08 * pot_ratio
            # 對手 all-in 後呼叫且贏 → trap 成功大 bonus
            if (action == ACTION_CHECK_CALL
                    and base_reward > 0 and pot_ratio > 0.5):
                shaping += 0.25 * pot_ratio
            # 激進對手面前 fold → 懲罰
            if action == ACTION_FOLD and pot_ratio > 0.3:
                shaping -= 0.15 * pot_ratio

        # ---- Bluff 模式（對手 fold 率高） ----
        elif opp_folds_a_lot:
            # 對 fold-heavy 對手 raise → bonus
            if action >= ACTION_RAISE_MIN:
                shaping += 0.12 * pot_ratio
            # 恭順 check/call 而不 raise → 税（应該利用對手 fold 率）
            if action == ACTION_CHECK_CALL and pot_ratio > 0.2:
                shaping -= 0.05 * pot_ratio

        # ---- 正常 pot odds 模式 ----
        else:
            if action == ACTION_FOLD and pot_ratio > 0.3:
                shaping -= 0.2 * pot_ratio
            if action >= ACTION_RAISE_MIN and my_stack > 0.5:
                shaping += 0.05 * pot_ratio

        return shaping
