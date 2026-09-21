# MDP Specification: Narrative Arbitrage

## 1. State Space ($\mathcal{S}$)
The state consists of first- and second-order derivatives of attention signals and minimal portfolio context.

### Attention Features:
- **Wikipedia:** Pageview velocity/acceleration, edit velocity/acceleration, edit-war ratio.
- **Reddit:** Comment acceleration, post acceleration, sentiment.
- **Google Trends:** Search interest slope.
- **Cross-platform:** Sentiment divergence, lead-lag indicators.

### Portfolio Context:
- Current position size (scaled -1 to 1).
- Available cash.
- Unrealized P&L.

## 2. Action Space ($\mathcal{A}$)
- **Type:** Continuous
- **Range:** $[-1, 1]$
- **Meaning:** Target position size (where -1 is max short, 0 is flat, and 1 is max long).

## 3. Reward Function ($R$)
The reward is designed as risk-adjusted P&L net of transaction costs and slippage.

**Formula (Placeholder):**
$R_t = \text{Return}_t - \lambda \cdot \text{Volatility}_t - \text{Cost}(\Delta \text{Position})$

- $\text{Return}_t$: Change in portfolio value.
- $\lambda$: Risk aversion coefficient.
- $\text{Cost}$: Linear or quadratic cost based on the change in position size.

## 4. Transition Dynamics
Transitions are driven by the movement of the underlying asset price (external) and the agent's action to modify the position.
