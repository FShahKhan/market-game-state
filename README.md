# Market Game State

This project applies **game theory** and **Quantitative Rational Expectations (QRE)** inference to financial markets to detect which *strategic game regime* the market is in—such as the **Prisoner's Dilemma**, **Chicken**, or **Battle of the Sexes**.

We use price return data from major ETFs like **SPY** and **QQQ** as a proxy for the behavior of "players" in the market, and use outcome patterns to infer strategic preferences and regime types.

## What This Does

- Classifies daily returns of SPY and QQQ into strategy outcomes:
  - `LL` → SPY Long, QQQ Long
  - `LS` → SPY Long, QQQ Short
  - `SL` → SPY Short, QQQ Long
  - `SS` → SPY Short, QQQ Short
- Infers **payoff structures** for two players based on observed market behavior.
- Uses **Quantal Response Equilibrium (QRE)** to estimate **strategic preferences** under noisy decision-making.
- Matches these preferences to canonical 2x2 games:
  - Prisoner's Dilemma
  - Chicken
  - Battle of the Sexes
- Outputs:
  - Best-fitting game regime
  - Preference rankings
  - Probabilities for each game type using softmax scores

## How It Works

1. **Data**:
   - Downloads daily price data using `yfinance` (e.g., SPY, QQQ)
   - Computes **daily returns** as % changes
   - Classifies outcomes into one of four action pairs: `LL`, `LS`, `SL`, `SS`
2. **Model**:
   - Uses observed frequencies of outcomes
   - Estimates latent payoffs via **KL divergence minimization**
   - Applies **QRE** to account for bounded rationality
3. **Game Inference**:
   - Ranks inferred preferences
   - Matches against known game types
   - Assigns match score and softmax-based probability

## What Do `LL`, `LS`, `SL`, `SS` Mean?

These represent the strategy combinations of two market players (or proxies like SPY and QQQ):

| Label | Meaning              | Interpretation                  |
|-------|----------------------|----------------------------------|
| LL    | Long SPY, Long QQQ   | Both assets had positive returns |
| LS    | Long SPY, Short QQQ  | SPY up, QQQ down                 |
| SL    | Short SPY, Long QQQ  | SPY down, QQQ up                 |
| SS    | Short SPY, Short QQQ | Both assets had negative returns |

**Important**:  
Short returns aren’t inherently negative.  
The **sign** of a return only indicates whether the asset’s price went **up or down**.  
A trader holding a short position profits when the return is negative, but in the data, the negative value simply means the price fell.

## Example Output

```
=== Inferred Preference Ranking ===
Player I: ['SL', 'LL', 'SS', 'LS']
Player II: ['LS', 'LL', 'SS', 'SL']

=== Best Matching Game ===
Game Type: Prisoners Dilemma
Match Score: 6 / 8

=== Game Type Probabilities (Softmax) ===
Prisoners Dilemma: 0.735
Chicken: 0.194
Battle of the Sexes: 0.071
```

## How To Run

**Google Colab (recommended)**:
-[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FShahKhan/market-game-state/blob/main/market_game_state.ipynb)


## License

This project is under a **Non-Commercial License**:  
You are free to use, modify, and extend this code for research or personal use —  
but **not for commercial redistribution** without permission.
