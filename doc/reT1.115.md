## Regarding Theorem 1.115 from the Seven Sketches

Strategic preorders<em>‼︎</em>

- $P$: preorder of strategies (e.g., ordered by risk)
- $Q$: be a preorder of game states (e.g., ordered by payoff)
- A function $f : P \to Q$ that finds the *best possible outcome* for a given strategy is **Left Adjoint**
- Due to T1.115 this is equivalent to $f$ preserving **joins** 
  - **Example:** being logical about "OR" choices, like "what's my best outcome if I can use strategy A *or* strategy B?"
- A function $g: Q \to P$ that finds the **least risky strategy** required to **guarantee** a certain outcome is a **Right Adjoint**.
- Due to T1.115, this is equivalent to $g$ preserving **meets**.
  - **Example**: being logical about "AND" requirements, like "what is the single safest strategy I must commit to if I want to guarantee I achieve an outcome of *at least* A **AND** an outcome of *at least* B?"