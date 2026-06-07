# Introduction to Data Science: Assignment #04
**AQI-Based Adaptive Traffic-Control System Using Reinforcement Learning**
*(Continuation of Assignment #03)*

**Student Name:** Touseeq iqbal
**Registration Number:** 2280237
**Submitted To:** Dr. Danish Mahmood
**Program:** BSSE-8, Computer Science Department, SZABIST-ISB
**GitHub Repository:** https://github.com/touseeqiqbal/AQI-Data-Science-Project

---

## 1. Connection with Assignment 3
In Assignment 3, a complete introductory data-science workflow was performed on the **Global
Urban Air Quality Index Dataset (2015-2025)**: data cleaning, exploratory visualization, AQI
category creation, and the KNN, Naive Bayes, K-Means, and PCA models. Assignment 4 continues the
**same project** by adding a simplified **Reinforcement Learning (RL)** component on top of the
already-cleaned dataset and AQI categories.

**Assignment 3 outputs reused in Assignment 4**

| Assignment 3 Output | How it is reused in Assignment 4 |
| :--- | :--- |
| Cleaned dataset (duplicates removed, medians imputed) | Acts as the **environment** the RL agent interacts with |
| Numerical `AQI` value | Mapped into the RL **states** (Low / Medium / High) |
| `AQI Category` (6 classes) | Used for interpretation and the optional state-distribution chart |
| `City` / `Country` / `Date` fields | Available for optional analysis of where/when actions are chosen |
| Pollutant features (PM2.5, PM10, CO, NO2, O3, SO2) | Available for optional state enhancement / interpretation |
| GitHub repository | Same repo; a new Assignment-4 notebook and `outputs/rl_results/` folder added |

---

## 2. RL Problem Statement
This project extends the AQI data-science analysis by designing a simplified reinforcement-learning
system for **adaptive traffic control**. The RL agent observes the AQI condition of a city and
learns whether to apply **No Restriction**, **Partial Restriction**, or a **High-Pollution Alert**.
The purpose is to understand how an agent can learn a decision policy using rewards based on
**public-health protection** and **traffic-disruption cost**.

**Main objective:** *Can a simple reinforcement learning agent learn suitable traffic-control
actions based on AQI conditions?*

---

## 3. RL Component Definitions

| RL Component | Definition in this Assignment |
| :--- | :--- |
| **Agent** | The adaptive traffic-control decision maker. |
| **Environment** | AQI conditions taken from the cleaned AQI dataset. |
| **State** | The AQI condition: *Low AQI*, *Medium AQI*, or *High AQI*. |
| **Action** | No Restriction (0), Partial Restriction (1), or High-Pollution Alert (2). |
| **Reward** | A numerical score that rewards suitable actions and penalizes unsuitable ones. |
| **Policy** | The learned mapping from each AQI state to its best action. |
| **Episode** | One training cycle in which the agent interacts with all training AQI records. |
| **Exploration** | Trying a random action (probability ε) to discover its outcome. |
| **Exploitation** | Choosing the best-known action `argmax(Q)` to maximise reward. |

---

## 4. AQI State Mapping (3-State Design)

The recommended **3-state** design is used. The continuous AQI value is binned into three
RL states.

| AQI Condition | AQI Range | RL State | State ID | Record Count |
| :--- | :--- | :--- | :--- | :--- |
| Good / Moderate | 0 - 100 | Low AQI | 0 | 328 |
| Unhealthy for Sensitive Groups / Unhealthy | 101 - 200 | Medium AQI | 1 | 137 |
| Very Unhealthy / Hazardous | 201+ | High AQI | 2 | 35 |

After the same cleaning as Assignment 3, the dataset contained **500 records** (505 raw rows minus
5 duplicates), which the agent treats as its environment.

---

## 5. Action Space

| Action ID | Action Name | Meaning |
| :--- | :--- | :--- |
| 0 | No Restriction | Normal traffic flow; no public warning required. |
| 1 | Partial Restriction | Reduce heavy traffic, advise sensitive groups, limit high-emission vehicles. |
| 2 | High-Pollution Alert | Strong warning / restriction during dangerous AQI conditions. |

---

## 6. Reward System

The reward table balances **public-health protection** against **traffic-disruption cost**.
Over-restricting during clean air is penalised, and ignoring dangerous pollution is strongly
penalised.

| AQI State | No Restriction | Partial Restriction | High-Pollution Alert | Expected Best Action |
| :--- | :--- | :--- | :--- | :--- |
| Low AQI | +10 | -2 | -5 | No Restriction |
| Medium AQI | -6 | +10 | +2 | Partial Restriction |
| High AQI | -10 | -3 | +10 | High-Pollution Alert |

---

## 7. Q-Learning Implementation Summary
A simple **tabular Q-learning** algorithm was implemented using only NumPy, Pandas, and Matplotlib.

1. Initialise a 3×3 Q-table (states × actions) with zeros.
2. Set learning parameters: `alpha = 0.1`, `gamma = 0.9`, `epsilon = 0.1`.
3. Split the records 80/20 into training (400) and testing (100) sets.
4. Train for **500 episodes**; each episode is one shuffled pass over the training records.
5. At each step select an action with the **epsilon-greedy** strategy.
6. Look up the reward in the reward table and apply the Q-learning update rule:

> `Q(s, a) = Q(s, a) + alpha * [ reward + gamma * max(Q(s', a')) - Q(s, a) ]`

7. After training, extract the best action per state with `argmax`.

---

## 8. Training and Evaluation Results

### Final Q-Table
| AQI State | No Restriction | Partial Restriction | High-Pollution Alert |
| :--- | :--- | :--- | :--- |
| Low AQI | **100.0** | 88.0 | 85.0 |
| Medium AQI | 84.0 | **100.0** | 92.0 |
| High AQI | 80.0 | 87.0 | **100.0** |

The highest value in every row lies on the diagonal, so the agent values the correct action most in
each state. (Q-values converge to 100 because the optimal fixed point is `10 / (1 - gamma) = 100`.)

### Learned Policy vs. Expected Action
| AQI State | Expected Action | Learned Action | Correct? |
| :--- | :--- | :--- | :--- |
| Low AQI | No Restriction | No Restriction | **Yes** |
| Medium AQI | Partial Restriction | Partial Restriction | **Yes** |
| High AQI | High-Pollution Alert | High-Pollution Alert | **Yes** |

### Evaluation on Held-Out Test Records (100 records)
- **Average reward after training:** **10.0** (every test record received the optimal +10 reward).
- **Action distribution:** No Restriction = 59, Partial Restriction = 32, High-Pollution Alert = 9.
- **Training reward:** rose from 3219 (episode 1) and stabilised around **~3635** (mean of last 100
  episodes), confirming convergence.

---

## 9. Required Visualizations

> *The notebook regenerates these charts on every run and saves them to `outputs/rl_results/`.*

1. **Total Reward per Episode** (`reward_plot.png`) - The total reward starts low (random
   exploration) and rises sharply within the first ~15 episodes as the agent discovers the best
   action per state, then remains high and stable around ~3,650. This shows the policy converged.
2. **Final Q-Table Heatmap** (`q_table_heatmap.png`) - The brightest cell in each row is on the
   diagonal, confirming the agent learned the correct action for every AQI state.
3. **Action Distribution** (`action_distribution.png`) - *No Restriction* is selected most often,
   *Partial Restriction* next, and *High-Pollution Alert* least, mirroring the real AQI state
   distribution (Low > Medium > High).
4. **(Optional) AQI State Distribution** (`aqi_state_distribution.png`) - Links the RL task back to
   the Assignment 3 categories and explains the action distribution: the dataset is dominated by Low
   AQI records.

---

## 10. Final RL Result Tables

| Metric | Value |
| :--- | :--- |
| Number of states used | 3 (Low / Medium / High AQI) |
| Number of actions used | 3 |
| Episodes used for training | 500 |
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.9 |
| Exploration rate (epsilon) | 0.1 |
| Average reward after training | 10.0 (greedy, test set) |
| Final learned policy | Low → No Restriction, Medium → Partial Restriction, High → High-Pollution Alert |

| AQI State | Best Action Learned by RL Agent | Explanation |
| :--- | :--- | :--- |
| Low AQI | No Restriction | Air is clean, so restricting traffic only adds disruption cost. |
| Medium AQI | Partial Restriction | Moderate pollution needs a balanced response that protects sensitive groups. |
| High AQI | High-Pollution Alert | Dangerous air requires the strongest protective action. |

---

## 11. Answers to Assignment Questions

1. **What is reinforcement learning?** A machine-learning approach where an *agent* learns by
   interacting with an *environment*, choosing *actions* and receiving *rewards*, gradually learning
   a *policy* that maximises long-term reward - no labelled answers are provided.
2. **What is the agent?** The adaptive traffic-control decision maker that selects the traffic /
   public-warning action for each AQI condition.
3. **What is the environment?** The cleaned Global Urban AQI dataset; each record presents an AQI
   condition (state) and returns a reward for the chosen action.
4. **Which AQI states did you define and why?** Three states - Low (0-100), Medium (101-200), and
   High (201+) AQI. The 3-state design is simple, interpretable, and maps cleanly to the three
   available actions.
5. **What are the three actions?** No Restriction (0), Partial Restriction (1), and High-Pollution
   Alert (2).
6. **How did you design the reward system?** Each state's correct action gives +10; over-restricting
   clean air or ignoring dangerous air gives strong negative rewards, balancing public-health
   protection against traffic-disruption cost.
7. **What does exploration vs. exploitation mean?** *Exploration* = trying a random action
   (probability ε) to discover its outcome; *exploitation* = picking the best-known action to
   maximise reward. Epsilon-greedy combines both.
8. **What did the final Q-table show?** The highest value in every row is on the diagonal,
   confirming the agent learned the optimal action per state.
9. **Which action for Low/Medium/High AQI?** Low → No Restriction, Medium → Partial Restriction,
   High → High-Pollution Alert.
10. **Did the agent make logical decisions?** Yes - all three learned actions match the expected
    public-health logic, so the learned policy is 100% correct.
11. **Limitations of this simplified RL model.** Only 3 coarse states; a static, hand-designed
    reward table; no real traffic feedback; states are treated independently (no genuine sequential
    dynamics); a fixed epsilon (a decaying schedule would converge faster); and the dataset covers
    only a few cities, limiting real-world generalisation.

---

## 12. Limitations and Conclusion

**Limitations.** The model uses a coarse 3-state representation, a fixed hand-designed reward table,
independent (non-sequential) records, a constant exploration rate, and a relatively small dataset of
~500 records from a limited set of cities.

**Conclusion.** Despite its simplicity, the Q-learning agent trained for 500 episodes learned the
**optimal policy with 100% state accuracy**: *No Restriction* for Low AQI, *Partial Restriction* for
Medium AQI, and *High-Pollution Alert* for High AQI. The positive average test reward (10.0) confirms
the policy generalises to unseen records. This demonstrates that even a basic tabular RL approach can
power a practical adaptive traffic-control system when grounded in real AQI data, directly answering
the project's main objective.

---

## 13. References and AI Tool Usage Declaration
- **Dataset:** Global Urban Air Quality Index Dataset (2015-2025), Kaggle (Syed M Talha Hasan).
- **Reference:** Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
- **Tools:** Python 3, NumPy, Pandas, Matplotlib, Seaborn, Jupyter.
- **AI Tool Usage Declaration:** An AI coding assistant was used to help scaffold the Q-learning
  code, debug the implementation, and draft this report. All RL design choices, parameters, results,
  and interpretations were reviewed and verified by the student.
