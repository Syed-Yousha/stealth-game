**Project Description**

The **AI-Based Stealth Spy Game** is a Python prototype in which the player must navigate a grid-based environment without being detected by AI-controlled guards. Rather than relying on fixed patrol routes, these guards adapt over time by:

* **Predicting** the player’s next moves via a lightweight supervised model (e.g., decision tree) trained on past play sequences.
* **Re-routing** patrols toward player “hotspots” discovered through unsupervised clustering of historical positions.
* **Reacting** in real time to line-of-sight or noise distractions (e.g., thrown rocks) via simple heuristics.

Players can move in four directions and create distractions. Guards switch between patrol, prediction-biased movement, and direct chase modes, with a growing “suspicion level” that decays if no further cues are detected. The entire game runs either in a terminal window (using `curses` or print updates) or a minimal Pygame interface, emphasizing clean AI logic over graphical complexity.

---

**Project Context**

* **Course:** Artificial Intelligence (Instructor: Ms. Alishba Subhani)

* **Team:**

  * Daniyal Hussain (22K-5118)
  * Taha Farooqui  (22K-5020)
  * Yousha Mehdi   (22K-6007)

* **Motivation:** Traditional stealth games often suffer from predictable AI patterns. By introducing simple machine-learning components, we aim to create a more engaging and replayable stealth experience without heavy compute or graphics.

* **Scope:**

  * **Language:** Python 3.x
  * **AI Libraries:** `numpy`, `scikit-learn` (for data handling, supervised classification, clustering)
  * **Rendering Options:** Terminal (`curses`/print) or basic 2D window via `pygame`

* **Milestones:**

  1. Map & movement mechanics
  2. Data-logging & ML model training scripts
  3. In-game integration of prediction and clustering models
  4. Heuristic-based line-of-sight and noise reactions
  5. Difficulty scaling and documentation

This context anchors the project within your AI course, clarifies the technical stack, and outlines why and how the adaptive AI components will improve the stealth gameplay.
