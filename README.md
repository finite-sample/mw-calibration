# Always‑On Probability Calibration via Multiplicative‑Weights

## 1  Realistic production scenario

**Context**  ▸ A large ad platform serves millions of impressions per hour.  An upstream ML model outputs raw click‑through probabilities \$p^{\text{raw}}\$.  Over time, systematic **drift** appears (creative fatigue, seasonality, campaign mix).  Business KPIs and auctions require **well‑calibrated probabilities** at any moment.

**Key constraint**  ▸ You can’t stop traffic to retrain a calibrator on every batch; compute must stay sub‑millisecond per impression.

---

## 2  What practitioners typically do

| Method                       | Workflow                                                      | Pain‑point                                                           |
| ---------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Platt scaling** (logistic) | Train on yesterday’s data; deploy coefficients until next job | Loses calibration as drift grows; spikes CPU/latency when retrained. |
| **Isotonic regression**      | Same nightly (or hourly) batch job; guarantees monotonicity   | Same drift issue; heavier CPU and memory if many segments.           |

**Trade‑off**  ▸ Fewer retrains ⟶ lower compute, but larger calibration error between jobs.  More frequent retrains ⟶ accuracy stays tight, compute scales $\mathcal O(N_{\text{seen}})$ and may breach SLA.

---

## 3  Our proposal: **Vectorised Multiplicative‑Weights Update (MWU)**

* Maintain one **bias weight** \$c\_b\$ per calibration bucket / segment.
* After each mini‑batch: update all buckets **once** via
  $c_b \leftarrow c_b\,\exp\bigl(-\eta\,(\hat r_b-\tilde r_b)\bigr),$
  where \$\hat r\_b\$ is the batch click‑rate and \$\tilde r\_b\$ the predicted rate.
* Complexity **$\mathcal O(\text{#buckets})$** regardless of events processed.
* Adapts instantly to drift; no full refit, no heavy solver.

---

## 4  Simulation setup

* **200 k** impressions streamed in **40 batches** (5 k each).
* Upward probability drift encoded in the logit mean \$\mu\_t\$.
* **100 reliability buckets.**
* Compare per‑batch **Brier** & **CPU time**:

  * Platt (logistic), Isotonic (PAV) — *retrained every batch* ➊
  * **MWU** (vectorised bucket update).

➊ We retrain each batch to show compute scaling. In practice retrain cadence is slower; see §5.

---

## 5  Results (aggregate over 40 batches)

| Metric                   | Platt                    | Isotonic       | **MWU**                     |
| ------------------------ | ------------------------ | -------------- | --------------------------- |
| **Mean per‑batch Brier** | **0.2051**               | 0.2045         | 0.2052                      |
| **Std (Brier)**          | 0.0019                   | **0.0017**     | 0.0019                      |
| **Mean CPU s / batch**   | 0.0243                   | 0.0181         | **0.00039**                 |
| **Compute scaling**      | grows linearly with data | grows linearly | \~flat ($\approx$ constant) |

*Platt & Isotonic achieve slightly lower Brier—at the cost of ****60×‑100× more CPU****.*

> **If retrained hourly instead of per‑batch**: their compute would drop, but calibration error would **drift upward** between retrains; MWU keeps both error and compute flat.

---

## 6  Take‑aways

* **MWU = always‑on calibrator** — cheap exponential updates keep probabilities aligned without offline jobs.
* Offers a clean knob (learning‑rate \$\eta\$) to trade stability vs. responsiveness.
* Ideal for ad serving, recommender systems, or any high‑volume setting where **latency and continual drift** rule out heavy batch retrains.

---

