# Full Rigorous Plan for a Strong ICLR Submission

## Proposed Core Idea

**Title suggestion:**
"Straightened Latent World Models: Combining Gaussian Regularization and Temporal Straightening for Stable End-to-End Planning"

**One-sentence pitch:**
We take the simple, stable two-term objective of LeWorldModel (prediction + SIGReg) and augment it with an explicit temporal straightening (curvature) loss. The resulting representations remain collapse-free, become significantly straighter, and enable reliable **gradient-based** planning that outperforms both the original CEM-based LeWM and strong baselines.

This directly addresses the two biggest practical limitations of current end-to-end JEPAs:
- Fragile / expensive planning (CEM)
- Lack of geometric regularity in the latent space for optimization

---

## 1. Technical Contributions (What Reviewers Will Care About)

1. **Unified simple objective**

   $$
   \mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_{\text{SIG}} \operatorname{SIGReg}(Z) + \lambda_{\text{curv}} \mathcal{L}_{\text{curv}}
   $$

   where $\mathcal{L}_{\text{curv}} = 1 - \text{cosine}(v_t, v_{t+1})$.

   Only two or three scalar hyperparameters.

2. **First demonstration that a fully end-to-end JEPA (no frozen DINO) can support high-performance gradient-based planning.**

3. **Empirical evidence** that SIGReg + Straightening is complementary:
   - SIGReg → prevents collapse and encourages diversity
   - Straightening → improves conditioning of the planning Hessian and makes Euclidean distance a better proxy for geodesic distance

4. **Strong practical outcome**: gradient-based open-loop and MPC planning that is both faster and higher-success than CEM on the same model.

5. **Clean ablations and analysis** (curvature measurements, loss landscape visualization, condition-number proxies, probing, violation-of-expectation).

---

## 2. Experimental Plan (Reuse Existing Code & Datasets)

**Environments** (already in LeWM):
Push-T, OGBench-Cube, Two-Room, Reacher.
Optionally add Wall / PointMaze if easy to port.

**Model**
- Keep exact LeWM architecture (ViT-Tiny encoder + Transformer predictor, 15M params).
- Only change: add the curvature loss on consecutive latent velocities.

**Training variants** (main table):

| Variant                      | SIGReg | Curvature | Planner     |
|-------------------------------|--------|-----------|-------------|
| LeWM (original)                | ✓      | ✗         | CEM         |
| LeWM + Straightening           | ✓      | ✓         | CEM         |
| LeWM + Straightening           | ✓      | ✓         | Grad (open) |
| LeWM + Straightening           | ✓      | ✓         | Grad (MPC)  |
| Ablation: only Straightening   | ✗      | ✓         | Grad        |
| Ablation: only SIGReg          | ✓      | ✗         | Grad        |

**Key metrics**
- Success rate (open-loop & MPC)
- Planning time / number of gradient steps vs CEM
- Final curvature (average cosine similarity)
- Latent distance vs geodesic correlation
- Probing accuracy of physical quantities
- Training stability (loss curves, variance across seeds)

**Must-have analyses**
- Loss landscape visualization (same style as the Straightening paper)
- Effect of $\lambda_{\text{curv}}$ (sensitivity plot)
- Does straightening hurt or help SIGReg's Gaussianity?
- Long-horizon stress test (H = 10–15)

---

## 3. Theoretical / Conceptual Angle (Nice-to-Have but Valuable)

- Empirically measure proxies for the condition number of the planning Jacobian/Hessian before and after adding straightening.
- Short discussion linking SIGReg (isotropic Gaussian) + low curvature → better-behaved controllability Gramian (inspired by Theorem 4.4 of the Straightening paper).
- Even a clean empirical verification is enough; a full proof is optional.

---

## 4. Positioning & Writing Strategy (Critical for ICLR)

**Narrative**
"Previous end-to-end JEPAs (PLDM, LeWM) solved the collapse problem but still relied on expensive sampling-based planners. Concurrent work showed that temporal straightening enables gradient-based planning, but only on top of frozen or heavily regularized encoders. We show that the two ideas are complementary and can be combined into a single simple, stable, fully end-to-end objective that supports fast gradient-based planning."

**Baselines to beat**
- Original LeWM (CEM)
- DINO-WM (frozen + CEM or Grad)
- PLDM
- Straightening paper's best numbers on overlapping environments (PushT)

**Claims to emphasize**
- Stability of training (two/three-term objective vs 6–7 terms)
- Planning speed (gradient descent is usually faster than CEM at inference)
- End-to-end from pixels with no pretrained vision backbone required

---

## 5. Risk Mitigation

| Risk                              | Mitigation |
|-----------------------------------|------------|
| Gradient descent still unstable   | Use learning-rate scheduling, gradient clipping, and multiple random restarts; report best-of-N |
| Straightening hurts diversity     | Show that SIGReg still keeps high effective rank / probing accuracy |
| Results only on easy environments | Include OGBench-Cube and longer horizons |
| Incrementalism criticism          | Emphasize that this is the first fully end-to-end (no frozen encoder) model that makes gradient planning work well |

---

## 6. Minimal Viable Strong Submission (Priority Order)

**Phase 1 (highest ROI)**
1. Add curvature loss to LeWM training.
2. Implement gradient-based open-loop + MPC planner.
3. Main comparison table (LeWM-CEM vs LeWM-Straight-CEM vs LeWM-Straight-Grad).
4. Curvature & success-rate plots.

**Phase 2**
5. Loss-landscape figures.
6. Ablations on $\lambda_{\text{curv}}$ and SIGReg strength.
7. Probing + VoE analysis.

**Phase 3 (if time)**
8. Slightly longer horizons.
9. One additional environment or real-robot transfer discussion.

---

## Final Recommendation

This combination is one of the cleanest and most feasible high-impact extensions you can do with the existing LeWM codebase. It has a clear story, addresses a real limitation (planning method), reuses all datasets, and produces both better performance and a more elegant training objective.

If executed cleanly with strong ablations and clear visualizations, it has a very realistic path to a positive ICLR outcome (poster or above).
