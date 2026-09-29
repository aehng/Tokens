# Canonical Predictor V2 Candidate Recall (DEV)

- Scope: `DEV` (135 / 135 DEV prompts)
- Dataset SHA-256: `3a8f59791f6fd06479b2b2869d09d71b57bd1e05b897740b70f18b1941f22ee6`
- DEV split SHA-256: `ef52826e58e0a3a0924e8856b781d177253f0960e897f73f42770f7f5e939e67`
- Startup to generation-ready (first invocation): 355.681 ms
- Candidate generator freeze: **not frozen**; offline DEV evidence must be combined with a small live integration check before freeze.
- Global oracle values retain solver lower/upper bounds; no unproven exact ceiling is inferred.
- Phrase-occurrence diagnosis uses each K=32 global-oracle incumbent; it does not prove phrases impossible to predict from prompt context.

| Strategy | Pool | K=32 capture interval | Global steps LB–UB | Pool steps LB–UB | CPU p50/p90/p99 ms | Useful phrases in pool / total | Rank ≤32 | Absent | Dead pool % | Index bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 256 | 0.353–0.622 | 17093–29574 | 10436–10624 | 2.454/3.922/5.045 | 713/4112 | 205 | 3399 | 73.5% | 3822460 |
| baseline | 512 | 0.377–0.672 | 17093–29574 | 11140–11486 | 3.101/8.664/18.764 | 757/4112 | 205 | 3355 | 77.4% | 3822460 |
| baseline | 1024 | 0.377–0.672 | 17093–29574 | 11159–11494 | 2.371/6.303/15.352 | 757/4112 | 205 | 3355 | 77.9% | 3822460 |
| baseline | 2048 | 0.378–0.673 | 17093–29574 | 11187–11501 | 2.691/4.230/6.284 | 757/4112 | 205 | 3355 | 77.9% | 3822460 |
| expanded_associations | 256 | 0.359–0.626 | 17093–29574 | 10618–10692 | 5.034/7.937/11.788 | 720/4112 | 133 | 3392 | 73.8% | 3822460 |
| expanded_associations | 512 | 0.400–0.717 | 17093–29574 | 11827–12249 | 4.450/7.795/11.101 | 824/4112 | 133 | 3288 | 81.0% | 3822460 |
| expanded_associations | 1024 | 0.406–0.732 | 17093–29574 | 11993–12506 | 4.084/6.473/9.423 | 830/4112 | 133 | 3282 | 83.3% | 3822460 |
| expanded_associations | 2048 | 0.406–0.732 | 17093–29574 | 11994–12508 | 4.077/6.351/9.665 | 830/4112 | 133 | 3282 | 83.4% | 3822460 |
| suffix_conditioned | 256 | 0.357–0.623 | 17093–29574 | 10546–10657 | 2.613/4.039/4.913 | 709/4112 | 222 | 3403 | 74.0% | 3822460 |
| suffix_conditioned | 512 | 0.379–0.670 | 17093–29574 | 11194–11451 | 2.490/4.218/6.509 | 755/4112 | 222 | 3357 | 77.3% | 3822460 |
| suffix_conditioned | 1024 | 0.379–0.671 | 17093–29574 | 11220–11475 | 2.506/3.675/4.894 | 757/4112 | 222 | 3355 | 77.5% | 3822460 |
| suffix_conditioned | 2048 | 0.379–0.671 | 17093–29574 | 11220–11475 | 2.458/3.890/5.692 | 757/4112 | 222 | 3355 | 77.5% | 3822460 |
| sparse_lexical | 256 | 0.357–0.630 | 17093–29574 | 10547–10773 | 8.863/10.654/13.541 | 702/4112 | 172 | 3410 | 75.9% | 3822460 |
| sparse_lexical | 512 | 0.400–0.723 | 17093–29574 | 11815–12351 | 8.665/10.381/13.146 | 793/4112 | 172 | 3319 | 80.6% | 3822460 |
| sparse_lexical | 1024 | 0.401–0.726 | 17093–29574 | 11858–12401 | 8.716/10.454/13.443 | 797/4112 | 172 | 3315 | 81.3% | 3822460 |
| sparse_lexical | 2048 | 0.401–0.726 | 17093–29574 | 11857–12401 | 8.583/10.326/12.976 | 797/4112 | 172 | 3315 | 81.3% | 3822460 |

## Domain capture at K=32

| Strategy | Pool | Code | Reasoning | Instruction |
|---|---:|---:|---:|---:|
| baseline | 256 | 0.321–0.625 | 0.499–0.805 | 0.216–0.368 |
| baseline | 512 | 0.341–0.671 | 0.538–0.886 | 0.225–0.383 |
| baseline | 1024 | 0.342–0.669 | 0.540–0.889 | 0.225–0.383 |
| baseline | 2048 | 0.342–0.672 | 0.543–0.888 | 0.225–0.383 |
| expanded_associations | 256 | 0.324–0.632 | 0.493–0.779 | 0.241–0.409 |
| expanded_associations | 512 | 0.364–0.725 | 0.550–0.907 | 0.263–0.448 |
| expanded_associations | 1024 | 0.366–0.731 | 0.564–0.941 | 0.264–0.449 |
| expanded_associations | 2048 | 0.366–0.731 | 0.564–0.941 | 0.264–0.449 |
| suffix_conditioned | 256 | 0.318–0.620 | 0.503–0.801 | 0.228–0.387 |
| suffix_conditioned | 512 | 0.340–0.672 | 0.537–0.868 | 0.235–0.399 |
| suffix_conditioned | 1024 | 0.340–0.672 | 0.540–0.872 | 0.235–0.399 |
| suffix_conditioned | 2048 | 0.340–0.672 | 0.540–0.872 | 0.235–0.399 |
| sparse_lexical | 256 | 0.321–0.634 | 0.499–0.804 | 0.229–0.390 |
| sparse_lexical | 512 | 0.373–0.754 | 0.548–0.910 | 0.253–0.430 |
| sparse_lexical | 1024 | 0.373–0.755 | 0.552–0.917 | 0.253–0.430 |
| sparse_lexical | 2048 | 0.373–0.755 | 0.552–0.917 | 0.253–0.430 |
