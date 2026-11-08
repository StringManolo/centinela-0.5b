## Centinela: confirmación con semillas nuevas

Baseline: 5 semillas, media 0.510 ± 0.014 (sd entre semillas).

Contraste frente a **ia3** con t de Welch (df Satterthwaite). `*` p<0.05 sin corregir, `**` p<0.05 con Bonferroni por 3 configuraciones.

| configuración | n | SCORE | sd | dSCORE vs baseline | dSCORE vs ia3 |
|---|---|---|---|---|---|
| rslora+rank32 | 5 | 0.740 | 0.041 | +0.230 | +0.059* |
| targets_all+lr3e4+rslora | 5 | 0.732 | 0.070 | +0.222 | +0.051 |
| targets_all+lr3e4+rank32 | 5 | 0.716 | 0.074 | +0.206 | +0.035 |
| ia3 | 5 | 0.682 | 0.021 | +0.172 | — |

### Veredicto

**Confirmadas frente a ia3** (Bonferroni, 3 comparaciones): **rslora+rank32** (+0.059*)

No confirmadas: targets_all+lr3e4+rslora (+0.051), targets_all+lr3e4+rank32 (+0.035).

### Notas

- Semillas **nuevas** (no usadas en el factorial): esta ronda no está inflada por selección.
- El baseline de esta ronda fija el ruido de referencia; si su sd difiere mucho del factorial (0.015), usa la más conservadora para juzgar diferencias sueltas.
- El contraste es de medias con varianzas posiblemente distintas (Welch), no un z-test.
- Para que una combinación se considere **confirmada** debe superar a ia3 con `**`.
