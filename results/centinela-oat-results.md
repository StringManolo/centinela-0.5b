## Centinela: comparacion de tecnicas (una a una)


### Baseline (5 seeds): media +- desviacion entre seeds

| TYPE | TYPE_SHIFT | MRR | DET | EASY_TYPE | EASY_DET | FIT |
|---|---|---|---|---|---|---|
| 0.518 +- 0.031 | 0.323 +- 0.023 | 0.681 +- 0.027 | 0.682 +- 0.013 | 0.360 +- 0.055 | 0.426 +- 0.021 | 0.660 +- 0.055 |

azar: TYPE 0.10, MRR 0.29, DET 0.50. TYPE/MRR/DET = media de iid, ood_near y ood_hard (plantilla vista); TYPE_SHIFT = con plantillas no vistas; EASY = set sencillo; FIT = ajuste a sus propios datos de entrenamiento.

### Diferencia frente a la media del baseline

`*` supera el 95% sin corregir; `**` supera el umbral con correccion de Bonferroni por 18 tecnicas (por columna). En toda esta tabla (18 tecnicas x 7 columnas) se esperan ~6.3 `*` y ~0.4 `**` solo por azar: fiate de `**` y de las mejoras que aparecen en varias columnas a la vez.

| tecnica | n | pasos | min | dTYPE | dTYPE_SHIFT | dMRR | dDET | dEASY_TYPE | dEASY_DET | dFIT |
|---|---|---|---|---|---|---|---|---|---|---|
| lora_plus (LoRA+ (lr de B x16)) | 1 | 150 | 19 | +0.299** | +0.266** | +0.194* | +0.041* | +0.340* | +0.294** | +0.340* |
| targets_all (LoRA en todos los lineales) | 1 | 150 | 28 | +0.238** | +0.466** | +0.186* | +0.091* | +0.540** | +0.354** | +0.340* |
| ia3 (IA3 (lr 3e-3)) | 1 | 150 | 26 | +0.166* | +0.160* | +0.146* | +0.090* | +0.340* | +0.274** | +0.340* |
| lr3e4 (LR 3e-4) | 1 | 150 | 19 | +0.160* | +0.166** | +0.126* | +0.056* | +0.240* | +0.054 | +0.240* |
| rslora (rsLoRA (escala alpha/sqrt(r))) | 1 | 150 | 32 | +0.154* | +0.016 | +0.107* | +0.038 | +0.140 | +0.094* | +0.240* |
| rank32 (LoRA rango 32) | 1 | 150 | 33 | +0.088 | +0.127* | +0.077 | +0.037 | +0.140 | +0.094* | +0.240* |
| small_batch (batch efectivo 1 (mismo computo)) | 1 | 600 | 30 | +0.038 | -0.051 | +0.034 | +0.041* | +0.240* | +0.034 | +0.040 |
| label_ce_smooth (label_ce + label smoothing 0.1) | 1 | 150 | 28 | +0.010 | -0.007 | +0.030 | +0.016 | +0.140 | +0.034 | +0.140 |
| dora (DoRA) | 1 | 150 | 17 | +0.004 | -0.034 | +0.029 | +0.022 | +0.040 | -0.026 | +0.040 |
| wd01 (weight decay 0.1) | 1 | 150 | 27 | +0.004 | -0.018 | +0.025 | +0.017 | +0.040 | +0.014 | +0.040 |
| label_ce (loss CE restringida a las 11 etiquetas) | 1 | 150 | 26 | -0.012 | -0.007 | +0.028 | +0.007 | +0.140 | -0.006 | +0.140 |
| dropout02 (LoRA dropout 0.2) | 1 | 150 | 16 | -0.018 | -0.023 | +0.010 | +0.009 | +0.040 | -0.016 | +0.040 |
| templates5 (5 plantillas de entrenamiento) | 1 | 150 | 28 | -0.023 | +0.132* | -0.040 | -0.077* | -0.060 | +0.024 | -0.060 |
| upper_half (LoRA solo en la mitad superior) | 1 | 150 | 22 | -0.057 | -0.051 | -0.039 | -0.032 | -0.260* | -0.066* | -0.060 |
| cosine (schedule coseno) | 1 | 150 | 24 | -0.112* | -0.079* | -0.092* | -0.086* | -0.060 | -0.016 | -0.160 |
| neftune (NEFTune alpha 5) | 1 | 150 | 30 | -0.157* | +0.082* | -0.131* | -0.056* | -0.160 | -0.106* | -0.260* |
| loha (LoHa) | 1 | 150 | 24 | -0.240** | -0.190** | -0.228** | -0.174** | -0.160 | -0.016 | -0.560** |
| lokr (LoKr) | 1 | 150 | 31 | -0.273** | -0.157* | -0.243** | -0.156** | -0.160 | -0.016 | -0.560** |

### Sumar resultados (ensamblado de los scores guardados)

| ensamblado | runs | TYPE | TYPE_SHIFT | MRR | DET | EASY_TYPE | EASY_DET | FIT |
|---|---|---|---|---|---|---|---|---|
| baseline (todas las seeds) | 5 | 0.556 (+0.038) | 0.350 (+0.027) | 0.703 (+0.022) | 0.692 (+0.010) | 0.500 (+0.140) | 0.450 (+0.024) | 0.600 (-0.060) |
| TODOS los runs | 23 | 0.761 (+0.243) | 0.422 (+0.099) | 0.855 (+0.174) | 0.740 (+0.058) | 0.600 (+0.240) | 0.540 (+0.114) | 1.000 (+0.340) |

(entre parentesis: diferencia con la media del baseline. El ensamblado cuesta N pasadas en el movil.)

### TTA gratuito (promedio de la plantilla vista y la no vista, dentro de cada run)

efecto medio sobre TYPE: -0.045 (sd 0.052), sobre DET: -0.008 (sd 0.019), en 23 runs.

### Como leer esto

- El ruido sale solo de las seeds del baseline; cada tecnica tiene pocas seeds, asi que una ganancia aislada es una hipotesis. Repite las ganadoras con seeds nuevas antes de combinarlas.
- La mejor de 18 tecnicas esta inflada por seleccion (mismo test para elegir y medir).
- Un `(tope de tiempo)` significa que el entrenamiento se detuvo antes de los pasos previstos.
- Las ganancias no se suman: combina las ganadoras en un run nuevo y verificalo.
