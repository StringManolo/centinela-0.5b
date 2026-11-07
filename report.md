## Centinela: combinaciones de las mejores tecnicas

Baseline: 4 semillas en este experimento. Ruido entre semillas (desviacion): TYPE 0.027, TYPE_SHIFT 0.025, MRR 0.021, DET 0.011, SCORE 0.015.

SCORE = media de TYPE, TYPE_SHIFT y DET (acierto de tipo, robustez ante frases nuevas, distinguir vulnerable de seguro). Una comparacion entre dos ejecuciones sueltas necesita una diferencia mayor que la del ruido x raiz de 2.

### Todas las configuraciones (mejor tecnica individual: **ia3**, SCORE 0.736)

`*` / `**`: la diferencia con la mejor individual supera el ruido (sin corregir / con Bonferroni por 26 combinaciones).

| configuracion | TYPE | TYPE_SHIFT | MRR | DET | SCORE | dSCORE vs mejor individual | notas |
|---|---|---|---|---|---|---|---|
| baseline (media de 4) | 0.524 | 0.365 | 0.698 | 0.665 | 0.518 | | |
| targets_all+lr3e4+rank32 (combinacion) | 0.856 | 0.800 | 0.887 | 0.792 | 0.816 | +0.079* |  |
| rslora+rank32 (combinacion) | 0.822 | 0.789 | 0.899 | 0.806 | 0.806 | +0.069* |  |
| targets_all+lr3e4+rslora (combinacion) | 0.794 | 0.789 | 0.861 | 0.803 | 0.796 | +0.059 |  |
| lora_plus+targets_all (combinacion) | 0.783 | 0.761 | 0.866 | 0.759 | 0.768 | +0.031 |  |
| targets_all+lr3e4 (combinacion) | 0.811 | 0.722 | 0.873 | 0.768 | 0.767 | +0.031 |  |
| targets_all+rank32 (combinacion) | 0.761 | 0.722 | 0.839 | 0.795 | 0.760 | +0.023 |  |
| targets_all+rslora (combinacion) | 0.806 | 0.678 | 0.869 | 0.749 | 0.744 | +0.008 |  |
| ia3 (individual) | 0.828 | 0.683 | 0.900 | 0.698 | 0.736 | (referencia) |  |
| lora_plus (individual) | 0.700 | 0.656 | 0.831 | 0.783 | 0.713 | -0.023 |  |
| lr3e4+rank32 (combinacion) | 0.767 | 0.589 | 0.855 | 0.736 | 0.697 | -0.039 |  |
| lora_plus+rank32 (combinacion) | 0.678 | 0.606 | 0.788 | 0.787 | 0.690 | -0.046 |  |
| targets_all (individual) | 0.633 | 0.617 | 0.795 | 0.753 | 0.668 | -0.069* |  |
| lr3e4+rslora (combinacion) | 0.644 | 0.667 | 0.778 | 0.690 | 0.667 | -0.069* |  |
| targets_all+rslora+rank32 (combinacion) | 0.644 | 0.628 | 0.756 | 0.728 | 0.667 | -0.070* |  |
| lora_plus+rslora (combinacion) | 0.689 | 0.578 | 0.794 | 0.727 | 0.664 | -0.072* |  |
| lr3e4+rslora+rank32 (combinacion) | 0.572 | 0.567 | 0.679 | 0.699 | 0.613 | -0.124* |  |
| lr3e4 (individual) | 0.628 | 0.517 | 0.775 | 0.684 | 0.609 | -0.127* |  |
| rslora (individual) | 0.650 | 0.456 | 0.785 | 0.683 | 0.596 | -0.140* |  |
| rank32 (individual) | 0.639 | 0.400 | 0.776 | 0.710 | 0.583 | -0.154* |  |
| lora_plus+targets_all+rslora (combinacion) | 0.500 | 0.539 | 0.611 | 0.660 | 0.566 | -0.170* |  |
| lora_plus+lr3e4+rslora (combinacion) | 0.394 | 0.367 | 0.595 | 0.653 | 0.471 | -0.265** |  |
| lora_plus+lr3e4 (combinacion) | 0.411 | 0.339 | 0.611 | 0.658 | 0.469 | -0.267** |  |
| lora_plus+targets_all+rank32 (combinacion) | 0.394 | 0.367 | 0.553 | 0.636 | 0.466 | -0.271** |  |
| targets_all+lr3e4+rslora+rank32 (combinacion) | 0.283 | 0.311 | 0.468 | 0.639 | 0.411 | -0.325** |  |
| lora_plus+lr3e4+rank32 (combinacion) | 0.272 | 0.206 | 0.473 | 0.664 | 0.381 | -0.356** |  |
| lora_plus+targets_all+lr3e4 (combinacion) | 0.194 | 0.211 | 0.457 | 0.679 | 0.361 | -0.375** |  |
| lora_plus+rslora+rank32 (combinacion) | 0.178 | 0.133 | 0.351 | 0.575 | 0.295 | -0.441** |  |
| lora_plus+lr3e4+rslora+rank32 (combinacion) | 0.156 | 0.094 | 0.335 | 0.584 | 0.278 | -0.458** |  |
| lora_plus+targets_all+lr3e4+rslora (combinacion) | 0.100 | 0.100 | 0.291 | 0.608 | 0.269 | -0.467** |  |
| lora_plus+targets_all+lr3e4+rslora+rank32 (combinacion) | 0.100 | 0.100 | 0.293 | 0.588 | 0.263 | -0.474** |  |
| lora_plus+targets_all+rslora+rank32 (combinacion) | 0.100 | 0.100 | 0.291 | 0.576 | 0.259 | -0.478** |  |
| lora_plus+targets_all+lr3e4+rank32 (combinacion) | 0.100 | 0.100 | 0.293 | 0.342 | 0.181 | -0.556** |  |

### ¿Alguna combinacion supera a la mejor tecnica individual?

| combinacion | dTYPE | dTYPE_SHIFT | dDET | dSCORE |
|---|---|---|---|---|
| targets_all+lr3e4+rank32 | +0.028 | +0.117* | +0.094** | +0.079* |
| rslora+rank32 | -0.006 | +0.106* | +0.108** | +0.069* |

La mejor de 26 combinaciones esta inflada por seleccion: confirma la ganadora con semillas nuevas.

### Analisis factorial: efecto de cada tecnica y sus interacciones

**Efecto principal** = cuanto cambia la medida, en promedio sobre todas las combinaciones de las demas tecnicas, al activar esa tecnica. `*` = supera el margen de error (Lenth).

| tecnica | efecto en TYPE | efecto en TYPE_SHIFT | efecto en MRR | efecto en DET | efecto en SCORE |
|---|---|---|---|---|---|
| lora_plus | -0.317* | -0.273* | -0.259* | -0.088* | -0.226* |
| targets_all | -0.053 | +0.013 | -0.063 | -0.014 | -0.018 |
| lr3e4 | -0.151 | -0.120 | -0.123* | -0.049 | -0.107 |
| rslora | -0.107 | -0.068 | -0.106 | -0.027 | -0.067 |
| rank32 | -0.121 | -0.116 | -0.121* | -0.041 | -0.093 |
| margen de error | 0.165 | 0.149 | 0.114 | 0.062 | 0.110 |

**Interacciones activas** (combinaciones de tecnicas cuyo efecto conjunto no es la suma de los individuales):

| interaccion | efecto en SCORE | efecto en TYPE | lectura |
|---|---|---|---|
| lora_plus x lr3e4 | -0.112* | -0.136 | se solapan: juntas ayudan menos que la suma de lo que ayudan por separado |

**Mejores configuraciones** segun el SCORE observado y segun el modelo que solo conserva los efectos activos (suaviza el ruido de cada celda):

| puesto | observado | SCORE | predicho por el modelo | SCORE predicho |
|---|---|---|---|---|
| 1 | targets_all+lr3e4+rank32 | 0.816 | lr3e4 | 0.725 |
| 2 | rslora+rank32 | 0.806 | targets_all+lr3e4 | 0.725 |
| 3 | targets_all+lr3e4+rslora | 0.796 | lr3e4+rslora | 0.725 |
| 4 | lora_plus+targets_all | 0.768 | targets_all+lr3e4+rslora | 0.725 |
| 5 | targets_all+lr3e4 | 0.767 | lr3e4+rank32 | 0.725 |

### Como leer esto

- Cada celda del diseno es una sola ejecucion; los efectos factoriales usan las 2^k ejecuciones, por eso son mas fiables que cada celda.
- La mejor configuracion observada esta inflada por seleccion; verificala con semillas nuevas antes de darla por buena.
- Una combinacion que `divergio` combina ritmos de aprendizaje demasiado altos: no es un fallo del experimento sino un resultado.
- IA3 y las demas tecnicas de referencia no se combinan (cambian el tipo de adaptador); se miden solo por separado.
