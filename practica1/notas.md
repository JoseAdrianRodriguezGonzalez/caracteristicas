# Tarea importante

Calcular los histogramas de sumas y diferencias para 12 ventanas, iniciando en tamaño 3 y avanzando en hasta 25 calcule 7 características usando histogramas: 
 * media
 * varianza
 * correlación
 * contraste
 * homogeneidad 
 * clúster de sombra 
 * clúster de prominencia

Se proporcionan 10 imágenes y 10 máscaras de 4 categorías. Extraer 5 recortes de 60x 60 píxeles para cada imagen tomando como referencia el dato de posición de la máscara. Por lo que hace 200 muestras para la parte 1.
Se entrega el código que lee el dataset y extrae los recortes de las imágenes y calcula las características y los SDH


# # SDH

Calculo

> $ I_s(k,l)=I(k,l)+I(k+\delta_k,l+\delta_l) $

> $I_D(k,l)=I(k,l)-I(k+\delta_k,l+\delta_l)$
Siendo una imagen de $I(k,l)$ donde $K \times L$ y cada pixel I(k,l) donde $k \in [0,K) ,l \in [0,L)$

Los histogramas de sumas y diferencias se calcul;an de la siogi8ehnte forma
$h_s(i;\delta_k,\delta_l)=h_s(i)=#\{(k,l) \in D,I_s(k,l)=i\}$

$h_D(i;\delta_k,\delta_l)=h_D(i)=#\{(k,l) \in D,I_D(k,l)=j\}$

$N=#\{D\}=k\times L =\Sigma_i h_s(i)=\Sigma_jh_d(j)$

Los histogramas de sumas y diferencias normaliados
$P_s=h_s(i)/N$
$P_D=h_D(i)/N$
