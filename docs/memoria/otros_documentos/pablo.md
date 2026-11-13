# Críticas a la memoria del TFG

Las páginas son las numeradas de la memoria. Al final hay unos comentarios generales por capítulo.

---

## Resumo / Abstract

1. **El resumen es demasiado denso.** Se tiene que poder entender sin conocimientos técnicos. Ahora mismo aparecen McR², los tensores U y D, el «cruce de arquitecturas»… cosas que el lector no puede saber qué son. Estructúralo como **contexto → problema → solución → resultado**, sin jerga.

## Capítulo 1 — Introdución

### 1.1 Contexto xeral (pp. 1–4)

2. **Fig. 1.1.** Más que un «comportamiento», el modelo aprende la relación entre entradas y salidas (la función *f*). Además, la comparación tendría más sentido contra un sistema de reglas o simbólico (sistemas expertos), que es la otra gran rama de la IA, que contra la «informática tradicional». El texto ya menciona los sistemas expertos, pero la figura no.
3. **La explicación de cómo funciona un modelo (p. 2) es muy liosa** para alguien que no sepa de IA.
4. **Mezclas en una sola sección qué es la IA, qué es un modelo y el contexto histórico.** Usa subsecciones; ahora mismo la lectura es confusa.
5. **«Polo exposto pode deducirse correctamente que o desenvolvemento de sistemas de I.A. baseados en A.P. é máis sinxelo de deseñar […] e así é» (p. 3).** Es una afirmación muy fuerte sin ninguna fuente aparte de tu opinión. Además, justo después dices que el primer problema del A.P., el coste computacional, «resolveuse», cuando tú mismo lo sufriste en el TFG (nota al pie del cap. 4: una GPU 32 veces más barata y 3 h por época). Matízalo y cítalo.

### 1.2 Interpretabilidade e explicabilidade (pp. 4–8)

6. **1.2.3: «as explicacións locais […] optan por achar unha nova función f̂ que aproxime f na contorna de x».** Yo lo entiendo; tu tribunal probablemente no. Explícalo con palabras o con un ejemplo antes de la notación.
7. **1.2.3: «Posto que os métodos de explicación baséanse nos principios do modelo […] só será aplicable para os membros desta».** ¿Fuente? La división entre métodos agnósticos y específicos del modelo está en cualquier revisión de XAI (p. ej. Adadi & Berrada, 2018, o el libro de Molnar). Cítala.
8. **1.2.3 (p. 7): «exporanse agora dúas técnicas de explicación…».** Estás en la sección general de explicabilidad. Si vas a hablar de técnicas concretas (LIME, CAM), dales su propia subsección.
9. **Fig. 1.3.** Está referenciada en la p. 7, pero aparece en la p. 6, casi dos páginas antes. Cuando hablas del husky o de «¿se fijó en las orejas?» tengo que ir a buscar la imagen. Colócala junto al texto que la comenta. Una figura debe apoyar lo que estás contando en ese momento.

### 1.3 Bloques da Aprendizaxe Profunda (pp. 8–11)

10. **Fig. 1.4 y 1.3.1.** Léela como alguien que nunca ha visto un MLP: ¿qué son las entradas? ¿Para qué sirve la combinación afín? ¿Y la función no lineal? Además, la figura aparece antes del texto que la explica, y ese texto no llega a explicar realmente qué es un MLP (la figura es una sola neurona, no un MLP).
11. **1.3.2: la convolución es crucial para tu trabajo** (es la base de las CNN con las que te comparas y del *patch embedding*) y no hay ni una fórmula ni una imagen. Está mejor explicado el MLP que la convolución.
12. **1.3.3: «tensores UQ, UK e UV […] chamados tensores "consulta", "chave" e "valor"».** ¿Qué es un tensor? ¿Qué diferencia a uno de consulta de uno de clave o de valor? Explica la intuición (qué busca cada token / qué ofrece / qué devuelve). Y aquí son matrices: «tensor» complica sin necesidad.
13. **1.3.3: «Recortan a entrada en porcións. Sílabas ou palabras no caso […] da linguaxe».** Entiendo lo que quieres decir, pero tu TFG es de visión. Si vas a poner ejemplos, usa el vocabulario de tu campo (parches, píxeles). Aquí lo de sílabas y palabras queda raro.
14. **1.3.3: «Converten esas porcións en vectores do mesmo tamaño, chamados tokens».** Distingue **token** de **embedding**. En lenguaje, un token es una unidad discreta (una subpalabra, de longitud variable) que se representa con un identificador; el embedding es el vector de tamaño fijo que se le asocia. En visión, los parches tienen todos el mismo tamaño y en la literatura de ViT/CRATE se llama «token» al vector del parche ya proyectado, así que en tu caso sí son del mismo tamaño. Pero hay que decirlo, y la generalización a lenguaje tal como está es incorrecta. Por cierto, el glosario define token como «vector que se aprende», lo que tampoco cuadra con esto.
15. **Ecuaciones.** Si las referencias y comentas en el texto, deben estar numeradas. Ahora unas lo están y otras no, y la numeración se repite (hay una ec. (1) en el cap. 1 y otra en el cap. 3).
16. **Fig. 1.5.** Es la primera vez que salen «embeddings», «Encoder», «bloques residuales» y «normalización», y ninguno está explicado: ¿qué son? ¿Por qué se normaliza?
17. **1.3.4: «unha instanciación do xa coñecido Transformer».** No puedes decir «el ya conocido Transformer» y seguir. ¿Qué es un Transformer? ¿En qué se diferencia de una RNA cualquiera? No se ve en ninguna figura. Como con la convolución, explicas y muestras más el MLP que la arquitectura troncal de tu TFG. Además el nombre está mal: es *Vision Transformer*, no «Vision in Transformer».
18. **1.3.4: «o Transformer demostrou moi bo rendemento».** ¿Fuente? ¿Buen rendimiento respecto a qué? ¿Desde el principio o ahora? Y lo de la «burbuja» lo protagonizan los LLM (GPT, Gemini, Llama…, todos basados en Transformer), no el Transformer original de 2017.

### 1.4 Representacións (pp. 11–15)

19. **1.4.1: «Es máis segundo o traballo fundacional de Barlow do 72 […] [12]».** La forma de citar es muy coloquial. Mejor: *«Ademais, tal e como propuxo Barlow no seu traballo fundacional no campo da neurociencia [12], …»*.
20. **1.4.1: «O Dropout [15] […] introduce dispersión arbitraria no adestramento […] ca premisa de que acelerará devandito proceso».** Dropout no está definido y la descripción es incorrecta. No acelera el entrenamiento (suele ralentizar la convergencia): es una técnica de **regularización contra el sobreajuste** que desactiva aleatoriamente neuronas (no conexiones) con una probabilidad dada. Además, en tus modelos no se usa (Táboa B.1: «sen Dropout»), así que plantéate quitarlo.
21. **1.4.3: «pasamos da representación orixinal […] a unha máis interesante ou representativa».** ¿Más interesante o representativa respecto a qué? Además, al usar «palabras» para referirte a elementos del diccionario mientras hablas de píxeles, parece que hablas literalmente de palabras. El término estándar es **átomos** del diccionario.
22. **Fig. 1.7: «algoritmo KSV».** ¿Por qué tengo que adivinar qué algoritmo es y qué hace? Es **K-SVD** (y en la bibliografía [13] sale «rmk-svd»). Lo mismo con «bases de Haar»: ¿qué es una base de Haar?
23. **1.4.4: «onde ε ∼ N(µ, σ²) nos albores do século XIX, chegando á coñecida forma cerrada».** Vuelves a suponer que el lector sabe lo mismo que tú. No tengo por qué saber de qué fórmula hablas (mínimos cuadrados de Gauss). Explícalo en una frase. Por cierto, el ruido debería tener media 0, no µ.
24. **1.4.4: «Andrey Tikhonov introduce a penalización segundo a norma L2».** ¿Dónde está la cita? A veces citas a personas y afirmaciones y otras veces no. Lo mismo en el pie de la Fig. 1.8 («L2 correspondente ca penalización Tikhonov»).
25. **Normas L1 y L2.** No explicas qué son; las usas directamente. Una frase intuitiva bastaría (L1 = suma de valores absolutos, empuja a que haya ceros; L2 = suma de cuadrados, encoge todo por igual). Piensa en cómo las explica Dani.
26. **1.4.4: «o algoritmo Ista atopa unha solución ó Lasso a través da iteración de descenso do gradiente…».** ¿Qué es el descenso del gradiente? ¿Tiene que ver con la optimización, con el entrenamiento? Lo explicas en el Apéndice A, pero el texto no lo menciona: referencia el apéndice la primera vez que aparezca.

### 1.5 – 1.7 Aprendizaxe desenrolada, MCR² y CRATE (pp. 15–21)

27. **1.5.1: «Nos modelos de A.P. vistos (M.L.P., C.N.N. e Autoencoders)».** Referencia las secciones (1.3.1, 1.3.2, 1.4.2), no solo el acrónimo.
28. **1.6: el primer párrafo (la cita de [22]) está mal formateado.**
29. **1.6: MCR² aparece como acrónimo antes de definirse** (se define en 1.6.1).
30. **1.6.1: «A MCR² Dispersa é unha función obxectivo».** ¿Qué es una función objetivo? No se ha explicado en ningún sitio antes.
31. **Fig. 1.10.** No se entiende lo que quieres decir porque lo complicas demasiado. «Asigna cada token a un espazo… αi,j é a influencia do espazo Uj no token zi»: léelo como alguien que no sabe de visión ni de IA. ¿Qué es un espacio? ¿Y una influencia? ¿Qué es U[K]? ¿Qué significa la ecuación? El pie de una figura es un complemento, y la figura debería entenderse sola. Si quieres explicar algo, hazlo en el texto, no en el pie.
32. **1.6.1: las tres ecuaciones de ΔR, R y Rc.** No veo qué aportan al texto así puestas. Si lo que quieres decir es «MCR² maximiza una diferencia: expandir todo el conjunto y comprimir cada grupo», dilo con esas palabras y deja las fórmulas como apoyo, explicando qué papel tiene cada término. Lo de «(aquí β é unha constante cuxa presencia é necesaria […] pero unha xustificación […] está fora de lugar)» sobra.
33. **1.7.1: «E dicir, segundo os autores, se os datos siguen unha estructura […] todo será doado».** ¿Qué autores? Nómbralos. Ojo: el primer autor de CRATE es Yaodong Yu (Yi Ma es el investigador principal), así que lo correcto es «Yu et al. [23]» o «o grupo de Yi Ma [23]».
34. **1.7.2: «mentras que no ViT a Atención dunha cabeza […] é: …».** Lo complicas de más. Explica con palabras cuál es la diferencia (en CRATE las tres matrices Q, K y V son la misma U) y deja la ecuación como apoyo. No debería tener que descifrarla para entenderlo.

### Comentario general del capítulo 1

35. **Mezclas IA básica, IA «avanzada» y conceptos de biología** (aplicables a la IA, pero irrelevantes para tu trabajo). Leyendo la introducción no me queda claro de qué va tu trabajo: mezclas marco teórico y algo de estado del arte con lo que debería ser la introducción.
36. **Llevo 20 páginas y no sé qué intentas con tu trabajo:** qué problema resuelves, qué te diferencia, si es complejo o no. No hay hilo conductor. Me empiezas a decir por qué debería interesarme tu TFG en la sección 1.8, después de 20 páginas complejas. Debería ser al revés. La introducción tiene que dejar claro qué problema resuelves, por qué es importante y qué te diferencia de otros trabajos. Después, en el marco teórico y el estado del arte, explicas cómo funcionan la IA, los modelos, etc., y lo complicas todo lo que quieras: el lector puede decidir si quiere ver las matemáticas. Pero no debería tardar 5–10 minutos en entender cada párrafo, y menos en la introducción.

## Capítulo 2 — Metodoloxía e planificación

37. **Cita inicial «O que queira peixes que molle o cu»: quítala.** Además, la referencia [32] apunta a otro refrán («O que algo quere, algo lle custa»).
38. **Fig. 2.1:** bastante redundante; es el flujo que seguiría cualquiera.
39. **2.2.2: «Merécese unha mención o mapa de cor pink […] o seu parecido co rosa institucional da UDC».** Completamente irrelevante. Lo mismo con las opiniones sobre Bash, Vim o tmux. Lo que sí falta son las versiones de Python y PyTorch.
40. **2.2.3: tus datos son de un campo muy concreto y de nicho.** De los ojos solo dices que tienen enfermedades raras y que hay una etiqueta de sano. Luego enumeras enfermedades y patologías, pero no sé qué me estás diciendo: nunca explicas, para alguien de fuera, qué hace que un ojo esté sano o no. Por ejemplo, en el pie de la Fig. 2.3(a), «Imaxe de fondo de retina con edema de disco óptico»: no tengo ni idea de qué es un edema de disco óptico, cómo se ve, si se aprecia a simple vista o si es grave. Podrías decir que es un ojo sano y para mí sería lo mismo. Señálalo en la imagen o cuéntalo en una frase.
41. **2.4: «sendo 2 titores, a conta resulta en 190 × 15 + 50 × 30 × 2 = 5850 euros».** ¿Tienes 2 tutores o 3? En la portada aparecen 2, en 2.2.1 dices «tres tutores» (uno cambió de universidad, según la nota al pie) y en los costes cuentas 2. Unifícalo y explica qué pasa con las horas del tercero.

## Capítulo 3 — Traballo proposto

### 3.2 Segundo orde (pp. 31–35)

42. **Referencias rotas.** En 3.2.1 aparece «(ver Sección [?])» y en 3.8 «na Sección ⁇».
43. **3.2.1: «esta operación intuitivamente corresponde con movelos […] cara un espazo onde os tokens máis parecidos segundo un criterio k o sexan máis…».** Esto es intuitivo para ti, no para el lector.
44. **3.2.1: «Devandita operación conlevaría K inversas o cal é computacionalmente complexo».** Es cierto, pero justifícalo: cita a [23] e indica el coste (K inversiones de matrices en cada capa y en cada paso de entrenamiento).
45. **3.2.1: «Neste traballo preguntámonos: Qué pasaría se a aproximación fora máis precisa?».** ¿Más precisa respecto a qué? No has dicho que la aproximación anterior sea imprecisa ni que haga falta más precisión. Tampoco sé si CRATE es mejor o peor que otros modelos del estado del arte, ni si tu aproximación corrige alguno de sus fallos.
46. **3.2.1: «Para consultar detalles técnicos o lector pode consultar [23]».** Básicamente le dices al lector que no lea tu TFG y se vaya a otro artículo. Tu trabajo es coger ese artículo y explicarlo de forma que lo entienda cualquiera. Si tu trabajo mejora un artículo, explica qué hace el artículo y luego tu mejora.
47. **3.2.1–3.2.2: no tengo por qué descifrar las ecuaciones** para entender lo que dices. Explica la diferencia con palabras, una tabla o una figura. No debería tener que pararme para poder seguir leyendo.
48. **Fig. 3.1.** Una figura es una explicación visual de lo que dices en el texto. Si quiero ver ecuaciones, me voy al anexo. Ahora hay matrices dentro de los nodos que tapan el diagrama. Además, en (b) las conexiones se cruzan y no queda claro qué entra en cada bloque: revísalas.

### Comentario general del capítulo 3

49. **En este capítulo las ecuaciones están numeradas y en los anteriores no.** Pero sobre todo, el capítulo es muy difícil de leer: no es un TFG de matemáticas ni un artículo científico, es un TFG de visión por computador. La p. 34, por ejemplo, es indescifrable. Yo la entiendo si me pongo, porque soy graduado en IA, sé de visión y me explicaste tu TFG. Pero piensa en quién lo va a leer: el tribunal sabe qué es un MLP, pero no tiene por qué saber qué es CRATE ni MCR², y probablemente mire las figuras 10 minutos antes de la defensa. Llegará a esta página y pensará «no sé qué está haciendo, ni qué problema resuelve, ni qué aporta». Es un fallo **muy grave**.
50. Además, hay varias páginas como la 34 (31–35 son casi solo ecuaciones) que el tribunal se saltará directamente. Y en general mucho del TFG es así; si le sumas que hay muy pocas figuras, el resultado es que **casi no van a leer tu TFG**.
51. **3.3.1: «Se ben todolos Transformadores […] teñen relacións de atención, e pode dicirse que dous tokens atendense moito ou pouco».** Empiezas la motivación de una sección con una afirmación sin fuente y que no es evidente.
52. **La estructura del capítulo es muy liosa.** ¿Tu TFG va de CRATE, de análisis de imágenes de ojos o de explicabilidad? Pones al mismo nivel la visualización de mapas de atención, CRATE, las métricas… Estructura los capítulos como un libro: *esto es CRATE → funciona así → nos permite visualizar así → medimos su rendimiento así → …*. Ahora parece una lista de la compra. Por ejemplo, ¿por qué existe la sección 3.3? La motivas al final de todo, pero debería estar ya planteada antes, de forma que 3.3 sea la solución al problema que has presentado.
53. **3.3.4: «Dado que neste traballo plantexamos unha nova arquitectura, significativamente distinta ó CRate orixinal (ver Sección 3.2.2)».** Aquí deberías recordar qué hace diferente a tu arquitectura, por qué es mejor y qué problemas resuelve, no solo decir que es diferente. Y sobre todo, pon una figura que lo muestre.
54. **3.3.4: «Aproveitando que na arquitectura segundo orde temos dous bloques de cabezas ligados estreitamente […] pensamos que o máis axeitado sería ter dous mapas de atención por cada cabeza».** Llevo dos horas y media leyendo y no sé de qué me hablas ni a qué te refieres con «segundo orden». No puedes pretender que, después de 40 páginas de lectura pesada, recuerde de qué arquitectura, bloques o mapas de atención hablas. Referencia la sección, pon una figura o vuelve a explicarlo brevemente.
55. **En resumen:** de este capítulo no he entendido casi nada y me he saltado muchas partes. En 20 páginas hay 4 figuras, que no se entienden por sí solas y aportan poco al texto. Desde la sección de segundo orden hasta la de métricas he estado completamente perdido.

## Capítulo 4 — Resultados

56. **Fig. 4.1:** la leyenda es demasiado pequeña y no se entiende (y son nombres de configuración, «orden:second, Dcompartido:1…», no nombres legibles).
57. **4.2.3: «presente no modelo CRate na cuarta cabeza (H3) e na variante de segundo orde na sexta cabeza (H5)…».** No me puedes hacer volver tres páginas atrás para ver de qué hablas, y encima buscar en la matriz de atención cuál es H3 y cuál H5. Marca las cabezas en la figura o pon la figura junto al texto.
58. **Fig. 4.6:** cero legibilidad visual (8 curvas superpuestas).
59. **Pies de figura:** sirven para explicar qué muestra la figura, no lo que hay detrás. Eso va en el texto. En general son demasiado largos y explicativos.
60. **El capítulo de resultados debería poder leerse en 5 minutos y saber si tu modelo es bueno o no.** Ahora mezclas explicaciones enormes con imágenes también enormes, con texto que no se lee ni haciendo zoom. Lo que busca el lector es ir a esa sección y ver algo como «exactitud comparable con un 20% menos de parámetros», «converge 1000 épocas antes», etc., y entender las imágenes. No tener que leer 20 páginas para saber qué resultados obtienes. Estás mostrando resultados, no explicando cómo funciona tu modelo. Empieza el capítulo con una **tabla resumen**: variante × conjunto → métrica final. Ojo: la explicabilidad no la mides cuantitativamente, así que no puedes dar un «+X% de explicabilidad»; o añades una métrica o lo presentas como resultado cualitativo.
61. **Separa expectativas de resultados.** En 4.3.2, el primer párrafo (qué esperarías ver en las cabezas) es motivación y va en el cap. 3, junto a la explicación de los mapas de atención. En Resultados deja solo lo que observaste.

## Capítulo 5 — Conclusións

62. **Las conclusiones están mal planteadas.** No pueden ser 8 páginas de texto: deben ocupar 1–2 páginas como máximo y dejar claro qué resolviste, si conseguiste tu objetivo, qué problemas encontraste, qué te llevas del trabajo y qué harías en el futuro.
63. **No deberías tener una sección de resultados dentro de las conclusiones, y menos tan larga.** Si hablas de resultados, que sea muy breve y no técnico, para apoyar la importancia del TFG. Por ejemplo: *«Compartir U e D mantén un rendemento comparable cunha rede un 20% máis pequena e produce mapas de atención máis variados na retina»*.
64. **Limitaciones:** no te quedes en «faltó tiempo». Pon también la complejidad del problema, los recursos (una GPU frente a las 16 A100 de los autores), los datos (test de DRIVE de una sola imagen, desbalanceo en RFMiD) y las limitaciones metodológicas: una sola ejecución por variante y sin semilla, evaluación de la explicabilidad solo cualitativa y sin validación clínica. Eso no te deja peor: te hace parecer riguroso.
65. **Trabajo futuro:** de longitud está bien (~1 página), pero está escrito como una lista de «entrenar más». Ordénalo por prioridad y enmárcalo como líneas de investigación: qué mejoras harías, qué problemas resolverías, qué arquitecturas complementarias probarías (p. ej. CRATE-α), qué evaluación cuantitativa añadirías y qué callejones querrías explorar.
66. **5.5: «Se os modelos presentados neste traballo non foron os máis útiles…» → fuera.** Le estás diciendo al tribunal, básicamente, que tu trabajo no sirvió. Lo mismo con «só sería deixar un programa executándose no servidor» en 5.3.
