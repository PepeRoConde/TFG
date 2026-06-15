**Teño que estar pendente constantemente por se hai modificacións do modelo oficial neste repositorio?**

Non. Para dar tranquilidade ao estudantado con respecto a posibles modificacións neste modelo, cun *mínimo de dous meses de antelación* á data límite de depósito de TFGs de cada convocatoria, "conxelarase" e etiquetarase a versión que será aplicable para esa convocatoria (https://gitlab.com/lauramcastro/modelo-tfg-fic/-/tags). A partir dese momento, só haberá modificacións no repositorio de se notificaren erros graves (https://gitlab.com/lauramcastro/modelo-tfg-fic/-/issues).

**Que ficheiros debo editar e cales non?**

Debes editar os ficheiros:

* `memoria_tfg.tex` para configurar a túa titulación, o teu nome, o título do teu TFG, o idioma da memoria, etc. no lugar indicado, así como para engadir paquetes específicos que precises (hai algúns exemplos comentados), e incorporar capítulos e apéndices da memoria cos correspondentes comandos `\include{contido/...}`.
* `bibliografia/bibliografia.bib`, para engadir referencias bibliográficas.
* `bibliografia/acronimos.tex` ou `bibliografia/glosario.tex`, para engadir acrónimos ou termos, en caso de precisares estes glosarios na túa memoria; de non ser así, podes comentar as liñas dos `\include` correspondentes no `memoria_tfg.tex` para que non aparezan baleiros no PDF xerado (podes facer o mesmo co índice de figuras ou táboas de ficaren baleiros no teu caso).
* `contido/introducion.tex`, `contido/conclusions.tex` e todos os demais ficheiros que conteñan capítulos da túa memoria, que deberías situar neste mesmo directorio `contido`.
* `portada/palabras_chave.tex`, para indicar as palabras chave que identifican o teu traballo (as que usarías nunha procura para atopalo).
* `portada/resumo.tex`, para incluír un breve resumo do teu traballo.

Non debes editar ningún outro ficheiro dos proporcionados co modelo, a excepción do `portada/portada.tex`, pero *só no caso de precisares engadir máis dunha persoa no rol de dirección* do TFG.

**Como se contabiliza o límite de 80 páxinas? Inclúe a portada, índices, etc.?**

O límite de 80 páxinas non inclúe a portada, agradecementos, resumo, índices,... esa parte do documento está numerada en números romanos (*I, II, III, IV...*). Tamén quedan fóra do límite os anexos, os glosarios e a bibliografía que se inclúen ao final do documento. É dicir, o límite de 80 páxinas aplica ao corpo principal da memoria, e polo tanto contabilízase a partir de que comeza o contido da mesma, isto é, cando a numeración pasa a ser arábiga (*1, 2, 3...*), e para cando empezan os anexos. De se superar o límite de 80 páxinas antes de rematar o corpo principal e comezar os anexos, a numeración arábiga pasará a mostrarse en vermello, xunto cunha mensaxe de aviso, nos pés das páxinas en exceso.

**Podo redactar a memoria en castelán ou inglés? Como o fago?**

Si. De feito, se cursaches o GEI no grupo de inglés, _debes_ redactar a memoria en inglés. Para facer o cambio de idiomas só tes que descomentar a liña correspondente no inicio do ficheiro `memoria_tfg.tex`.

É importante saber que a portada da memoria sempre se xerará en galego, o idioma oficial da universidade, independentemente do idioma configurado e usado para o resto do documento. En ningún caso debes alterar o modelo para modificar este comportamento.

**O resumo e as palabras chave deben ir obrigatoriamente en dous idiomas?**

Si. Se o idioma principal da memoria é o castelán ou o galego, entón o resumo e as palabras chave irán nese idioma e en inglés. Se o idioma principal da memoria é o inglés, entón o resumo e as palabras chave irán nese idioma e en galego.

**Incluín no ficheiro `bibliografia/bibliografia.bib` algunhas referencias, por que non aparecen no PDF?**

Na sección de referencias do PDF aparecerán só as referencias do ficheiro `bibliografia/bibliografia.bib` que estean citadas (co comando `\cite{...}`) ao longo do documento. As referencias sen citar serán ignoradas.

**O índice de figuras/táboas/glosario/relación de acrónimos queda(n) baleiro(s) na miña memoria, podo eliminalos?**

Si, se calquera destes catro elementos non teñen contido no caso da memoria que estás a redactar, podes suprimilos (simplemente comentándoos no código fonte LaTeX) de xeito que non aparezan no PDF final.

**Non me aparecen os glosarios no PDF**

Se incluíches termos no ficheiro `bibliografia/acronimos.tex` e están referenciados no teu documento empregando polo menos unha das ordes `\acrfull`, `\acrlong`  ou `\acrshort`, pero aínda así non se xera a lista de acrónimos no PDF, ou ocorre algo semellante cos termos do glosario (é dicir, están incluídos no `bibliografia/glosario.tex` e referenciados no documento con polo menos unha orde `gls` ou `Gls` pero non se xera a lista de termos), debes revisar a estrutura do teu proxecto.

Todo o contido do teu proxecto en Overleaf debe estar no raíz, sen que haxa directorios intermedios, pois isto impide que algunhas ferramentas (nomeadamente, as que procesan e xeran os glosarios) atopen os ficheiros axeitadamente. Dito doutro xeito, os únicos subdirectorios que debería conter o teu proxecto en Overleaf son os que xa están no propio modelo (por exemplo, `anexos/`, `contido/`, `imaxes/`...). Se no teu caso estes directorios (e o resto de ficheiros que neste modelo están directamente na raíz, como o `memoria_tfg.tex` ou o `estilo_tfg.sty`) están dentro dalgún outro subdirectorio, debes eliminar ese nivel intermedio.

Non modifiques a estrutura de ficheiros e directorios do teu proxecto en Overleaf, respecta a estrutura que presenta este modelo. Isto xa ocorre automaticamente se clonas o proxecto en Overleaf. Se creaches un proxecto LaTeX desde cero e despois incorporas os ficheiros deste modelo de TFG (descargados, por exemplo, en formato ZIP), asegúrate de que non os inclúes nun subdirectorio, senón directamente no raíz do proxecto.
```
