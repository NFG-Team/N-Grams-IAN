import streamlit as st
from ngram import NGramLM, prepare_data
import altair as alt
import pandas as pd


# Define the app
def main():
    st.title("Playground Trabajo N-Gramas")
    st.markdown(
        """
        Integrantes: 
            <a href="https://github.com/nicocanta20" style="color: #FF6347;" target="_blank">Nicolás Cantarovici</a>
            <a href="https://github.com/florianreyes" style="color: #FF6347;" target="_blank">Florian Reyes</a>
            <a href="https://github.com/gonzaslucki" style="color: #FF6347;" target="_blank">Gonzalo Slucki</a>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for file selection
    st.sidebar.header("Configuracion")
    text_file = st.sidebar.selectbox(
        "Seleccionar el archivo de texto:",
        ("frankenstein.txt", "alice_in_wonderland.txt", "don_quixote.txt"),
    )

    # Load and train the model using the selected file
    vocab, data = prepare_data(f"./{text_file}")

    # Recommendations for phrases based on text file
    phrase_recommendations = {
        "frankenstein.txt": [
            "the night",
            "i saw",
            "my thought",
            "the monster",
            "the creature",
        ],
        "alice_in_wonderland.txt": [
            "alice could",
            "the caterpillar",
            "the queen",
            "the baby",
            "the hatter",
        ],
        "don_quixote.txt": [
            "sancho panza",
            "the knight",
            "his lance",
            "in the village",
            "the squire",
        ],
    }

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["Informe", "Predecir Siguiente Palabra", "Visualizar Top Predicciones"]
    )

    with tab1:
        st.markdown(
            """
            # Trabajo Práctico: Modelo de N-gramas

            > **Nota**: Este informe detalla el análisis y evaluación realizados originalmente con el libro "Frankenstein". 
            > Para la demostración interactiva de esta aplicación, agregamos dos mas opciones: 
            > "Alice in Wonderland" y "Don Quixote"

            ## Índice
            1. [Preparación del Corpus](#1-preparación-del-corpus)
                - Obtención y limpieza
                - Procesamiento de datos
                - División del dataset

            2. [Implementación del Modelo](#2-implementación-del-modelo-de-n-gramas)
                - Clase NGramLM
                - Métodos principales
                - Sistema de evaluación

            3. [Análisis de Resultados](#3-análisis-de-resultados)
                - 3.1 Calidad de textos generados
                - 3.2 Similitud con textos originales
                - 3.3 Creatividad e inteligencia

            ## 1. Preparación del Corpus

            Para este trabajo práctico, utilizamos como corpus el libro "Frankenstein" de Mary Shelley descargado en Proyecto Gutenberg. El proceso de preparación de datos incluyó los siguientes pasos:

            1. Implementación de la función prepare_data() que:
            - Lee el archivo de texto
            - Realiza limpieza del texto:
                - Convierte todo a minúsculas
                - Elimina signos de puntuación
                - Elimina espacios extras
            - Tokeniza el texto en oraciones
            - Aplica lematización usando WordNetLemmatizer de NLTK
            - Agrega el token </s> al final de cada oración
            - Crea el vocabulario único

            2. División del dataset:
            - 70% para entrenamiento
            - 15% para validación 
            - 15% para testing

            ## 2. Implementación del Modelo de N-gramas

            ### Clase NGramLM

            Se implementó una clase NGramLM que maneja el modelo de n-gramas con las siguientes características:

            #### Inicialización
            - Parámetros para especificar N (tamaño del n-grama)
            - Opción de smoothing con parámetro k
            - Estructuras de datos para almacenar:
                - Vocabulario
                - Probabilidades
                - Frecuencias

            #### Métodos Principales

            1. train()
                - Entrena el modelo contando frecuencias de n-gramas
                - Maneja casos especiales para unigramas
                - Calcula probabilidades condicionales

            2. get_ngram_prob() y get_ngram_logprob()
                - Calculan probabilidades de n-gramas
                - Manejan casos de n-gramas no vistos
                - Implementan smoothing cuando está configurado

            3. perplexity()
                - Calcula la perplexity del modelo sobre un conjunto de datos
                - Sirve como métrica de evaluación

            4. generate_next() y generate_next_n()
                - Generan texto prediciendo las siguientes palabras más probables
                - Utilizan el contexto previo según el tamaño del n-grama

            ### Evaluación del Modelo

            Se implementó un sistema de evaluación que:
            - Entrena modelos con diferentes valores de N (2,3,4,5)
            - Calcula perplexity en conjuntos de:
                - Entrenamiento
                - Validación
                - Testing
            - Compara resultados entre diferentes tamaños de n-gramas

            ## 3. Análisis de Resultados

            ### 3.1 Calidad de los Textos Generados

            Al analizar los textos generados con diferentes valores de n, observamos patrones interesantes:

            #### Perplexity
            Los resultados muestran una clara mejora en la perplexity a medida que aumenta n:
            - 2-gram: 3753.92 (validación)
            - 3-gram: 1651.85 (validación)
            - 4-gram: 773.43 (validación)
            - 5-gram: 388.00 (validación)

            Esta disminución significativa en la perplexity indica que los modelos con n más alto capturan mejor las dependencias del lenguaje.

            #### Tipos de Errores y Calidad

            1. *Bigramas (n=2)*:
                - Producen textos muy cortos o repetitivos
                - Ejemplo: "i saw the same time i was a i was a i was a i was a"
                - Error principal: loops de repetición
                - Carecen de coherencia narrativa

            2. *Trigramas (n=3)*:
                - Generan oraciones más largas y variadas
                - Ejemplo: "i saw the dull yellow eye of the most miserable of his child </s>"
                - Pueden mantener sentido por fragmentos cortos
                - Error principal: no mantener la coherencia a largo plazo

            3. *4-gramas (n=4)*:
                - Producen las oraciones más naturales y coherentes
                - Ejemplo: "the night passed away, and the sun rose from the ocean </s>"
                - Mantienen mejor el contexto y la estructura narrativa
                - Capturan mejor las construcciones típicas del texto original

            #### Metodología de Generación de Ejemplos

            Para evaluar sistemáticamente el comportamiento de los diferentes modelos, implementamos un proceso de generación de texto utilizando cuatro contextos iniciales diferentes: "the night", "my father", "i saw" y "my thought". Para cada contexto, entrenamos modelos de 2, 3 y 4-gramas (con smoothing_k=0) y generamos texto con un límite máximo de 15 palabras. Este enfoque nos permitió comparar directamente la calidad y características del texto generado bajo diferentes condiciones.

            #### Ejemplos Detallados por Contexto

            Para ilustrar mejor el comportamiento de los diferentes modelos, analizamos las generaciones a partir de estos contextos iniciales:

            1. *Contexto: "the night"*
            - 2-gram: "the night </s>"
                * Demasiado simple, termina inmediatamente
            - 3-gram: "the night on which the murder of poor william had been the favourite dream of my friend"
                * Genera una narrativa más extensa y coherente con el tono gótico del libro
            - 4-gram: "the night passed away, and the sun rose from the ocean </s>"
                * Produce una oración completa y poética, muy similar al estilo original

            2. *Contexto: "my father"*
            - 2-gram: "my father </s>"
                * Nuevamente, termina sin desarrollo
            - 3-gram: "my father had taken place in the same time that the fiend that lurked in my own"
                * Mezcla diferentes contextos narrativos, perdiendo coherencia hacia el final
            - 4-gram: "my father was not scientific, and i was left to struggle with a child's blindness, added to"
                * Genera una narrativa personal coherente, reflejando temas del libro

            3. *Contexto: "i saw"*
            - 2-gram: "i saw the same time i was a i was a i was a i was a"
                * Muestra el problema típico de loops en bigramas
            - 3-gram: "i saw the dull yellow eye of the most miserable of his child </s>"
                * Captura elementos descriptivos característicos del libro
            - 4-gram: "i saw him too </s>"
                * Aunque corta, es una frase natural y completa

            4. *Contexto: "my thought"*
            - 2-gram: "my thought of the same time i was a i was a i was a i was"
                * Cae en patrones repetitivos típicos de bigramas
            - 3-gram: "my thought </s>"
                * Sorprendentemente corta para un trigrama
            - 4-gram: "my thought and every feeling of wonder and admiration on this lovely girl, eagerly communicated her history"
                * Genera una narrativa compleja y coherente, manteniendo el estilo del texto original

            Estos ejemplos ilustran claramente cómo:
            
                - Los bigramas tienden a caer en loops repetitivos o terminar prematuramente
                - Los trigramas pueden generar contenido más extenso pero a veces pierden coherencia
                - Los 4-gramas producen el mejor balance entre coherencia y naturalidad, generando oraciones que podrían pasar por fragmentos del texto original

            ### 3.2 Similitud con Textos Originales

            La similitud con el texto original aumenta notablemente con n más altos. Para ilustrar esto, comparamos algunos fragmentos generados por el modelo de 4-gramas con sus contrapartes en el texto original:

            #### Comparación Directa de Fragmentos

            1. *Ejemplo 1:*
                - Original: "My thoughts and every feeling of my soul have been drunk up by the interest for my guest which this tale and his own elevated and gentle manners have created."
                - Generado: "my thought and every feeling of wonder and admiration on this lovely girl, eagerly communicated her history"
                - Análisis: Mantiene la estructura inicial y el tono emotivo, aunque diverge en el contenido específico.

            2. *Ejemplo 2:*
                - Original: "I saw him too"
                - Generado: "I saw him too"
                - Análisis: Reproducción exacta del texto original, demostrando cómo los 4-gramas pueden capturar frases completas.

            3. *Ejemplo 3:*
                - Original: "My father was not scientific, and I was left to struggle with a child's blindness, added to a student's thirst for knowledge"
                - Generado: "my father was not scientific, and i was left to struggle with a child's blindness, added to"
                - Análisis: Reproduce casi perfectamente el fragmento original hasta donde alcanza.

            4. *Ejemplo 4:*
                - Original: "the night passed away, and the sun rose from the ocean"
                - Generado: "the night passed away, and the sun rose from the ocean"
                - Análisis: Reproducción exacta, mostrando la capacidad del modelo para capturar frases descriptivas completas.

            #### Observaciones sobre el Preprocesamiento

            Es importante notar que nuestro proceso de lematización tuvo algunos efectos interesantes:
                - Transformó palabras plurales a singular, lo que a veces altera la naturalidad del texto (ej: "thoughts" → "thought")
                - Detectamos un comportamiento inesperado donde "was" se transformaba a "wa", lo cual nos pareció un error en el proceso de lematización
                - Estas transformaciones afectan la comparación directa con el texto original, pero mantienen la coherencia semántica

            ### 3.3 Creatividad e Inteligencia

            #### Creatividad:
            - *Niveles de Creatividad por N-grama*:
                - Bigramas (n=2): Muestran creatividad muy limitada, tendiendo a repeticiones y patrones simples
                - Trigramas (n=3): Exhiben mayor variabilidad creativa, generando combinaciones más interesantes
                - 4-gramas (n=4): Logran el mejor balance entre creatividad y coherencia

            - *Combinaciones Contextuales*: 
                - Ejemplo con trigrama: "the night on which the murder of poor william"
                - Combina elementos narrativos del texto original de formas nuevas pero coherentes
            
            - *Variaciones Estilísticas*:
                - Los 4-gramas generan variaciones poéticas como "the night passed away, and the sun rose from the ocean"
                - Mantienen el estilo gótico mientras crean nuevas expresiones

            - *Limitaciones Creativas*:
                - La creatividad está restringida al vocabulario y patrones del texto de entrenamiento
                - No puede generar conceptos verdaderamente nuevos
                - La coherencia narrativa se deteriora en secuencias más largas

            #### Inteligencia:
            - *Patrones estadísticos*: El modelo muestra "inteligencia" en forma de patrones aprendidos, no comprensión real
            - *Limitaciones*:
                - No entiende el significado semántico
                - No puede mantener coherencia en historias largas
                - No puede generar ideas originales
            - *Fortalezas*:
                - Captura bien patrones lingüísticos locales
                - Puede generar texto gramaticalmente correcto en fragmentos cortos
                - Mantiene el estilo del autor en n-gramas altos

            En conclusión, aunque el modelo demuestra una capacidad notable para generar secuencias cortas de palabras con sentido gramatical y semántico dentro de un contexto inmediato, especialmente al utilizar n-gramas más altos, carece de verdadera comprensión o creatividad. Su ‘inteligencia’ se limita a patrones estadísticos aprendidos del texto de entrenamiento, lo que lo hace efectivo para producir fragmentos breves, pero revela limitaciones significativas al intentar generar textos más extensos y coherentes.
                """
        )

    with tab2:
        st.header("Generación de Texto con N-gramas")

        n_gram_size = st.slider(  # Slider for N-gram size
            "Elegir el tamaño del N-grama:",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            key="n_gram_size_tab2",
        )

        # Dropdown for recommended phrases
        st.subheader("Proba las siguientes frases:")

        # Text input for user-provided context
        context_input = st.selectbox(
            "Elegir contexto:",
            phrase_recommendations[text_file],
            key="context_input_tab2",
        )
        # Number of words to predict
        n = st.number_input(
            "Cantidad de palabras a generar (N):",
            min_value=1,
            step=1,
            max_value=20,
            value=5,
        )

        if st.button("Generar Texto"):
            # Train the model based on selected N-gram size
            lm = NGramLM(N=n_gram_size)
            lm.train(vocab, data)
            if context_input:
                # Display the input context
                st.markdown(
                    f"Contexto Elegido:  <span style='color: #FF6347;'>{context_input}</span>",
                    unsafe_allow_html=True,
                )

                output = lm.generate_next_n(context_input, int(n))

                st.write("Texto Generado:")
                st.code("".join(output), language="text")
            else:
                st.warning("Por favor, selecciona un contexto para generar texto.")

    with tab3:

        def get_next_word_probabilities(model, context):
            context = (model.N - 1) * "<s> " + context
            context = context.split()
            ngram_context_list = context[-(model.N - 1) :]
            ngram_context = tuple(ngram_context_list)

            if ngram_context in model.prob:
                candidates = model.prob[ngram_context]
                most_probable_words = sorted(
                    candidates.items(), key=lambda kv: kv[1], reverse=True
                )

                df = pd.DataFrame(most_probable_words, columns=["word", "probability"])

                return df
            else:
                return None

        st.header("Visualización de Probabilidades de Palabras")

        # Model parameters
        n_gram = st.slider(
            "Elegir el tamaño del N-grama:", min_value=2, max_value=5, value=3
        )

        input_text = st.selectbox(
            "Elegir un contexto:",
            phrase_recommendations[text_file],
            key="input_text_tab3",
        )

        if st.button("Mostrar Probabilidades"):
            # Initialize model
            model = NGramLM(N=n_gram)
            model.train(vocab, data)
            df = get_next_word_probabilities(model, input_text)
            if df is not None:
                st.write(f"Contexto: '{input_text}'")
                st.write("Distribución de Probabilidades:")

                # Create Altair chart with Streamlit's red color
                chart = (
                    alt.Chart(df)
                    .mark_bar(color="#FF4B4B")  # Streamlit's default red color
                    .encode(
                        x=alt.X("word:N", sort="-y"),
                        y=alt.Y(
                            "probability:Q",
                            scale=alt.Scale(domain=[0, max(df.probability) * 1.1]),
                        ),
                    )
                    .properties(height=500, width=800)
                )

                st.altair_chart(chart, use_container_width=True)
            else:
                st.error(
                    "Contexto no encontrado en el modelo. Por favor, elige otro contexto."
                )


if __name__ == "__main__":
    main()
