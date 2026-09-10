"""
Chatbot basado en "intenciones" (intents): a cada intención se le asocian
varios patrones de ejemplo y varias respuestas posibles. El chatbot detecta
la intención más parecida al mensaje del usuario usando similitud coseno
sobre una representación TF-IDF.

Nota de diseño: se usa TF-IDF SIN eliminar palabras vacías (stopwords).
Como los patrones suelen ser frases muy cortas ("quién eres", "qué hora es"),
quitar palabras como "quién" o "es" puede dejar la frase vacía y perder toda
señal útil. TF-IDF ya se encarga de bajarle el peso a las palabras que se
repiten en muchos patrones distintos, sin necesidad de una lista fija.
"""
import random
import numpy as np
from .vectorizador import TFIDF


class Chatbot:
    def __init__(self, nombre="Asistente"):
        self.nombre = nombre
        self.intenciones = []  # [{"tag":str, "patrones":[str], "respuestas":[str]}, ...]
        self.vectorizador = TFIDF(quitar_vacias=False)
        self.matriz_patrones = None
        self.etiquetas_patrones = []
        self.entrenado = False

    def agregar_intencion(self, tag, patrones, respuestas):
        """
        tag: identificador de la intención (ej: 'saludo')
        patrones: frases de ejemplo que representan esa intención
        respuestas: posibles respuestas (se elige una al azar)
        """
        self.intenciones.append({"tag": tag, "patrones": patrones, "respuestas": respuestas})
        self.entrenado = False
        return self

    def entrenar(self):
        """Vectoriza todos los patrones de todas las intenciones para poder compararlos luego."""
        todos_los_patrones, self.etiquetas_patrones = [], []
        for intencion in self.intenciones:
            for patron in intencion["patrones"]:
                todos_los_patrones.append(patron)
                self.etiquetas_patrones.append(intencion["tag"])

        if not todos_los_patrones:
            raise RuntimeError("No hay intenciones registradas. Usa agregar_intencion() primero.")

        self.matriz_patrones = self.vectorizador.ajustar_transformar(todos_los_patrones)
        self.entrenado = True
        return self

    @staticmethod
    def _similitud_coseno(vector, matriz):
        norma_vector = np.linalg.norm(vector)
        normas_matriz = np.linalg.norm(matriz, axis=1)
        normas_matriz = np.where(normas_matriz == 0, 1e-9, normas_matriz)
        norma_vector = norma_vector if norma_vector != 0 else 1e-9
        return (matriz @ vector) / (norma_vector * normas_matriz)

    def detectar_intencion(self, texto, umbral=0.15):
        """Devuelve (tag, confianza) de la intención más parecida, o (None, confianza) si no supera el umbral."""
        if not self.entrenado:
            self.entrenar()
        vector_entrada = self.vectorizador.transformar([texto])[0]
        similitudes = self._similitud_coseno(vector_entrada, self.matriz_patrones)
        indice_mejor = int(np.argmax(similitudes))
        mejor_similitud = float(similitudes[indice_mejor])

        if mejor_similitud < umbral:
            return None, mejor_similitud
        return self.etiquetas_patrones[indice_mejor], mejor_similitud

    def responder(self, texto):
        """Devuelve una respuesta adecuada al texto de entrada."""
        tag, _confianza = self.detectar_intencion(texto)
        if tag is None:
            return "No estoy seguro de haber entendido. ¿Puedes reformular tu pregunta?"
        for intencion in self.intenciones:
            if intencion["tag"] == tag:
                return random.choice(intencion["respuestas"])
        return "..."

    def chatear_en_consola(self):
        """Modo interactivo por terminal. Escribe 'salir' para terminar."""
        print(f"{self.nombre}: ¡Hola! Escribe 'salir' para terminar la conversación.\n")
        while True:
            entrada = input("Tú: ")
            if entrada.lower().strip() in ("salir", "adios", "adiós", "exit", "quit"):
                print(f"{self.nombre}: ¡Hasta luego!")
                break
            print(f"{self.nombre}: {self.responder(entrada)}")