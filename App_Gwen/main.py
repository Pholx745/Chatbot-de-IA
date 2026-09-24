#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MI_GWEN.PY — Framework de Inteligencia Artificial propio (archivo único)
================================================================================

Un solo archivo, sin dependencias externas más allá de NumPy, que contiene
CUATRO sistemas de IA completos y funcionales, construidos desde cero:

    1) NÚCLEO DE RED NEURONAL   -> motor de deep learning (backpropagation,
                                    Adam/SGD/RMSProp, dropout, softmax, etc.)
    2) MACHINE LEARNING CLÁSICO -> regresión, clasificación (4 algoritmos)
                                    y clustering (K-Medias)
    3) NLP + CHATBOT            -> tokenización, TF-IDF y chatbot por
                                    intenciones (similitud coseno)
    4) ASISTENTE CONVERSACIONAL -> memoria persistente + herramientas
                                    (function calling) + conector LLM opcional

CÓMO EJECUTARLO
---------------
    python3 mi_ia.py

    (en Linux/Mac también puedes hacerlo directamente ejecutable con:
        chmod +x mi_ia.py
        ./mi_ia.py
    )

Esto abre un menú interactivo por consola para probar cada sistema.

CÓMO USARLO EN TU PROPIO CÓDIGO
--------------------------------
    from mi_ia import RedNeuronal, CapaDensa, CapaActivacion, Adam, EntropiaCruzadaBinaria
    from mi_ia import RegresionLineal, KMedias, ArbolDecision
    from mi_ia import Chatbot
    from mi_ia import AsistenteIA

    asistente = AsistenteIA(nombre="Gwen")
    print(asistente.responder("hola"))

Requisitos: Python 3.8+ y NumPy (pip install numpy).
El conector LLM opcional necesita además 'requests' (pip install requests),
solo si decides enchufar un modelo de lenguaje externo por API.

Autor: generado por Rodricks y kosko a partir de Jarvis.
================================================================================
"""

# ============================================================================
# IMPORTS GLOBALES (todo el archivo usa únicamente estos)
# ============================================================================
import os
import re
import ast
import json
import random
import pickle
import operator
import datetime
import numpy as np
import threading
from collections import Counter

try:
    import ollama
except ImportError:
    ollama = None

def crear_motor_ollama(modelo="llama3.2:1b"):
    def motor(mensajes):
        if ollama is None:
            raise ImportError("La librería 'ollama' no está instalada.")
        respuesta = ollama.chat(model=modelo, messages=mensajes)
        return respuesta['message']['content']
    return motor 

# ============================================================================
# SECCIÓN 1: NÚCLEO DE RED NEURONAL (DEEP LEARNING DESDE CERO CON NUMPY)
# ============================================================================

# ---------------------------------------------------------------- #
# 1a. Funciones de activación y sus derivadas
# ---------------------------------------------------------------- #
def relu(x):
    """Rectified Linear Unit: max(0, x). La más usada en capas ocultas."""
    return np.maximum(0, x)


def relu_derivada(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    """Variante de ReLU que no "mata" neuronas con valores negativos."""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivada(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def sigmoide(x):
    """Aplasta la salida entre 0 y 1. Útil en la capa final de clasificación binaria."""
    x_seguro = np.clip(x, -500, 500)  # evita overflow en np.exp
    return 1.0 / (1.0 + np.exp(-x_seguro))


def sigmoide_derivada(x):
    s = sigmoide(x)
    return s * (1 - s)


def tanh(x):
    """Similar a la sigmoide pero centrada en 0 (rango -1 a 1)."""
    return np.tanh(x)


def tanh_derivada(x):
    return 1.0 - np.tanh(x) ** 2


def lineal(x):
    """Sin transformación. Se usa en la capa final para regresión."""
    return x


def lineal_derivada(x):
    return np.ones_like(x)


def softmax(x):
    """Convierte un vector de valores en una distribución de probabilidad (clasificación multiclase)."""
    x_estable = x - np.max(x, axis=1, keepdims=True)  # estabilidad numérica
    exponenciales = np.exp(x_estable)
    return exponenciales / np.sum(exponenciales, axis=1, keepdims=True)


ACTIVACIONES = {
    "relu": (relu, relu_derivada),
    "leaky_relu": (leaky_relu, leaky_relu_derivada),
    "sigmoide": (sigmoide, sigmoide_derivada),
    "tanh": (tanh, tanh_derivada),
    "lineal": (lineal, lineal_derivada),
}


# ---------------------------------------------------------------- #
# 1b. Capas de la red neuronal
# ---------------------------------------------------------------- #
class Capa:
    """Clase base de la que heredan todas las capas."""

    def __init__(self):
        self.entrada = None
        self.salida = None
        self.entrenable = False

    def hacia_adelante(self, entrada):
        raise NotImplementedError

    def hacia_atras(self, gradiente_salida):
        raise NotImplementedError


class CapaDensa(Capa):
    """Capa totalmente conectada: salida = entrada @ pesos + sesgos."""

    def __init__(self, n_entradas, n_neuronas, inicializacion="he"):
        super().__init__()
        self.entrenable = True

        if inicializacion == "he":        # recomendado con ReLU
            escala = np.sqrt(2.0 / n_entradas)
        elif inicializacion == "xavier":  # recomendado con tanh/sigmoide
            escala = np.sqrt(1.0 / n_entradas)
        else:
            escala = 0.01

        self.pesos = np.random.randn(n_entradas, n_neuronas) * escala
        self.sesgos = np.zeros((1, n_neuronas))
        self.grad_pesos = None
        self.grad_sesgos = None

    def hacia_adelante(self, entrada):
        self.entrada = entrada
        self.salida = entrada @ self.pesos + self.sesgos
        return self.salida

    def hacia_atras(self, gradiente_salida):
        self.grad_pesos = self.entrada.T @ gradiente_salida
        self.grad_sesgos = np.sum(gradiente_salida, axis=0, keepdims=True)
        return gradiente_salida @ self.pesos.T


class CapaActivacion(Capa):
    """Envuelve una función de activación (relu, sigmoide, tanh, leaky_relu, lineal)."""

    def __init__(self, nombre_activacion):
        super().__init__()
        if nombre_activacion not in ACTIVACIONES:
            raise ValueError(f"Activación '{nombre_activacion}' no reconocida. Usa una de: {list(ACTIVACIONES)}")
        self.nombre = nombre_activacion
        self.funcion, self.derivada = ACTIVACIONES[nombre_activacion]

    def hacia_adelante(self, entrada):
        self.entrada = entrada
        self.salida = self.funcion(entrada)
        return self.salida

    def hacia_atras(self, gradiente_salida):
        return gradiente_salida * self.derivada(self.entrada)


class CapaSoftmax(Capa):
    """
    Capa Softmax para clasificación multiclase (siempre como última capa),
    combinada con EntropiaCruzadaCategorica, que ya calcula el gradiente
    combinado (y_pred - y_real); por eso hacia_atras() deja pasar el gradiente.
    """

    def hacia_adelante(self, entrada):
        self.entrada = entrada
        self.salida = softmax(entrada)
        return self.salida

    def hacia_atras(self, gradiente_salida):
        return gradiente_salida


class CapaAbandono(Capa):
    """Dropout: apaga aleatoriamente neuronas durante el entrenamiento para evitar sobreajuste."""

    def __init__(self, tasa=0.2):
        super().__init__()
        self.tasa = tasa
        self.mascara = None

    def hacia_adelante(self, entrada, modo_entrenamiento=True):
        if modo_entrenamiento:
            self.mascara = (np.random.rand(*entrada.shape) > self.tasa) / (1 - self.tasa)
            return entrada * self.mascara
        return entrada

    def hacia_atras(self, gradiente_salida):
        return gradiente_salida * self.mascara


# ---------------------------------------------------------------- #
# 1c. Funciones de pérdida
# ---------------------------------------------------------------- #
class Perdida:
    def calcular(self, y_pred, y_real):
        raise NotImplementedError

    def gradiente(self, y_pred, y_real):
        raise NotImplementedError


class ErrorCuadraticoMedio(Perdida):
    """Para REGRESIÓN. Combinar con activación 'lineal' en la última capa."""

    def calcular(self, y_pred, y_real):
        return np.mean((y_pred - y_real) ** 2)

    def gradiente(self, y_pred, y_real):
        return 2 * (y_pred - y_real) / y_real.shape[0]


class EntropiaCruzadaBinaria(Perdida):
    """Para clasificación BINARIA. Combinar con activación 'sigmoide' en la última capa."""

    def calcular(self, y_pred, y_real):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

    def gradiente(self, y_pred, y_real):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (y_pred - y_real) / (y_pred * (1 - y_pred)) / y_real.shape[0]


class EntropiaCruzadaCategorica(Perdida):
    """Para clasificación MULTICLASE. Combinar con CapaSoftmax. y_real en formato one-hot."""

    def calcular(self, y_pred, y_real):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_real * np.log(y_pred), axis=1))

    def gradiente(self, y_pred, y_real):
        return (y_pred - y_real) / y_real.shape[0]


PERDIDAS = {
    "ecm": ErrorCuadraticoMedio,
    "entropia_binaria": EntropiaCruzadaBinaria,
    "entropia_categorica": EntropiaCruzadaCategorica,
}


# ---------------------------------------------------------------- #
# 1d. Optimizadores
# ---------------------------------------------------------------- #
class Optimizador:
    def actualizar(self, capa):
        raise NotImplementedError


class SGD(Optimizador):
    """Descenso de gradiente estocástico, con momento opcional."""

    def __init__(self, tasa_aprendizaje=0.01, momento=0.0):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.momento = momento
        self._velocidades = {}

    def actualizar(self, capa):
        clave = id(capa)
        if clave not in self._velocidades:
            self._velocidades[clave] = {"pesos": np.zeros_like(capa.pesos), "sesgos": np.zeros_like(capa.sesgos)}
        v = self._velocidades[clave]
        v["pesos"] = self.momento * v["pesos"] - self.tasa_aprendizaje * capa.grad_pesos
        v["sesgos"] = self.momento * v["sesgos"] - self.tasa_aprendizaje * capa.grad_sesgos
        capa.pesos += v["pesos"]
        capa.sesgos += v["sesgos"]


class RMSProp(Optimizador):
    """Adapta la tasa de aprendizaje por parámetro usando un promedio móvil del gradiente al cuadrado."""

    def __init__(self, tasa_aprendizaje=0.001, beta=0.9, epsilon=1e-8):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.beta = beta
        self.epsilon = epsilon
        self._cache = {}

    def actualizar(self, capa):
        clave = id(capa)
        if clave not in self._cache:
            self._cache[clave] = {"pesos": np.zeros_like(capa.pesos), "sesgos": np.zeros_like(capa.sesgos)}
        c = self._cache[clave]
        c["pesos"] = self.beta * c["pesos"] + (1 - self.beta) * capa.grad_pesos ** 2
        c["sesgos"] = self.beta * c["sesgos"] + (1 - self.beta) * capa.grad_sesgos ** 2
        capa.pesos -= self.tasa_aprendizaje * capa.grad_pesos / (np.sqrt(c["pesos"]) + self.epsilon)
        capa.sesgos -= self.tasa_aprendizaje * capa.grad_sesgos / (np.sqrt(c["sesgos"]) + self.epsilon)


class Adam(Optimizador):
    """El optimizador más usado en la práctica: combina momento con tasa adaptativa por parámetro."""

    def __init__(self, tasa_aprendizaje=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m = {}
        self._v = {}
        self._t = {}

    def actualizar(self, capa):
        clave = id(capa)
        if clave not in self._m:
            self._m[clave] = {"pesos": np.zeros_like(capa.pesos), "sesgos": np.zeros_like(capa.sesgos)}
            self._v[clave] = {"pesos": np.zeros_like(capa.pesos), "sesgos": np.zeros_like(capa.sesgos)}
            self._t[clave] = 0

        self._t[clave] += 1
        t = self._t[clave]
        m, v = self._m[clave], self._v[clave]

        for nombre, grad in (("pesos", capa.grad_pesos), ("sesgos", capa.grad_sesgos)):
            m[nombre] = self.beta1 * m[nombre] + (1 - self.beta1) * grad
            v[nombre] = self.beta2 * v[nombre] + (1 - self.beta2) * grad ** 2
            m_corregido = m[nombre] / (1 - self.beta1 ** t)
            v_corregido = v[nombre] / (1 - self.beta2 ** t)
            actualizacion = self.tasa_aprendizaje * m_corregido / (np.sqrt(v_corregido) + self.epsilon)
            if nombre == "pesos":
                capa.pesos -= actualizacion
            else:
                capa.sesgos -= actualizacion


OPTIMIZADORES = {"sgd": SGD, "rmsprop": RMSProp, "adam": Adam}


# ---------------------------------------------------------------- #
# 1e. Clase principal RedNeuronal (modelo secuencial)
# ---------------------------------------------------------------- #
class RedNeuronal:
    def __init__(self):
        self.capas = []
        self.perdida = None
        self.optimizador = None
        self.historial = {"perdida": [], "perdida_val": []}

    def agregar(self, capa):
        """Agrega una capa al final de la red. Permite encadenar llamadas."""
        self.capas.append(capa)
        return self

    def compilar(self, perdida, optimizador):
        self.perdida = perdida
        self.optimizador = optimizador
        return self

    def predecir(self, X, modo_entrenamiento=False):
        salida = X
        for capa in self.capas:
            if isinstance(capa, CapaAbandono):
                salida = capa.hacia_adelante(salida, modo_entrenamiento=modo_entrenamiento)
            else:
                salida = capa.hacia_adelante(salida)
        return salida

    def entrenar(self, X, y, epocas=100, tamano_lote=32, verbose=True,
                 X_val=None, y_val=None, paciencia=None):
        """Entrena con descenso de gradiente por mini-lotes. 'paciencia' activa early stopping."""
        if self.perdida is None or self.optimizador is None:
            raise RuntimeError("Debes llamar a .compilar(perdida, optimizador) antes de entrenar.")

        n_muestras = X.shape[0]
        mejor_perdida_val = np.inf
        epocas_sin_mejora = 0

        for epoca in range(epocas):
            indices = np.random.permutation(n_muestras)
            X_barajado, y_barajado = X[indices], y[indices]
            perdida_acumulada, n_lotes = 0.0, 0

            for inicio in range(0, n_muestras, tamano_lote):
                fin = inicio + tamano_lote
                X_lote, y_lote = X_barajado[inicio:fin], y_barajado[inicio:fin]

                y_pred = self.predecir(X_lote, modo_entrenamiento=True)
                perdida_acumulada += self.perdida.calcular(y_pred, y_lote)
                n_lotes += 1

                grad = self.perdida.gradiente(y_pred, y_lote)
                for capa in reversed(self.capas):
                    grad = capa.hacia_atras(grad)
                    if capa.entrenable:
                        self.optimizador.actualizar(capa)

            perdida_promedio = perdida_acumulada / n_lotes
            self.historial["perdida"].append(perdida_promedio)

            perdida_val = None
            if X_val is not None and y_val is not None:
                perdida_val = self.perdida.calcular(self.predecir(X_val), y_val)
                self.historial["perdida_val"].append(perdida_val)

                if paciencia is not None:
                    if perdida_val < mejor_perdida_val:
                        mejor_perdida_val, epocas_sin_mejora = perdida_val, 0
                    else:
                        epocas_sin_mejora += 1
                    if epocas_sin_mejora >= paciencia:
                        if verbose:
                            print(f"Early stopping en la época {epoca + 1} (sin mejora en {paciencia} épocas).")
                        break

            if verbose and (epoca % max(1, epocas // 10) == 0 or epoca == epocas - 1):
                mensaje = f"Época {epoca + 1}/{epocas} - pérdida: {perdida_promedio:.4f}"
                if perdida_val is not None:
                    mensaje += f" - pérdida_val: {perdida_val:.4f}"
                print(mensaje)

        return self.historial

    def evaluar(self, X, y):
        return self.perdida.calcular(self.predecir(X), y)

    def guardar(self, ruta):
        with open(ruta, "wb") as archivo:
            pickle.dump(self, archivo)

    @staticmethod
    def cargar(ruta):
        with open(ruta, "rb") as archivo:
            return pickle.load(archivo)

    def resumen(self):
        print("=" * 55)
        print("RESUMEN DE LA RED NEURONAL")
        print("=" * 55)
        total_parametros = 0
        for i, capa in enumerate(self.capas):
            detalle = ""
            if isinstance(capa, CapaDensa):
                n_params = capa.pesos.size + capa.sesgos.size
                total_parametros += n_params
                detalle = f"({capa.pesos.shape[0]} -> {capa.pesos.shape[1]})  parámetros: {n_params}"
            print(f"  [{i + 1}] {type(capa).__name__:<15} {detalle}")
        print("-" * 55)
        print(f"Total de parámetros entrenables: {total_parametros}")
        print("=" * 55)


# ============================================================================
# SECCIÓN 2: MACHINE LEARNING CLÁSICO
# ============================================================================

# ---------------------------------------------------------------- #
# 2a. Preprocesamiento de datos
# ---------------------------------------------------------------- #
def dividir_datos(X, y, proporcion_prueba=0.2, semilla=None):
    """Divide X e y en conjuntos de entrenamiento y prueba de forma aleatoria."""
    if semilla is not None:
        np.random.seed(semilla)
    n = X.shape[0]
    indices = np.random.permutation(n)
    n_prueba, = (int(n * proporcion_prueba),)
    idx_prueba, idx_entrenamiento = indices[:n_prueba], indices[n_prueba:]
    return X[idx_entrenamiento], X[idx_prueba], y[idx_entrenamiento], y[idx_prueba]


def normalizar_estandar(X, media=None, desviacion=None):
    """Escala los datos para que tengan media 0 y desviación estándar 1 (z-score)."""
    if media is None:
        media = X.mean(axis=0)
    if desviacion is None:
        desviacion = X.std(axis=0)
        desviacion = np.where(desviacion == 0, 1, desviacion)
    return (X - media) / desviacion, media, desviacion


def normalizar_min_max(X, minimo=None, maximo=None):
    """Escala los datos al rango [0, 1]."""
    if minimo is None:
        minimo = X.min(axis=0)
    if maximo is None:
        maximo = X.max(axis=0)
    rango = np.where((maximo - minimo) == 0, 1, maximo - minimo)
    return (X - minimo) / rango, minimo, maximo


def codificar_one_hot(y, n_clases=None):
    """Convierte etiquetas enteras (0,1,2,...) en vectores one-hot."""
    y = np.asarray(y).astype(int).ravel()
    if n_clases is None:
        n_clases = int(y.max()) + 1
    one_hot = np.zeros((y.shape[0], n_clases))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def decodificar_one_hot(y_one_hot):
    """Convierte vectores one-hot de vuelta a etiquetas enteras."""
    return np.argmax(y_one_hot, axis=1)


# ---------------------------------------------------------------- #
# 2b. Métricas de evaluación
# ---------------------------------------------------------------- #
def exactitud(y_real, y_pred):
    return float(np.mean(np.asarray(y_real).ravel() == np.asarray(y_pred).ravel()))


def matriz_confusion(y_real, y_pred, n_clases=None):
    y_real, y_pred = np.asarray(y_real).ravel().astype(int), np.asarray(y_pred).ravel().astype(int)
    if n_clases is None:
        n_clases = int(max(y_real.max(), y_pred.max())) + 1
    matriz = np.zeros((n_clases, n_clases), dtype=int)
    for real, pred in zip(y_real, y_pred):
        matriz[real][pred] += 1
    return matriz


def precision(y_real, y_pred, clase_positiva=1):
    y_real, y_pred = np.asarray(y_real).ravel(), np.asarray(y_pred).ravel()
    vp = np.sum((y_pred == clase_positiva) & (y_real == clase_positiva))
    fp = np.sum((y_pred == clase_positiva) & (y_real != clase_positiva))
    return float(vp / (vp + fp)) if (vp + fp) > 0 else 0.0


def exhaustividad(y_real, y_pred, clase_positiva=1):
    """También conocida como 'recall' o sensibilidad."""
    y_real, y_pred = np.asarray(y_real).ravel(), np.asarray(y_pred).ravel()
    vp = np.sum((y_pred == clase_positiva) & (y_real == clase_positiva))
    fn = np.sum((y_pred != clase_positiva) & (y_real == clase_positiva))
    return float(vp / (vp + fn)) if (vp + fn) > 0 else 0.0


def puntuacion_f1(y_real, y_pred, clase_positiva=1):
    p, r = precision(y_real, y_pred, clase_positiva), exhaustividad(y_real, y_pred, clase_positiva)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def reporte_clasificacion(y_real, y_pred):
    """Imprime un resumen legible con las métricas principales por clase."""
    y_real, y_pred = np.asarray(y_real).ravel(), np.asarray(y_pred).ravel()
    clases = np.unique(np.concatenate([y_real, y_pred]))
    print(f"{'Clase':<10}{'Precisión':<12}{'Exhaustividad':<16}{'F1':<10}")
    for c in clases:
        print(f"{str(c):<10}{precision(y_real, y_pred, c):<12.3f}"
              f"{exhaustividad(y_real, y_pred, c):<16.3f}{puntuacion_f1(y_real, y_pred, c):<10.3f}")
    print(f"\nExactitud global: {exactitud(y_real, y_pred):.3f}")


def error_cuadratico_medio(y_real, y_pred):
    return float(np.mean((np.asarray(y_real).ravel() - np.asarray(y_pred).ravel()) ** 2))


def raiz_error_cuadratico_medio(y_real, y_pred):
    return float(np.sqrt(error_cuadratico_medio(y_real, y_pred)))


def error_absoluto_medio(y_real, y_pred):
    return float(np.mean(np.abs(np.asarray(y_real).ravel() - np.asarray(y_pred).ravel())))


def r_cuadrado(y_real, y_pred):
    """Coeficiente de determinación R²: qué tan bien explica el modelo la varianza de los datos."""
    y_real, y_pred = np.asarray(y_real).ravel(), np.asarray(y_pred).ravel()
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------- #
# 2c. Regresión
# ---------------------------------------------------------------- #
class RegresionLineal:
    """Regresión lineal entrenada con descenso de gradiente: y_pred = X @ pesos + sesgo."""

    def __init__(self, tasa_aprendizaje=0.01, iteraciones=1000, regularizacion_l2=0.0):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.regularizacion_l2 = regularizacion_l2
        self.pesos = None
        self.sesgo = None
        self.historial_perdida = []

    def entrenar(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
        n_muestras, n_caracteristicas = X.shape
        self.pesos, self.sesgo, self.historial_perdida = np.zeros(n_caracteristicas), 0.0, []

        for _ in range(self.iteraciones):
            y_pred = X @ self.pesos + self.sesgo
            error = y_pred - y
            grad_pesos = (X.T @ error) / n_muestras + self.regularizacion_l2 * self.pesos
            grad_sesgo = np.mean(error)
            self.pesos -= self.tasa_aprendizaje * grad_pesos
            self.sesgo -= self.tasa_aprendizaje * grad_sesgo
            self.historial_perdida.append(np.mean(error ** 2))
        return self

    def predecir(self, X):
        return np.asarray(X, dtype=float) @ self.pesos + self.sesgo


class RegresionPolinomica(RegresionLineal):
    """Como RegresionLineal, pero expande cada característica a sus potencias (x, x², x³, ...)."""

    def __init__(self, grado=2, **kwargs):
        super().__init__(**kwargs)
        self.grado = grado

    def _expandir(self, X):
        X = np.asarray(X, dtype=float)
        columnas = [X[:, i:i + 1] ** g for i in range(X.shape[1]) for g in range(1, self.grado + 1)]
        return np.hstack(columnas)

    def entrenar(self, X, y):
        return super().entrenar(self._expandir(X), y)

    def predecir(self, X):
        return super().predecir(self._expandir(X))


# ---------------------------------------------------------------- #
# 2d. Clasificación
# ---------------------------------------------------------------- #
class RegresionLogistica:
    def __init__(self, tasa_aprendizaje=0.1, iteraciones=1000):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = None
        self.sesgo = None

    def _sigmoide(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def entrenar(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
        n_muestras, n_caracteristicas = X.shape
        self.pesos, self.sesgo = np.zeros(n_caracteristicas), 0.0

        for _ in range(self.iteraciones):
            y_pred = self._sigmoide(X @ self.pesos + self.sesgo)
            error = y_pred - y
            self.pesos -= self.tasa_aprendizaje * (X.T @ error) / n_muestras
            self.sesgo -= self.tasa_aprendizaje * np.mean(error)
        return self

    def predecir_proba(self, X):
        return self._sigmoide(np.asarray(X, dtype=float) @ self.pesos + self.sesgo)

    def predecir(self, X, umbral=0.5):
        return (self.predecir_proba(X) >= umbral).astype(int)


class KVecinosCercanos:
    """K-Nearest Neighbors: no "entrena", solo memoriza los datos."""

    def __init__(self, k=5):
        self.k = k
        self.X_entrenamiento = None
        self.y_entrenamiento = None

    def entrenar(self, X, y):
        self.X_entrenamiento = np.asarray(X, dtype=float)
        self.y_entrenamiento = np.asarray(y).ravel()
        return self

    def predecir(self, X):
        X = np.asarray(X, dtype=float)
        predicciones = []
        for punto in X:
            distancias = np.sqrt(np.sum((self.X_entrenamiento - punto) ** 2, axis=1))
            vecinos = self.y_entrenamiento[np.argsort(distancias)[:self.k]]
            predicciones.append(Counter(vecinos.tolist()).most_common(1)[0][0])
        return np.array(predicciones)


class NaiveBayesGaussiano:
    """Asume que cada característica sigue una distribución normal dentro de cada clase."""

    def entrenar(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y).ravel()
        self.clases = np.unique(y)
        self.medias, self.varianzas, self.prioris = {}, {}, {}
        for c in self.clases:
            X_c = X[y == c]
            self.medias[c] = X_c.mean(axis=0)
            self.varianzas[c] = X_c.var(axis=0) + 1e-9
            self.prioris[c] = X_c.shape[0] / X.shape[0]
        return self

    def _log_verosimilitud(self, X, clase):
        media, varianza = self.medias[clase], self.varianzas[clase]
        return np.sum(-((X - media) ** 2) / (2 * varianza) - 0.5 * np.log(2 * np.pi * varianza), axis=1)

    def predecir(self, X):
        X = np.asarray(X, dtype=float)
        log_posteriores = np.array([np.log(self.prioris[c]) + self._log_verosimilitud(X, c)
                                     for c in self.clases]).T
        return self.clases[np.argmax(log_posteriores, axis=1)]


class _NodoArbol:
    __slots__ = ("caracteristica", "umbral", "izquierda", "derecha", "valor")

    def __init__(self, caracteristica=None, umbral=None, izquierda=None, derecha=None, valor=None):
        self.caracteristica, self.umbral = caracteristica, umbral
        self.izquierda, self.derecha, self.valor = izquierda, derecha, valor


class ArbolDecision:
    """Árbol de decisión (CART simplificado) usando el índice de Gini."""

    def __init__(self, profundidad_maxima=5, muestras_minimas=2):
        self.profundidad_maxima = profundidad_maxima
        self.muestras_minimas = muestras_minimas
        self.raiz = None

    def _gini(self, y):
        _, conteos = np.unique(y, return_counts=True)
        probabilidades = conteos / conteos.sum()
        return 1 - np.sum(probabilidades ** 2)

    def _mejor_division(self, X, y):
        mejor_ganancia, mejor_caracteristica, mejor_umbral = -1, None, None
        gini_padre = self._gini(y)
        n_muestras, n_caracteristicas = X.shape
        for caracteristica in range(n_caracteristicas):
            for umbral in np.unique(X[:, caracteristica]):
                mascara_izq = X[:, caracteristica] <= umbral
                y_izq, y_der = y[mascara_izq], y[~mascara_izq]
                if len(y_izq) == 0 or len(y_der) == 0:
                    continue
                gini_hijos = (len(y_izq) / n_muestras) * self._gini(y_izq) + \
                             (len(y_der) / n_muestras) * self._gini(y_der)
                ganancia = gini_padre - gini_hijos
                if ganancia > mejor_ganancia:
                    mejor_ganancia, mejor_caracteristica, mejor_umbral = ganancia, caracteristica, umbral
        return mejor_caracteristica, mejor_umbral, mejor_ganancia

    def _construir(self, X, y, profundidad=0):
        if (profundidad >= self.profundidad_maxima or X.shape[0] < self.muestras_minimas or
                len(np.unique(y)) == 1):
            return _NodoArbol(valor=Counter(y.tolist()).most_common(1)[0][0])
        caracteristica, umbral, ganancia = self._mejor_division(X, y)
        if caracteristica is None or ganancia <= 0:
            return _NodoArbol(valor=Counter(y.tolist()).most_common(1)[0][0])
        mascara_izq = X[:, caracteristica] <= umbral
        izquierda = self._construir(X[mascara_izq], y[mascara_izq], profundidad + 1)
        derecha = self._construir(X[~mascara_izq], y[~mascara_izq], profundidad + 1)
        return _NodoArbol(caracteristica=caracteristica, umbral=umbral, izquierda=izquierda, derecha=derecha)

    def entrenar(self, X, y):
        self.raiz = self._construir(np.asarray(X, dtype=float), np.asarray(y).ravel())
        return self

    def _predecir_uno(self, x, nodo):
        if nodo.valor is not None:
            return nodo.valor
        return self._predecir_uno(x, nodo.izquierda if x[nodo.caracteristica] <= nodo.umbral else nodo.derecha)

    def predecir(self, X):
        return np.array([self._predecir_uno(x, self.raiz) for x in np.asarray(X, dtype=float)])


# ---------------------------------------------------------------- #
# 2e. Agrupamiento (clustering)
# ---------------------------------------------------------------- #
class KMedias:
    """K-Means: agrupa los datos en 'n_grupos' clústeres minimizando la distancia a sus centroides."""

    def __init__(self, n_grupos=3, iteraciones_maximas=300, semilla=None):
        self.n_grupos = n_grupos
        self.iteraciones_maximas = iteraciones_maximas
        self.semilla = semilla
        self.centroides = None
        self.etiquetas = None

    def entrenar(self, X):
        X = np.asarray(X, dtype=float)
        if self.semilla is not None:
            np.random.seed(self.semilla)
        n_muestras = X.shape[0]
        self.centroides = X[np.random.choice(n_muestras, self.n_grupos, replace=False)].copy()

        for _ in range(self.iteraciones_maximas):
            distancias = np.array([np.sqrt(np.sum((X - c) ** 2, axis=1)) for c in self.centroides])
            etiquetas = np.argmin(distancias, axis=0)
            nuevos_centroides = np.array([
                X[etiquetas == i].mean(axis=0) if np.any(etiquetas == i) else self.centroides[i]
                for i in range(self.n_grupos)
            ])
            if np.allclose(nuevos_centroides, self.centroides):
                break
            self.centroides = nuevos_centroides

        self.etiquetas = etiquetas
        return self

    def predecir(self, X):
        X = np.asarray(X, dtype=float)
        distancias = np.array([np.sqrt(np.sum((X - c) ** 2, axis=1)) for c in self.centroides])
        return np.argmin(distancias, axis=0)

    def inercia(self, X):
        X = np.asarray(X, dtype=float)
        etiquetas = self.predecir(X)
        return float(sum(np.sum((X[etiquetas == i] - self.centroides[i]) ** 2) for i in range(self.n_grupos)))


# ============================================================================
# SECCIÓN 3: GENERADORES DE DATOS SINTÉTICOS (para probar todo sin datasets externos)
# ============================================================================
def generar_xor(n_muestras=200, ruido=0.15, semilla=42):
    """Problema clásico NO separable linealmente. Bueno para probar redes neuronales."""
    rng = np.random.default_rng(semilla)
    X = rng.uniform(-1, 1, size=(n_muestras, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    X = X + rng.normal(0, ruido, X.shape)
    return X, y.reshape(-1, 1)


def generar_blobs(n_muestras=300, n_grupos=3, dispersion=1.2, semilla=42):
    """Grupos de puntos alrededor de centros aleatorios. Bueno para clasificación y clustering."""
    rng = np.random.default_rng(semilla)
    centros = rng.uniform(-10, 10, size=(n_grupos, 2))
    listas_X, listas_y = [], []
    por_grupo = n_muestras // n_grupos
    for i, centro in enumerate(centros):
        listas_X.append(centro + rng.normal(0, dispersion, size=(por_grupo, 2)))
        listas_y.extend([i] * por_grupo)
    return np.vstack(listas_X), np.array(listas_y)


def generar_lunas(n_muestras=300, ruido=0.12, semilla=42):
    """Dos grupos con forma de media luna entrelazadas. NO separables linealmente."""
    rng = np.random.default_rng(semilla)
    n = n_muestras // 2
    angulo1, angulo2 = rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)
    X1 = np.column_stack([np.cos(angulo1), np.sin(angulo1)])
    X2 = np.column_stack([1 - np.cos(angulo2), 1 - np.sin(angulo2) - 0.5])
    X = np.vstack([X1, X2]) + rng.normal(0, ruido, (2 * n, 2))
    return X, np.array([0] * n + [1] * n)


def generar_lineal(n_muestras=100, pendiente=2.5, intercepto=1.0, ruido=1.0, semilla=42):
    """Datos con relación lineal + ruido gaussiano. Bueno para probar regresión."""
    rng = np.random.default_rng(semilla)
    X = rng.uniform(0, 10, size=(n_muestras, 1))
    y = pendiente * X[:, 0] + intercepto + rng.normal(0, ruido, n_muestras)
    return X, y


# ============================================================================
# SECCIÓN 4: PROCESAMIENTO DE LENGUAJE NATURAL (NLP) Y CHATBOT
# ============================================================================

# ---------------------------------------------------------------- #
# 4a. Tokenización y limpieza de texto en español
# ---------------------------------------------------------------- #
PALABRAS_VACIAS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
    "mas", "pero", "sus", "le", "ya", "o", "este", "si",
    "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "tambien",
    "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos",
    "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese",
    "eso", "ante", "ellos", "e", "esto", "mi", "antes", "algunos",
    "unos", "yo", "otro", "otras", "otra", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar",
    "estas", "algunas", "algo", "nosotros", "mis", "tu", "te", "ti", "tus",
    "ellas", "nosotras", "vosotros", "vosotras", "os", "es", "soy", "eres",
    "somos", "sois", "son",
}

_TABLA_ACENTOS = str.maketrans("áéíóúü", "aeiouu")


def quitar_acentos(texto):
    return texto.translate(_TABLA_ACENTOS)


def limpiar_texto(texto):
    """Minúsculas, sin acentos, sin signos de puntuación, espacios normalizados."""
    texto = quitar_acentos(texto.lower().strip())
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto):
    return limpiar_texto(texto).split()


def quitar_palabras_vacias(tokens):
    return [t for t in tokens if t not in PALABRAS_VACIAS_ES]


def preprocesar(texto, quitar_vacias=True):
    tokens = tokenizar(texto)
    return quitar_palabras_vacias(tokens) if quitar_vacias else tokens


# ---------------------------------------------------------------- #
# 4b. Vectorización de texto: Bolsa de Palabras y TF-IDF
# ---------------------------------------------------------------- #
class BolsaDePalabras:
    """Bag of Words: representa cada texto como un vector de conteo de palabras."""

    def __init__(self, quitar_vacias=True):
        self.quitar_vacias = quitar_vacias
        self.vocabulario = {}

    def ajustar(self, textos):
        palabras = set()
        for texto in textos:
            palabras.update(preprocesar(texto, self.quitar_vacias))
        self.vocabulario = {palabra: i for i, palabra in enumerate(sorted(palabras))}
        return self

    def transformar(self, textos):
        matriz = np.zeros((len(textos), len(self.vocabulario)))
        for i, texto in enumerate(textos):
            for palabra, cuenta in Counter(preprocesar(texto, self.quitar_vacias)).items():
                if palabra in self.vocabulario:
                    matriz[i, self.vocabulario[palabra]] = cuenta
        return matriz

    def ajustar_transformar(self, textos):
        self.ajustar(textos)
        return self.transformar(textos)


class TFIDF(BolsaDePalabras):
    """Como BolsaDePalabras, pero reduce el peso de palabras muy comunes entre documentos."""

    def ajustar(self, textos):
        super().ajustar(textos)
        n_documentos = len(textos)
        self.idf = np.zeros(len(self.vocabulario))
        for palabra, indice in self.vocabulario.items():
            n_docs_con_palabra = sum(1 for t in textos if palabra in preprocesar(t, self.quitar_vacias))
            self.idf[indice] = np.log((n_documentos + 1) / (n_docs_con_palabra + 1)) + 1
        return self

    def transformar(self, textos):
        tf = super().transformar(textos)
        sumas = np.where(tf.sum(axis=1, keepdims=True) == 0, 1, tf.sum(axis=1, keepdims=True))
        return (tf / sumas) * self.idf


# ---------------------------------------------------------------- #
# 4c. Chatbot basado en intenciones (similitud coseno sobre TF-IDF)
# ---------------------------------------------------------------- #
class Chatbot:
    """
    A cada intención se le asocian patrones de ejemplo y respuestas posibles.
    Se usa TF-IDF SIN quitar palabras vacías, porque los patrones son frases muy
    cortas ("quién eres") y quitarlas podría dejar la frase vacía.
    """

    def __init__(self, nombre="Asistente"):
        self.nombre = nombre
        self.intenciones = []
        self.vectorizador = TFIDF(quitar_vacias=False)
        self.matriz_patrones = None
        self.etiquetas_patrones = []
        self.entrenado = False

    def agregar_intencion(self, tag, patrones, respuestas):
        self.intenciones.append({"tag": tag, "patrones": patrones, "respuestas": respuestas})
        self.entrenado = False
        return self

    def entrenar(self):
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
        norma_vector = np.linalg.norm(vector) or 1e-9
        normas_matriz = np.where(np.linalg.norm(matriz, axis=1) == 0, 1e-9, np.linalg.norm(matriz, axis=1))
        return (matriz @ vector) / (norma_vector * normas_matriz)

    def detectar_intencion(self, texto, umbral=0.15):
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
        if hasattr(self, 'motor_llm') and self.motor_llm is not None:
            return self.motor_llm([{"role": "user", "content": texto}])
            
        # Fallback al sistema de intenciones local si no hay LLM
        tag, _ = self.detectar_intencion(texto)
        if tag is None:
            return "No estoy seguro de haber entendido. ¿Puedes reformular tu pregunta?"
            
        for intencion in self.intenciones:
            if intencion["tag"] == tag:
                return random.choice(intencion["respuestas"])
                
        return "..."

    def chatear_en_consola(self):
        print(f"{self.nombre}: ¡Hola! Escribe 'salir' para terminar la conversación.\n")
        while True:
            entrada = input("Tú: ")
            if entrada.lower().strip() in ("salir", "adios", "adiós", "exit", "quit"):
                print(f"{self.nombre}: ¡Hasta luego!")
                break
            print(f"{self.nombre}: {self.responder(entrada)}")


# ============================================================================
# SECCIÓN 5: ASISTENTE CONVERSACIONAL (MEMORIA + HERRAMIENTAS + CHATBOT)
# ============================================================================

# ---------------------------------------------------------------- #
# 5a. Memoria de conversación con persistencia opcional en JSON
# ---------------------------------------------------------------- #
class MemoriaConversacion:
    def __init__(self, limite_turnos=50, ruta_persistencia=None):
        self.turnos = []
        self.limite_turnos = limite_turnos
        self.ruta_persistencia = ruta_persistencia
        self.datos_usuario = {}
        if self.ruta_persistencia and os.path.exists(self.ruta_persistencia):
            self.cargar()

    def agregar_turno(self, rol, texto):
        self.turnos.append({
            "rol": rol, "texto": texto,
            "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        if len(self.turnos) > self.limite_turnos:
            self.turnos = self.turnos[-self.limite_turnos:]
        if self.ruta_persistencia:
            self.guardar()

    def recordar(self, clave, valor):
        self.datos_usuario[clave] = valor
        if self.ruta_persistencia:
            self.guardar()

    def obtener_recuerdo(self, clave, por_defecto=None):
        return self.datos_usuario.get(clave, por_defecto)

    def historial_como_texto(self, ultimos_n=10):
        return "\n".join(f"{t['rol']}: {t['texto']}" for t in self.turnos[-ultimos_n:])

    def guardar(self):
        with open(self.ruta_persistencia, "w", encoding="utf-8") as archivo:
            json.dump({"turnos": self.turnos, "datos_usuario": self.datos_usuario},
                      archivo, ensure_ascii=False, indent=2)

    def cargar(self):
        with open(self.ruta_persistencia, "r", encoding="utf-8") as archivo:
            contenido = json.load(archivo)
        self.turnos = contenido.get("turnos", [])
        self.datos_usuario = contenido.get("datos_usuario", {})

    def limpiar(self):
        self.turnos, self.datos_usuario = [], {}
        if self.ruta_persistencia:
            self.guardar()


# ---------------------------------------------------------------- #
# 5b. Herramientas (function calling) que el asistente puede invocar
# ---------------------------------------------------------------- #
class RegistroHerramientas:
    """
    Guarda funciones registradas y permite ejecutarlas por nombre. Para agregar
    tu propia herramienta:

        @herramientas.registrar("mi_herramienta", "Qué hace")
        def mi_herramienta(argumento):
            return resultado
    """

    def __init__(self):
        self._herramientas = {}

    def registrar(self, nombre=None, descripcion=""):
        def decorador(funcion):
            self._herramientas[nombre or funcion.__name__] = {"funcion": funcion, "descripcion": descripcion}
            return funcion
        return decorador

    def ejecutar(self, nombre, *args, **kwargs):
        if nombre not in self._herramientas:
            raise ValueError(f"Herramienta '{nombre}' no encontrada. Disponibles: {list(self._herramientas)}")
        return self._herramientas[nombre]["funcion"](*args, **kwargs)

    def listar(self):
        return {nombre: info["descripcion"] for nombre, info in self._herramientas.items()}


herramientas = RegistroHerramientas()

# --- Calculadora segura (sin eval crudo): mini-intérprete aritmético con 'ast' ---
_OPERADORES_PERMITIDOS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _evaluar_nodo(nodo):
    if isinstance(nodo, ast.Constant):
        if isinstance(nodo.value, (int, float)):
            return nodo.value
        raise ValueError("Solo se permiten números en la expresión.")
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES_PERMITIDOS:
        return _OPERADORES_PERMITIDOS[type(nodo.op)](_evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right))
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES_PERMITIDOS:
        return _OPERADORES_PERMITIDOS[type(nodo.op)](_evaluar_nodo(nodo.operand))
    raise ValueError("Expresión matemática no permitida.")


@herramientas.registrar("calculadora", "Evalúa una expresión aritmética. Ej: calculadora('3 + 4 * (2 - 1)')")
def calculadora(expresion):
    try:
        return _evaluar_nodo(ast.parse(expresion, mode="eval").body)
    except Exception as error:
        return f"Error al calcular: {error}"


@herramientas.registrar("fecha_hora", "Devuelve la fecha y hora actuales del sistema")
def fecha_hora():
    ahora = datetime.datetime.now()
    dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    return (f"{dias[ahora.weekday()]} {ahora.day} de {meses[ahora.month - 1]} de {ahora.year}, "
            f"{ahora.strftime('%H:%M:%S')}")


@herramientas.registrar("convertir_temperatura", "Convierte temperaturas entre C, F y K")
def convertir_temperatura(valor, origen="C", destino="F"):
    origen, destino = origen.upper(), destino.upper()
    if origen == "C":
        celsius = valor
    elif origen == "F":
        celsius = (valor - 32) * 5 / 9
    elif origen == "K":
        celsius = valor - 273.15
    else:
        return "Unidad de origen no reconocida (usa C, F o K)."

    if destino == "C":
        return round(celsius, 2)
    elif destino == "F":
        return round(celsius * 9 / 5 + 32, 2)
    elif destino == "K":
        return round(celsius + 273.15, 2)
    return "Unidad de destino no reconocida (usa C, F o K)."


@herramientas.registrar("contar_texto", "Cuenta palabras y caracteres de un texto")
def contar_texto(texto):
    return {
        "palabras": len(texto.split()),
        "caracteres": len(texto),
        "caracteres_sin_espacios": len(texto.replace(" ", "")),
    }


@herramientas.registrar("convertir_moneda_simple", "Convierte un monto usando una tasa de cambio dada por el usuario")
def convertir_moneda_simple(monto, tasa_cambio):
    return round(monto * tasa_cambio, 2)


# ------------------------------------------------------------------ #
# 5d. Conector Local a Ollama (Llama 3.2)
# ------------------------------------------------------------------ #
def crear_motor_ollama(modelo="llama3.2:1b"):
    def motor(mensajes):
        if ollama is None:
            raise ImportError("La librería 'ollama' no está instalada o no se importó correctamente.")

        respuesta = ollama.chat(model=modelo, messages=mensajes)
        contenido = respuesta['message']['content']
        
        # Limpiar posibles prefijos no deseados que devuelva la API
        if contenido.startswith("assistant\n"):
            contenido = contenido.replace("assistant\n", "", 1)
        elif contenido.startswith("assistant:"):
            contenido = contenido.replace("assistant:", "", 1)
            
        return contenido.strip()

    return motor
      
    if requests is None:
        raise ImportError(
            "Este conector requiere 'requests'. "
            "Instálalo con: pip install requests"
        )

    # ------------------------------------------------------------
    # Obtener API KEY
    # ------------------------------------------------------------

    api_key = os.environ.get(variable_entorno_api_key)

    if not api_key:
        raise EnvironmentError(
            f"No se encontró la variable de entorno "
            f"'{variable_entorno_api_key}'."
        )

    # ------------------------------------------------------------
    # MOTOR LLM
    # ------------------------------------------------------------

    def motor_llm(mensajes):
        """
        Recibe una lista estructurada de mensajes.

        Ejemplo:

        [
            {
                "role": "system",
                "content": "Eres GWEN..."
            },
            {
                "role": "user",
                "content": "Hola"
            },
            {
                "role": "assistant",
                "content": "¡Hola!"
            },
            {
                "role": "user",
                "content": "¿Cómo estás?"
            }
        ]
        """

        if not isinstance(mensajes, list):
            raise TypeError(
                "El motor LLM esperaba una lista de mensajes."
            )

        # --------------------------------------------------------
        # Validar mensajes
        # --------------------------------------------------------

        mensajes_validos = []

        for mensaje in mensajes:

            if not isinstance(mensaje, dict):
                continue

            role = mensaje.get("role")
            content = mensaje.get("content")

            if role not in (
                "system",
                "user",
                "assistant",
                "tool",
            ):
                continue

            if content is None:
                continue

            content = str(content).strip()

            if not content:
                continue

            mensajes_validos.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        if not mensajes_validos:
            raise ValueError(
                "No hay mensajes válidos para enviar al modelo."
            )

        # --------------------------------------------------------
        # Headers
        # --------------------------------------------------------

        encabezados = {
            "Content-Type": "application/json",
            encabezado_auth: f"Bearer {api_key}",
        }

        # --------------------------------------------------------
        # Cuerpo de la petición
        # --------------------------------------------------------

        cuerpo = {
            "model": modelo,
            "messages": mensajes_validos,
            "max_tokens": max_tokens,
            "temperature": temperatura,
        }

        # --------------------------------------------------------
        # Petición HTTP
        # --------------------------------------------------------

        try:

            respuesta = requests.post(
                url_api,
                headers=encabezados,
                json=cuerpo,
                timeout=120,
            )

            respuesta.raise_for_status()

        except requests.Timeout as error:

            raise RuntimeError(
                "El servidor LLM tardó demasiado en responder."
            ) from error

        except requests.ConnectionError as error:

            raise RuntimeError(
                "No se pudo conectar con el servidor LLM.\n"
                f"URL: {url_api}"
            ) from error

        except requests.HTTPError as error:

            cuerpo_error = ""

            try:
                cuerpo_error = respuesta.text
            except Exception:
                pass

            raise RuntimeError(
                "El servidor LLM devolvió un error HTTP.\n"
                f"Código: {respuesta.status_code}\n"
                f"Respuesta: {cuerpo_error}"
            ) from error

        except requests.RequestException as error:

            raise RuntimeError(
                f"Error de conexión con el servidor LLM: {error}"
            ) from error

        # --------------------------------------------------------
        # Procesar respuesta
        # --------------------------------------------------------

        try:

            datos = respuesta.json()

        except ValueError as error:

            raise RuntimeError(
                "El servidor LLM no devolvió JSON válido."
            ) from error

        # --------------------------------------------------------
        # Extraer respuesta estándar
        # --------------------------------------------------------

        try:

            respuesta_modelo = (
                datos["choices"][0]
                ["message"]["content"]
            )

        except (KeyError, IndexError, TypeError):

            # Si el servidor devuelve algo diferente,
            # entregamos el JSON para poder inspeccionarlo.
            return json.dumps(
                datos,
                ensure_ascii=False,
                indent=2,
            )

        return str(respuesta_modelo).strip()

    return motor_llm

# ---------------------------------------------------------------- #
# 5d. AsistenteIA: orquesta chatbot + memoria + herramientas
# ---------------------------------------------------------------- #
class AsistenteIA:
    """
    Núcleo conversacional de GWEN.

    Responsabilidades:
    - Mantener el historial de conversación.
    - Mantener una identidad/personalidad estable.
    - Construir correctamente el contexto para el LLM.
    - Guardar las respuestas en memoria.
    - Permitir herramientas.
    - Mantener un fallback hacia el chatbot local.
    """

    def __init__(
        self,
        nombre="GWEN",
        ruta_memoria=None,
        motor_llm=None,
        umbral_confianza=0.15,
        limite_contexto=20,
    ):
        self.nombre = nombre
        self.motor_llm = motor_llm
        self.umbral_confianza = umbral_confianza

        # Número máximo de mensajes recientes que se envían al modelo.
        self.limite_contexto = limite_contexto

        # Sistema NLP local.
        self.chatbot = Chatbot(nombre=nombre)

        # Memoria persistente.
        self.memoria = MemoriaConversacion(
            limite_turnos=200,
            ruta_persistencia=ruta_memoria,
        )

        # Registro global de herramientas.
        self.herramientas = herramientas

       # Identidad de GWEN.
        self.system_prompt = f"""\
Eres {self.nombre}, un asistente de inteligencia artificial útil, amable y eficiente.

Instrucciones:
1. Responde de forma directa, clara y precisa a las preguntas del usuario.
2. No repitas tu presentación ni te saludes en cada mensaje si la conversación ya inició.
3. Si el usuario te pide una explicación, da una respuesta clara e informativa.

Reglas principales:

1. Mantén el contexto de la conversación.
2. Utiliza los mensajes anteriores para interpretar mensajes cortos o ambiguos.
3. No trates cada mensaje como una conversación independiente.
4. Si el usuario cambia de tema, sigue el nuevo tema sin perder completamente el contexto anterior.
5. No inventes información.
6. Si no sabes algo, dilo claramente.
7. Responde de manera natural y comprensible.
8. Responde en español salvo que el usuario use otro idioma.
9. Evita repetir innecesariamente lo que ya se dijo.
10. Cuando una pregunta dependa de información anterior, utiliza el contexto disponible.
11. No afirmes haber realizado acciones que realmente no realizaste.

Tu nombre es {self.nombre}.
""".strip()

        self._configurar_intenciones_base()

    # ============================================================
    # CONFIGURACIÓN DEL CHATBOT LOCAL
    # ============================================================

    def _configurar_intenciones_base(self):
        self.chatbot.agregar_intencion(
            "saludo",
            [
                "hola",
                "buenas",
                "qué tal",
                "hey",
                "buenos días",
                "buenas tardes",
                "buenas noches",
                "que show",
                "que pasa causa gaa",
                "jarvis jarvis",
                "e w",

            ],
            [
                f"¡Hola! Soy {self.nombre}. ¿En qué puedo ayudarte?",
                "¡Hola! ¿Cómo estás?",
                "¡Buenas! ¿Qué hacemos hoy?",
            ],
        )

        self.chatbot.agregar_intencion(
            "despedida",
            [
                "adiós",
                "hasta luego",
                "nos vemos",
                "chao",
                "me voy",
                "bye",
            ],
            [
                "¡Hasta luego!",
                "Nos vemos pronto.",
                "¡Cuídate!",
            ],
        )

        self.chatbot.agregar_intencion(
            "agradecimiento",
            [
                "gracias",
                "te lo agradezco",
                "muchas gracias",
                "mil gracias",
            ],
            [
                "¡De nada!",
                "Con gusto.",
                "Para eso estoy.",
            ],
        )

        self.chatbot.agregar_intencion(
            "identidad",
            [
                "quién eres",
                "cómo te llamas",
                "qué eres",
                "cuál es tu nombre",
            ],
            [
                f"Soy {self.nombre}, un asistente de inteligencia artificial.",
            ],
        )

        self.chatbot.agregar_intencion(
            "ayuda_herramientas",
            [
                "qué herramientas tienes",
                "qué puedes hacer",
                "qué funciones tienes",
                "ayuda",
                "funciones disponibles",
            ],
            [
                "Tengo disponibles estas herramientas: "
                + ", ".join(self.herramientas.listar().keys())
            ],
        )

        self.chatbot.entrenar()

    # ============================================================
    # INTENCIONES PERSONALIZADAS
    # ============================================================

    def agregar_intencion_personalizada(
        self,
        tag,
        patrones,
        respuestas,
    ):
        self.chatbot.agregar_intencion(
            tag,
            patrones,
            respuestas,
        )

        self.chatbot.entrenar()

        return self

    # ============================================================
    # HERRAMIENTAS
    # ============================================================

    def usar_herramienta(self, nombre, *args, **kwargs):
        return self.herramientas.ejecutar(
            nombre,
            *args,
            **kwargs,
        )

    def herramientas_disponibles(self):
        return self.herramientas.listar()

    # ============================================================
    # CONSTRUCCIÓN DEL CONTEXTO
    # ============================================================

    def construir_contexto(self):
        
        #Convierte la memoria interna en mensajes estructurados
        #compatibles con modelos conversacionales.
        

        mensajes = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        # Tomamos solamente los mensajes recientes.
        turnos = self.memoria.turnos[
            -self.limite_contexto:
        ]

        for turno in turnos:
            rol = turno.get("rol", "")
            texto = turno.get("texto", "")

            if not texto:
                continue

            # Convertimos los nombres internos de NOVA
            # a los roles estándar de un LLM.
            if rol == "usuario":
                role = "user"

            elif rol == "asistente":
                role = "assistant"

            elif rol == "system":
                role = "system"

            elif rol == "tool":
                role = "tool"

            else:
                # Ignorar roles desconocidos.
                continue

            mensajes.append(
                {
                    "role": role,
                    "content": texto,
                }
            )

        return mensajes

    # ============================================================
    # RESPUESTA
    # ============================================================

    def responder(self, texto_usuario):
        
        #Procesa un mensaje y genera una respuesta manteniendo
        #el contexto de la conversación.
        
        texto_usuario = str(texto_usuario).strip()

        if not texto_usuario:
            return "Escribe algo para que pueda responderte."

        # 1. Guardar mensaje del usuario en el historial
        self.memoria.agregar_turno("user", texto_usuario)

        # 2. Si hay un motor LLM (IA real) conectado, generar respuesta con IA
        if self.motor_llm is not None:
            try:
                contexto = self.construir_contexto()
                respuesta = self.motor_llm(contexto)
            except Exception as e:
                respuesta = f"Error al consultar el modelo: {e}"
        else:
            # 3. Si motor_llm es None, recurre al sistema local
            respuesta = self.chatbot.responder(texto_usuario)

        # 4. Guardar respuesta en el historial
        self.memoria.agregar_turno("assistant", respuesta)

        return respuesta

        # --------------------------------------------------------
        # 1. GUARDAR MENSAJE DEL USUARIO
        # --------------------------------------------------------

        self.memoria.agregar_turno(
            "usuario",
            texto_usuario,
        )

        # --------------------------------------------------------
        # 2. GENERAR RESPUESTA
        # --------------------------------------------------------

        if self.motor_llm is not None:

            # Construimos TODO el contexto.
            mensajes = self.construir_contexto()

            # Enviamos mensajes estructurados al LLM.
            respuesta = self.motor_llm(mensajes)

        else:

            # Fallback al chatbot local.
            respuesta = self.chatbot.responder(
                texto_usuario
            )

        # --------------------------------------------------------
        # 3. LIMPIAR RESPUESTA
        # --------------------------------------------------------

        if respuesta is None:
            respuesta = "No pude generar una respuesta."

        respuesta = str(respuesta).strip()

        if not respuesta:
            respuesta = "No pude generar una respuesta."

        # --------------------------------------------------------
        # 4. GUARDAR RESPUESTA
        # --------------------------------------------------------

        self.memoria.agregar_turno(
            "asistente",
            respuesta,
        )

        return respuesta

    # ============================================================
    # HISTORIAL
    # ============================================================

    def obtener_historial(self):
        return list(self.memoria.turnos)

    def limpiar_conversacion(self):
     
        #Borra únicamente el historial conversacional.
        

        self.memoria.turnos = []

        if self.memoria.ruta_persistencia:
            self.memoria.guardar()

    # ============================================================
    # CONSOLA
    # ============================================================

    def chatear_en_consola(self):

        print(
            f"{self.nombre}: ¡Hola! "
            "Escribe 'salir' para terminar.\n"
        )

        while True:

            try:
                entrada = input("Tú: ").strip()

            except (KeyboardInterrupt, EOFError):
                print(
                    f"\n{self.nombre}: ¡Hasta luego!"
                )
                break

            if not entrada:
                continue

            if entrada.lower() in (
                "salir",
                "exit",
                "quit",
            ):
                print(
                    f"{self.nombre}: ¡Hasta luego!"
                )
                break

            if entrada.lower() == "/historial":
                for turno in self.obtener_historial():
                    print(
                        f"{turno['rol']}: "
                        f"{turno['texto']}"
                    )
                continue

            if entrada.lower() == "/limpiar":
                self.limpiar_conversacion()
                print(
                    f"{self.nombre}: Conversación limpiada."
                )
                continue

            try:
                respuesta = self.responder(
                    entrada
                )

                print(
                    f"{self.nombre}: {respuesta}"
                )

            except Exception as error:
                print(
                    f"{self.nombre}: "
                    f"Ocurrió un error: {error}"
                )
# ============================================================================
# SECCIÓN 6: EJEMPLOS / DEMOS DE CADA SISTEMA
# ============================================================================
def ejemplo_xor():
    print("\n" + "=" * 60)
    print("RED NEURONAL — Problema XOR (no se resuelve con una línea recta)")
    print("=" * 60)
    np.random.seed(0)
    X, y = generar_xor(n_muestras=300, ruido=0.1, semilla=0)
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y, proporcion_prueba=0.2, semilla=0)

    red = RedNeuronal()
    red.agregar(CapaDensa(2, 16, "he")).agregar(CapaActivacion("relu"))
    red.agregar(CapaDensa(16, 8, "he")).agregar(CapaActivacion("relu"))
    red.agregar(CapaDensa(8, 1, "xavier")).agregar(CapaActivacion("sigmoide"))
    red.compilar(EntropiaCruzadaBinaria(), Adam(tasa_aprendizaje=0.01))
    red.resumen()
    red.entrenar(X_tr, y_tr, epocas=300, tamano_lote=16, verbose=True, X_val=X_te, y_val=y_te)

    pred = (red.predecir(X_te) >= 0.5).astype(int)
    print(f"\nExactitud en datos de prueba: {exactitud(y_te, pred):.3f}")


def ejemplo_multiclase():
    print("\n" + "=" * 60)
    print("RED NEURONAL — Clasificación multiclase con Softmax (4 grupos)")
    print("=" * 60)
    np.random.seed(1)
    X, y = generar_blobs(n_muestras=400, n_grupos=4, dispersion=1.3, semilla=1)
    X_norm, _, _ = normalizar_estandar(X)
    y_oh = codificar_one_hot(y, n_clases=4)
    X_tr, X_te, y_tr, y_te = dividir_datos(X_norm, y_oh, proporcion_prueba=0.2, semilla=1)

    red = RedNeuronal()
    red.agregar(CapaDensa(2, 32, "he")).agregar(CapaActivacion("relu")).agregar(CapaAbandono(0.1))
    red.agregar(CapaDensa(32, 4, "xavier")).agregar(CapaSoftmax())
    red.compilar(EntropiaCruzadaCategorica(), Adam(tasa_aprendizaje=0.02))
    red.entrenar(X_tr, y_tr, epocas=150, tamano_lote=32, verbose=True)

    pred, real = decodificar_one_hot(red.predecir(X_te)), decodificar_one_hot(y_te)
    print(f"\nExactitud en datos de prueba: {exactitud(real, pred):.3f}")


def ejemplo_guardar_y_cargar():
    print("\n" + "=" * 60)
    print("RED NEURONAL — Guardar y cargar un modelo entrenado")
    print("=" * 60)
    X, y = generar_xor(n_muestras=100, semilla=5)
    red = RedNeuronal()
    red.agregar(CapaDensa(2, 8)).agregar(CapaActivacion("relu"))
    red.agregar(CapaDensa(8, 1)).agregar(CapaActivacion("sigmoide"))
    red.compilar(EntropiaCruzadaBinaria(), Adam(tasa_aprendizaje=0.05))
    red.entrenar(X, y, epocas=100, verbose=False)

    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "red_ejemplo.pkl")
    red.guardar(ruta)
    print(f"Modelo guardado en: {ruta}")
    red_cargada = RedNeuronal.cargar(ruta)
    print("¿Las predicciones coinciden tras cargar el modelo?",
          np.allclose(red.predecir(X[:5]), red_cargada.predecir(X[:5])))


def ejemplo_regresion():
    print("\n" + "=" * 60)
    print("MACHINE LEARNING — Regresión (lineal y polinómica)")
    print("=" * 60)
    X, y = generar_lineal(n_muestras=200, pendiente=3.0, intercepto=-2.0, ruido=1.5, semilla=10)
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y, proporcion_prueba=0.25, semilla=10)

    modelo_lineal = RegresionLineal(tasa_aprendizaje=0.02, iteraciones=800).entrenar(X_tr, y_tr)
    pred = modelo_lineal.predecir(X_te)
    print(f"Regresión lineal   -> pendiente={modelo_lineal.pesos[0]:.3f} (real=3.0)  "
          f"R²={r_cuadrado(y_te, pred):.3f}  RMSE={raiz_error_cuadratico_medio(y_te, pred):.3f}")

    # Normalizar antes de expandir a potencias evita que los gradientes exploten
    X_tr_norm, media, desv = normalizar_estandar(X_tr)
    X_te_norm, _, _ = normalizar_estandar(X_te, media, desv)
    modelo_poli = RegresionPolinomica(grado=3, tasa_aprendizaje=0.01, iteraciones=2000).entrenar(X_tr_norm, y_tr)
    pred_poli = modelo_poli.predecir(X_te_norm)
    print(f"Regresión polinómica (grado 3) -> R²={r_cuadrado(y_te, pred_poli):.3f}")


def ejemplo_clasificacion():
    print("\n" + "=" * 60)
    print("MACHINE LEARNING — 4 algoritmos de clasificación sobre datos 'lunas'")
    print("=" * 60)
    X, y = generar_lunas(n_muestras=300, ruido=0.15, semilla=11)
    X_norm, _, _ = normalizar_estandar(X)
    X_tr, X_te, y_tr, y_te = dividir_datos(X_norm, y, proporcion_prueba=0.25, semilla=11)

    modelos = {
        "Regresión Logística": RegresionLogistica(tasa_aprendizaje=0.3, iteraciones=500),
        "K-Vecinos (k=5)": KVecinosCercanos(k=5),
        "Naive Bayes": NaiveBayesGaussiano(),
        "Árbol de Decisión": ArbolDecision(profundidad_maxima=6),
    }
    for nombre, modelo in modelos.items():
        modelo.entrenar(X_tr, y_tr)
        pred = modelo.predecir(X_te)
        print(f"  {nombre:<22} exactitud={exactitud(y_te, pred):.3f}  f1={puntuacion_f1(y_te, pred):.3f}")


def ejemplo_clustering():
    print("\n" + "=" * 60)
    print("MACHINE LEARNING — Agrupamiento no supervisado con K-Medias")
    print("=" * 60)
    X, y_real = generar_blobs(n_muestras=240, n_grupos=4, dispersion=1.0, semilla=12)
    modelo = KMedias(n_grupos=4, semilla=12).entrenar(X)
    print(f"Inercia final (menor es mejor): {modelo.inercia(X):.2f}")
    for i, centro in enumerate(modelo.centroides):
        print(f"  Grupo {i}: ({centro[0]:.2f}, {centro[1]:.2f})")


def construir_chatbot_tienda():
    # Chatbot de ejemplo para atención al cliente de una tienda ficticia.
    bot = Chatbot(nombre="TiendaBot")
    bot.agregar_intencion("saludo", ["hola", "buenas", "qué tal", "buenos días", "hey"],
                           ["¡Hola! Bienvenido a la tienda. ¿En qué puedo ayudarte?"])
    bot.agregar_intencion("horario", ["cuál es el horario", "a qué hora abren", "a qué hora cierran"],
                           ["Abrimos de lunes a sábado, de 9:00 a 20:00."])
    bot.agregar_intencion("envios", ["hacen envíos", "cuánto tarda el envío", "envían a domicilio"],
                           ["Sí, hacemos envíos a todo el país. Tardan entre 2 y 5 días hábiles."])
    bot.agregar_intencion("devoluciones", ["puedo devolver un producto", "política de devoluciones"],
                           ["Tienes 30 días desde la compra para devoluciones o cambios, con el ticket original."])
    bot.agregar_intencion("despedida", ["adiós", "hasta luego", "gracias, eso es todo", "chao"],
                           ["¡Gracias por tu visita! Que tengas un excelente día."])
    bot.entrenar()
    return bot


def demo_automatica_chatbot(bot):
    print("=" * 60)
    print("CHATBOT — Demo automática")
    print("=" * 60)
    preguntas = [
        "hola, buenas tardes", "a qué hora abren los sábados?", "cuánto tarda en llegar mi pedido?",
        "quiero devolver algo que compré", "cuál es la capital de Perú", "muchas gracias, hasta luego",
    ]
    for pregunta in preguntas:
        print(f"Usuario:   {pregunta}")
        print(f"TiendaBot: {bot.responder(pregunta)}\n")


def demo_automatica_asistente(motor_llm=None):
    print("=" * 60)
    print("ASISTENTE – Demo automática")
    print("=" * 60)
    
    ruta_memoria = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memoria_asistente_demo.json")
    if os.path.exists(ruta_memoria):
        os.remove(ruta_memoria)

    # 1. Instanciar pasándole el motor_llm recibido
    asistente = AsistenteIA(nombre="Gwen", ruta_memoria=ruta_memoria, motor_llm=motor_llm)
    
    # (Opcional) Si no hay LLM, mantiene la intención local como respaldo
    if motor_llm is None:
        asistente.agregar_intencion_personalizada(
            "sobre_python", ["qué es python", "para qué sirve python"],
            ["Python es un lenguaje de programación versátil, muy usado en IA, análisis de datos y automatización."]
        )

    # 2. Imprimir dinámicamente usando el nombre del asistente ({asistente.nombre})
    for mensaje in ["hola", "quién eres", "qué es python", "qué herramientas tienes", "gracias"]:
        print(f"Tú:   {mensaje}")
        print(f"{asistente.nombre}: {asistente.responder(mensaje)}\n")

    print("-" * 60)
    print("Usando herramientas directamente:")
    print("  calculadora('12 * (3 + 2)') ->", asistente.usar_herramienta("calculadora", "12 * (3 + 2)"))
    print("  fecha_hora()                ->", asistente.usar_herramienta("fecha_hora"))
    print("  convertir_temperatura(0,'C','K') ->", asistente.usar_herramienta("convertir_temperatura", 0, "C", "K"))
    print("-" * 60)
    print(f"Conversación guardada en: {ruta_memoria}")


# ============================================================================
# SECCIÓN 7: MENÚ PRINCIPAL INTERACTIVO
# ============================================================================
def menu_red_neuronal():
    print("\n1) XOR   2) Multiclase (Softmax)   3) Guardar/Cargar modelo   0) Volver")
    opcion = input("Elige una opción: ").strip()
    {"1": ejemplo_xor, "2": ejemplo_multiclase, "3": ejemplo_guardar_y_cargar}.get(opcion, lambda: None)()


def menu_machine_learning():
    print("\n1) Regresión   2) Clasificación   3) Clustering   4) Todo   0) Volver")
    opcion = input("Elige una opción: ").strip()
    if opcion == "1":
        ejemplo_regresion()
    elif opcion == "2":
        ejemplo_clasificacion()
    elif opcion == "3":
        ejemplo_clustering()
    elif opcion == "4":
        ejemplo_regresion(); ejemplo_clasificacion(); ejemplo_clustering()


def menu_chatbot():
    print("\n1) Demo automática   2) Chat interactivo   0) Volver")
    opcion = input("Elige una opción: ").strip()
    bot = construir_chatbot_tienda()
    if opcion == "1":
        demo_automatica_chatbot(bot)
    elif opcion == "2":
        bot.chatear_en_consola()


def menu_asistente():
    print("\n1) Demo automática   2) Chat interactivo   0) Volver")
    opcion = input("Elige una opción: ").strip()

    # Intento de creación y prueba del motor Ollama
    try:
        motor_llm = crear_motor_ollama("llama3.2:1b")
        print("[SISTEMA] Conectando con Ollama...")
        motor_llm([{"role": "user", "content": "hola"}])
        print("[SISTEMA] ¡Conexión exitosa con Ollama!")
    except Exception as e:
        print(f"\n[ERROR OLLAMA] No se pudo activar el motor LLM: {e}")
        print("[AVISO] Se usará el chatbot de respaldo por intenciones local.\n")
        motor_llm = None

    if opcion == "1":
        demo_automatica_asistente(motor_llm)
    elif opcion == "2":
        ruta_memoria = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memoria_asistente.json")
        print(f"(La memoria se guardará en: {ruta_memoria})")

        asistente = AsistenteIA(
            nombre="Gwen",
            ruta_memoria=ruta_memoria,
            motor_llm=motor_llm
        )
        asistente.chatear_en_consola()

def main():
    opciones = {
        "1": ("Red neuronal desde cero (deep learning)", menu_red_neuronal),
        "2": ("Machine Learning (regresión / clasificación / clustering)", menu_machine_learning),
        "3": ("Chatbot conversacional (NLP)", menu_chatbot),
        "4": ("Asistente con memoria y herramientas", menu_asistente),
    }
    while True:
        print("\n" + "=" * 60)
        print(" MI_IA — Framework de Inteligencia Artificial propio")
        print("=" * 60)
        for clave, (titulo, _) in opciones.items():
            print(f"  {clave}) {titulo}")
        print("  0) Salir")
        print("-" * 60)

        eleccion = input("Elige un sistema para probar: ").strip()
        if eleccion == "0":
            print("¡Hasta luego!")
            break
        elif eleccion in opciones:
            try:
                opciones[eleccion][1]()
            except KeyboardInterrupt:
                print("\n(Interrumpido por el usuario)")
            except Exception as error:
                print(f"Ocurrió un error: {error}")
        else:
            print("Opción no válida, intenta de nuevo.")


if __name__ == "__main__":
    main()