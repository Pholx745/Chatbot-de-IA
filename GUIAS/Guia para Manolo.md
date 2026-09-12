<img align="left" width="250" src="../ASSETS/frieren1.jfif" alt="Guía de Manolo - Frieren">
# Guía individual — Manuel wn :v

## Tu rol en una frase
Convertir el código de `main.py` en un asistente que use un modelo de lenguaje real y que, al final, responda mensajes de WhatsApp de verdad. Cada cosa que practiques debe acercar a Gwen un paso en esa ruta.

## Ruta de aprendizaje, en orden

**Paso 0 — Python sólido (si aún te sientes inseguro)**
- Curso completo desde cero, en video.
- Práctica diaria corta en una plataforma interactiva.

**Paso 1 — Git en equipo (GitHub Flow)**
```
git checkout main
git pull origin main
git checkout -b feature/nombre-de-tu-tarea
# trabajas, pruebas
git push origin feature/nombre-de-tu-tarea
# abres el Pull Request para que lo revisen
```

**Paso 2 — Entender la POO que ya usa Gwen**
Antes de escribir código nuevo, que Claude (u otra IA) te explique método por método las clases `Chatbot`, `AsistenteIA` y `MemoriaConversacion` de `main.py`. Ejercicio: escribe una clase nueva propia (ej. `BaseConocimiento`) copiando ese mismo estilo.

**Paso 3 — Conectar un LLM real**
Reemplazar el sistema de intenciones local por una llamada de verdad a un modelo de lenguaje, escribiendo una función parecida a `crear_motor_ollama()` (línea 68 de `main.py`) pero usando la API de Anthropic.

**Paso 4 — Sacar a Gwen de la consola: un servidor web con Flask**
Primero practica un "hola mundo" en Flask solo, después mete la lógica de `asistente.responder()` adentro.

**Paso 5 — El proyecto final: WhatsApp de verdad**
Seguir el tutorial de Twilio + Flask paso a paso, usando el Sandbox gratuito (no requiere aprobación de Meta todavía).

## Recursos para ti

| Etapa | Recurso |
|---|---|
| Python desde cero | [Curso de Python desde cero (MoureDev, YouTube)](https://www.youtube.com/watch?v=nKPbfIU442g) |
| Práctica diaria con proyectos reales | [Practica Python creando 6 proyectos (freeCodeCamp Español)](https://www.freecodecamp.org/espanol/news/practica-python-creando-6-proyectos-curso-gratis-paso-a-paso/) |
| Ejercicios con mentoría humana | [Python en Exercism](https://exercism.org/tracks/python/exercises) |
| Git en equipo | [GitHub Flow (oficial)](https://docs.github.com/en/get-started/quickstart/github-flow) |
| Conectar la API de Claude | [Guía de inicio rápido de la API de Claude (oficial, español)](https://docs.anthropic.com/es/docs/quickstart-guide) |
| El proyecto final — WhatsApp con Python | [Crear un chatbot de WhatsApp con Python, Flask y Twilio (Twilio, oficial, español)](https://www.twilio.com/es-mx/blog/crear-un-chatbot-de-whatsapp-con-python-flask-y-twilio) |
| Mismo tutorial en video | [Building a WhatsApp Chatbot with Twilio and Python (YouTube, 2026)](https://www.youtube.com/watch?v=dsJgse8gc2o) |

## Cómo pedir ayuda a una IA (regla de oro)
No pidas la solución completa de una vez. Pide que te expliquen antes de tocar el código, pide ejercicios chiquitos, y pega siempre el error completo cuando algo falle — no "no funciona".
