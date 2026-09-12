<img align="right" width="250" src="../ASSETS/benja1.jfif" alt="Guía de Benja">
# Guía individual — Diseño conversacional y contenido a cargo de Benjamin o(￣┰￣*)ゞ

## Tu rol en una frase
Diseñar el mapa de conversación (qué pregunta un cliente y cómo responde el bot) y organizar la información del negocio, de forma que se pueda traducir directo a código sin que el programador tenga que adivinar nada.

## Cómo trabajar, paso a paso

**1. Elige el negocio ficticio** (una cevichería, una ferretería, lo que sea, para practicar).

**2. Dibuja el flujo básico**
Usa una herramienta de diagramas: bienvenida → 4-5 ramas de preguntas típicas de WhatsApp (horario, precios, delivery, reclamos) → qué pasa si el bot no entiende (¿deriva a una persona?).

**3. Prototipa el flujo de forma interactiva**
En vez de solo dibujar, arma el mismo flujo en Voiceflow y pruébalo tú mismo simulando ser el cliente. Ahí notas huecos que en un dibujo estático no se ven.

**4. Documenta y entrega**
Captura de pantalla + notas de texto claras (no solo el dibujo) para que el programador lo convierta en `agregar_intencion()` sin tener que preguntarte cosas obvias.

**5. Junta la base de conocimiento del negocio**
Horarios, catálogo, precios, política de devoluciones, preguntas frecuentes reales — esto alimenta tanto tus flujos como el "system prompt" que le da personalidad al bot.

## Recursos para ti

| Para qué | Recurso |
|---|---|
| Dibujar el mapa de conversación | [draw.io (diagrams.net)](https://app.diagrams.net/) — gratis, sin registro |
| Alternativa colaborativa (los tres a la vez) | [Miro — creador de diagramas de flujo](https://miro.com/es/diagrama-de-flujo/) |
| Prototipar sin programar | [Voiceflow — crea chatbots sin código](https://liora.io/es/voiceflow-todo-sobre) |
| Aprender los conceptos base | [Cómo diseñar un flujo de conversación para tu chatbot](https://planetachatbot.com/como-disenar-un-flujo-de-conversacion-para-tu-chatbot-empathybots/) |
| Plantillas específicas para WhatsApp | [Diagramas de flujo para chatbot de WhatsApp: guía + ejemplos](https://crmwhata.com/diagramas-de-flujo-chatbot-whatsapp/) |
| Para profundizar más adelante | [Cursos de Conversational UX Design (CUX Academy)](https://www.cux.academy/) |

## Cómo pedir ayuda a una IA
Pide que te hagan de "cliente difícil" simulando preguntas raras contra tu flujo, para encontrar huecos antes de pasárselo al programador. Eso es más útil que pedir que te generen un flujo genérico de la nada.
