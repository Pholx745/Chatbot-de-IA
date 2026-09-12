<img align="right" width="250" src="../ASSETS/teto1.jfif" alt="Kasane Teto - Guía del Kosko">
# Guía individual — Joaquín (Ensamblador / líder técnico)

## Tu rol en una frase
No eres quien más código escribe. Eres quien mantiene el repositorio sano, revisa lo que entra a `main`, y entiende el tramo final de la ruta técnica (servidor → WhatsApp) para poder guiar al equipo cuando lleguen ahí.

## Checklist de esta etapa

**1. Configurar el repositorio**
- [ ] Crear un `.gitignore` que excluya `main.build/`, `main.dist/`, `*.exe` y los `memoria_*.json` (datos personales de conversaciones, no deben subirse).
- [ ] Activar protección de la rama `main` en GitHub (Settings → Branches → Branch protection rule): exigir pull request antes de fusionar, y que al menos otra persona apruebe.
- [ ] Definir la convención de nombres de rama del equipo: `feature/lo-que-hace` (ej. `feature/intencion-horarios`).

**2. Aprender a revisar código (aunque no lo escribas)**
- Cuando llegue un pull request: léelo completo, pruébalo corriendo `chatear_en_consola()`, y pregunta "¿por qué lo hiciste así?" cuando algo no esté claro. Eso ya es una revisión válida.

**3. Entender (sin implementar todavía) el tramo final**
- Qué es un webhook y cómo un servidor Flask recibe mensajes de WhatsApp.
- Qué diferencia hay entre el Sandbox de Twilio (gratis, para pruebas) y la API de WhatsApp Business real (requiere verificación de empresa en Meta).
- Qué significa manejar una API key con cuidado (nunca subirla al repo — va en variables de entorno, nunca en el código).

## Recursos para ti

| Tema | Recurso |
|---|---|
| Flujo de ramas del equipo | [GitHub Flow (oficial)](https://docs.github.com/en/get-started/quickstart/github-flow) |
| Plantilla de .gitignore para Python | [gitignore.io / toptal](https://www.toptal.com/developers/gitignore) — genera uno buscando "python" |
| El tramo final que vas a supervisar | [Crear un chatbot de WhatsApp con Python, Flask y Twilio (Twilio, oficial, español)](https://www.twilio.com/es-mx/blog/crear-un-chatbot-de-whatsapp-con-python-flask-y-twilio) |
| Créditos gratis para hosting (cuando llegue el momento de desplegar) | [GitHub Student Developer Pack](https://education.github.com/pack) — con tu correo de Tecsup deberías calificar; incluye créditos de DigitalOcean/Azure y GitHub Copilot gratis |

## Tu prompt de IA (ya lo tienes, recordatorio de para qué sirve)
Úsalo con cualquiera de las IAs de la lista general de recursos cuando necesites pensar en voz alta sobre arquitectura, revisar si una decisión técnica tiene sentido, o entender un concepto antes de explicárselo al equipo.
